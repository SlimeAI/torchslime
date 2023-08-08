from typing import Dict, Sequence, Union, List, Callable, Any
from torchslime.utils import IterTool, safe_divide, type_cast, \
    terminal as Cursor
from torchslime.utils.bases import BaseList, is_none_or_nothing
from torchslime.utils.decorators import CallDebug
from torchslime.utils.formatter import progress_format, eta_format
from torchslime.core.context.base import BaseContext
from torchslime.core.handlers import Handler, HandlerContainer, H_SEQ
from torchslime.log import logger
from torch import set_grad_enabled
from functools import wraps


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @wraps(func)
    def grad_switch(self, ctx: BaseContext):
        # only when context status is in ['TRAIN'] is the grad enabled
        with set_grad_enabled(str(ctx.hook_ctx.state) in ['TRAIN']):
            func(self, ctx)
    return grad_switch


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torchslime.core.handler.Handler): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='EmptyHandler')
    def handle(self, _: BaseContext):
        """do nothing"""
        pass


# lambda or sequence of lambdas
L_SEQ = Union[Callable[[BaseContext], Any], Sequence[Callable[[BaseContext], Any]]]


class LambdaHandler(Handler, BaseList):
    
    def __init__(self, _lambda: L_SEQ, *args, **kwargs):
        Handler.__init__(self, *args, **kwargs)
        BaseList.__init__(self, _lambda)
    
    def handle(self, ctx: BaseContext):
        # execute lambda functions
        for _lambda in self.get_list__():
            _lambda(ctx)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)

    @CallDebug(module_name='EpochIterationHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('iteration_ctx.total_epochs', silent=False)
        # epoch loops
        for current in range(ctx.iteration_ctx.total_epochs):
            # set current epoch to the context
            ctx.iteration_ctx.current_epoch = current
            # output epoch info. TODO: change logger operation to a handler?
            logger.log('Epoch {}\n'.format(ctx.iteration_ctx.current_epoch + 1))
            super().handle(ctx)


class IterationHandler(HandlerContainer):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)

    @CallDebug(module_name='IterationHandler')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        loader = ctx.hook_ctx.state.get_loader(ctx)
        # loader check
        if is_none_or_nothing(loader):
            logger.warn('Got empty data loader.')
            return
        
        for batch, progress, time, current, total in IterTool(loader, True, True, True, True):
            ctx.step_ctx.from_dict__({
                'batch': batch,  # original batch data of the dataset
                'progress': progress,  # progress of iteration(includes current step and total steps)
                'time': time,  # time of the iter(current time)
                'current': current,  # the current step
                'total': total  # total steps of iteration
            })
            # current global step increases by 1
            ctx.iteration_ctx.current_step += 1
            # carry out the subsequent actions
            super().handle(ctx)


class StepIterationHandler(HandlerContainer):
    # TODO: step iteration
    @CallDebug(module_name='StepIterationHandler')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        return super().handle(ctx)


class ForwardHandler(Handler):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @CallDebug(module_name='ForwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model',
            'device',
            'run.data_parser',
            'step'
        ], silent=False)
        # forward
        x, y_true, extra = ctx.run_ctx.data_parser(ctx)
        y_pred = ctx.model(type_cast(x, ctx.device))
        y_true = type_cast(y_true, ctx.device)
        # clone and update context info
        ctx.step_ctx.from_dict__({
            # the result of the forward progress
            'x': x,
            'y_true': y_true,
            'y_pred': y_pred,
            'extra': extra
        })


class LossHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='LossHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check('run.loss_func') is True:
            # compute loss
            loss = ctx.run_ctx.loss_func(ctx.step_ctx.y_pred, ctx.step_ctx.y_true)
            ctx.step_ctx.loss = loss
            ctx.step_ctx.loss_value = self._parse_float(ctx.run_ctx.loss_wrapper.get_copy(loss)).decode()
    
    def _parse_float(self, loss_dict):
        for key in loss_dict:
            loss_dict[key] = float(loss_dict[key])
        return loss_dict


class BackwardHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @CallDebug(module_name='BackwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check(['step.loss']) is True:
            last = ctx.step_ctx.total % ctx.run_ctx.grad_acc
            grad_acc = ctx.run_ctx.grad_acc if (ctx.step_ctx.total - ctx.step_ctx.current - 1) >= last else last
            # backward
            (ctx.run_ctx.loss_reduction(ctx) / grad_acc).backward()


class OptimizerHandler(HandlerContainer):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)
    
    @CallDebug(module_name='OptimizerHandler')
    def handle(self, ctx: BaseContext):
        # backward handler
        super().handle(ctx)
        if ctx.ctx_check(['run.optimizer']) is True and \
            ((ctx.step_ctx.current + 1) % ctx.run_ctx.grad_acc == 0 or ctx.step_ctx.current + 1 == ctx.step_ctx.total):
            ctx.run_ctx.optimizer.step()
            ctx.run_ctx.optimizer.zero_grad()


class MetricsHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='MetricsHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('step', silent=False)
        if ctx.ctx_check('run.metrics') is True:
            ctx.step_ctx.metrics = ctx.run_ctx.metrics(ctx)


class GatherAverageHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='GatherAverageHandler')
    def handle(self, ctx: BaseContext):
        from torchslime.core.context import Context
        from torchslime.components.metric import LossWrapper
        ctx: Context
        torch_comm = ctx.distributed_ctx.torch_comm
        # gather data
        gathered_loss_values: List[LossWrapper] = \
            torch_comm.all_gather_object(ctx.run_ctx.loss_wrapper.get_copy(ctx.step_ctx.loss_value))
        gathered_metrics: List[Dict] = torch_comm.all_gather_object(ctx.step_ctx.metrics)
        
        """
        Compute average loss values.
        """
        loss_value = ctx.run_ctx.loss_wrapper.get(self._avg_dict(gathered_loss_values))
        # if and only if all gathered loss values wrapped, is ``loss_values.__wrapped`` is True
        loss_value.set_wrapped(all(gathered_loss_value.get_wrapped() for gathered_loss_value in gathered_loss_values))
        ctx.step_ctx.loss_value = loss_value.decode()
        
        """
        Compute average metrics.
        """
        ctx.step_ctx.metrics = self._avg_dict(gathered_metrics)
    
    def _avg_dict(self, dict_list) -> dict:
        item_dict = {}
        item_count = {}
        # iter every dict
        for _dict in dict_list:
            # iter every dict item
            for key, value in _dict.items():
                # sum dict value
                item_dict.setdefault(key, 0)
                item_dict[key] += value
                # count occurrences of dict items
                item_count.setdefault(key, 0)
                item_count[key] += 1
        # compute average values
        for key, value in item_dict.items():
            item_dict[key] = safe_divide(value, item_count.get(key, 0))
        return item_dict


class AverageInitHandler(Handler):
    
    # inner context key
    INNER_KEY = 'AVERAGE_INNER'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='AverageInitHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook_ctx.state.init_avg_inner_ctx(ctx, self.INNER_KEY)
        # reset avg info
        ctx.hook_ctx.state.clear_avg_info(ctx, self.INNER_KEY)


class AverageHandler(Handler):

    # inner context key
    INNER_KEY = 'AVERAGE_INNER'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='AverageHandler')
    def handle(self, ctx: BaseContext):
        from torchslime.components.metric import LossWrapper
        # get inner context variables
        summary = ctx.hook_ctx.state.get_avg_inner_ctx(ctx, self.INNER_KEY)
        
        """
        Get average loss and metrics.
        """
        loss_value: LossWrapper = ctx.run_ctx.loss_wrapper.get(ctx.step_ctx.loss_value)
        summary_loss_value: LossWrapper = summary['loss_value']
        # update wrapped
        summary_loss_value.set_wrapped(summary_loss_value.get_wrapped() and loss_value.get_wrapped())
        summary_loss_value_count: dict = summary['loss_value_count']
        avg_loss = self._compute_avg(
            loss_value, summary_loss_value, summary_loss_value_count
        )
        avg_loss: LossWrapper = ctx.run_ctx.loss_wrapper.get(avg_loss)
        avg_loss.set_wrapped(summary_loss_value.get_wrapped())
        
        avg_metrics = self._compute_avg(
            ctx.step_ctx.metrics, summary['metrics'], summary['metrics_count']
        )
        ctx.hook_ctx.state.set_avg_loss_value_and_metrics(ctx, avg_loss.decode(), avg_metrics)

    def _compute_avg(self, item_dict: dict, value_dict: dict, count_dict: dict):
        result = {}
        for key, value in item_dict.items():
            value_dict.setdefault(key, 0)
            value_dict[key] += value
            count_dict.setdefault(key, 0)
            count_dict[key] += 1
        for key, value in value_dict.items():
            result[key] = safe_divide(value, count_dict.get(key, 0))
        return result


class DisplayHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='DisplayHandler')
    def handle(self, ctx: BaseContext):
        current = ctx.step_ctx.current
        total = ctx.step_ctx.total

        loss_value, metrics = ctx.hook_ctx.state.get_avg_loss_value_and_metrics(ctx)
        data = {**loss_value, **metrics}
        data = ' - '.join(
            list(map(lambda item: '{0}: {1:.5f}'.format(*item), data.items()))
        )

        with Cursor.cursor_invisible():
            Cursor.refresh_print(
                str(ctx.hook_ctx.state),
                # progress bar
                progress_format(ctx.step_ctx.progress, newline=False),
                # eta with color blue
                '{0}ETA: {1}{2}'.format(
                    Cursor.single_color('b'),
                    eta_format(ctx.step_ctx.time, total - current - 1),
                    Cursor.reset_style()
                ),
                # loss and metrics output
                data,
                # print new line if progress end
                end='\n' if current + 1 == total else ''
            )


class LRDecayHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug(module_name='LRDecayHandler')
    def handle(self, ctx: BaseContext):
        if ctx.ctx_check(['run.lr_decay']) is True:
            ctx.run_ctx.lr_decay.step()

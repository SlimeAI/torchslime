from typing import Dict, Sequence, Union, List, Callable, Any
from torchslime.utils import BaseList, IterTool, safe_divide, type_cast, \
    CallDebug, SmartWraps, terminal as Cursor
from torchslime.utils.formatter import progress_format, eta_format
from torchslime.core.context.base import BaseContext
from torchslime.core.handlers import Handler, HandlerContainer, H_SEQ
from torchslime.log import logger
from torch import set_grad_enabled


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @SmartWraps(func)
    def grad_switch(self, ctx: BaseContext):
        # only when context status is in ['TRAIN'] is the grad enabled
        with set_grad_enabled(str(ctx.hook.state) in ['TRAIN']):
            func(self, ctx)
    return grad_switch


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torchslime.core.handler.Handler): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('EmptyHandler')
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
        for _lambda in self:
            _lambda(ctx)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)

    @CallDebug('EpochIterationHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('epoch.total', silent=False)
        # epoch loops
        for current in range(ctx.iteration.total_epochs):
            # set current epoch to the context
            ctx.iteration.current_epoch = current
            # output epoch info. TODO: change logger operation to a handler?
            logger.log('Epoch {}\n'.format(ctx.iteration.current_epoch + 1))
            super().handle(ctx)


class IterationHandler(HandlerContainer):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)

    @CallDebug('IterationHandler')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check('run.dataset') is True:
            for batch, progress, time, current, total in IterTool(ctx.run.dataset, True, True, True, True):
                ctx.step.from_dict__({
                    'batch': batch, # original batch data of the dataset
                    'progress': progress, # progress of iteration(includes current step and total steps)
                    'time': time, # time of the iter(current time)
                    'current': current, # the current step
                    'total': total # total steps of iteration
                })
                # carry out the subsequent actions
                super().handle(ctx)


class ForwardHandler(Handler):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @CallDebug('ForwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model',
            'device',
            'run.data_parser',
            'step'
        ], silent=False)
        # forward
        x, y_true, extra = ctx.run.data_parser(ctx)
        y_pred = ctx.model(type_cast(x, ctx.device))
        y_true = type_cast(y_true, ctx.device)
        # clone and update context info
        ctx.step.from_dict__({
            # the result of the forward progress
            'x': x,
            'y_true': y_true,
            'y_pred': y_pred,
            'extra': extra
        })


class LossHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('LossHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check('run.loss_func') is True:
            # compute loss
            loss = ctx.run.loss_func(ctx.step.y_pred, ctx.step.y_true)
            ctx.step.loss = loss
            ctx.step.loss_value = self._parse_float(ctx.run.loss_wrapper.get_copy(loss)).decode()
    
    def _parse_float(self, loss_dict):
        for key in loss_dict:
            loss_dict[key] = float(loss_dict[key])
        return loss_dict


class BackwardHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @CallDebug('BackwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check(['step.loss']) is True:
            last = ctx.step.total % ctx.run.grad_acc
            grad_acc = ctx.run.grad_acc if (ctx.step.total - ctx.step.current - 1) >= last else last
            # backward
            (ctx.run.loss_reduction(ctx) / grad_acc).backward()


class OptimizerHandler(HandlerContainer):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)
    
    @CallDebug('OptimizerHandler')
    def handle(self, ctx: BaseContext):
        # backward handler
        super().handle(ctx)
        if ctx.ctx_check(['run.optimizer']) is True and \
            ((ctx.step.current + 1) % ctx.run.grad_acc == 0 or ctx.step.current + 1 == ctx.step.total):
            ctx.run.optimizer.step()
            ctx.run.optimizer.zero_grad()


class MetricsHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('MetricsHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('step', silent=False)
        if ctx.ctx_check('run.metrics') is True:
            ctx.step.metrics = ctx.run.metrics(ctx)


class GatherAverageHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('GatherAverageHandler')
    def handle(self, ctx: BaseContext):
        from torchslime.core.context import Context
        from torchslime.components.metric import LossWrapper
        ctx: Context = ctx
        torch_comm = ctx.distributed.torch_comm
        # gather data
        gathered_loss_values: List[LossWrapper] = \
            torch_comm.all_gather_object(ctx.run.loss_wrapper.get_copy(ctx.step.loss_value))
        gathered_metrics: List[Dict] = torch_comm.all_gather_object(ctx.step.metrics)
        
        """
        Compute average loss values.
        """
        loss_value = ctx.run.loss_wrapper.get(self._avg_dict(gathered_loss_values))
        # if and only if all gathered loss values wrapped, is ``loss_values.__wrapped`` is True
        loss_value.set_wrapped(all(gathered_loss_value.get_wrapped() for gathered_loss_value in gathered_loss_values))
        ctx.step.loss_value = loss_value.decode()
        
        """
        Compute average metrics.
        """
        ctx.step.metrics = self._avg_dict(gathered_metrics)
    
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
    
    @CallDebug('AverageInitHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook.state.init_avg_inner_ctx(ctx, self.INNER_KEY)
        # reset avg info
        ctx.hook.state.clear_avg_info(ctx, self.INNER_KEY)


class AverageHandler(Handler):

    # inner context key
    INNER_KEY = 'AVERAGE_INNER'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('AverageHandler')
    def handle(self, ctx: BaseContext):
        from torchslime.components.metric import LossWrapper
        # get inner context variables
        summary = ctx.hook.state.get_avg_inner_ctx(ctx, self.INNER_KEY)
        
        """
        Get average loss and metrics.
        """
        loss_value: LossWrapper = ctx.run.loss_wrapper.get(ctx.step.loss_value)
        summary_loss_value: LossWrapper = summary['loss_value']
        # update wrapped
        summary_loss_value.set_wrapped(summary_loss_value.get_wrapped() and loss_value.get_wrapped())
        summary_loss_value_count: dict = summary['loss_value_count']
        avg_loss = self._compute_avg(
            loss_value, summary_loss_value, summary_loss_value_count
        )
        avg_loss: LossWrapper = ctx.run.loss_wrapper.get(avg_loss)
        avg_loss.set_wrapped(summary_loss_value.get_wrapped())
        
        avg_metrics = self._compute_avg(
            ctx.step.metrics, summary['metrics'], summary['metrics_count']
        )
        ctx.hook.state.set_avg_loss_value_and_metrics(ctx, avg_loss.decode(), avg_metrics)

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
    
    @CallDebug('DisplayHandler')
    def handle(self, ctx: BaseContext):
        current = ctx.step.current
        total = ctx.step.total

        loss_value, metrics = ctx.hook.state.get_avg_loss_value_and_metrics(ctx)
        data = {**loss_value, **metrics}
        data = ' - '.join(
            list(map(lambda item: '{0}: {1:.5f}'.format(*item), data.items()))
        )

        with Cursor.cursor_invisible():
            Cursor.refresh_print(
                str(ctx.hook.state),
                # progress bar
                progress_format(ctx.step.progress, newline=False),
                # eta with color blue
                '{0}ETA: {1}{2}'.format(
                    Cursor.single_color('b'),
                    eta_format(ctx.step.time, total - current - 1),
                    Cursor.reset_style()
                ),
                # loss and metrics output
                data,
                # print new line if progress end
                end='\n' if current + 1 == total else ''
            )


class DatasetHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('DatasetHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('status', silent=False)
        # get dataset through status
        ctx.hook.state.get_dataset(ctx)


class StateHandler(Handler):

    def __init__(self, state: str = 'train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get status supported
        from torchslime.core.hooks.state import ctx_state
        mode_supported = list(ctx_state.keys())
        if state not in mode_supported:
            logger.warn('An unsupported status is set, this may cause some problems.')
        self.state = state
    
    @CallDebug('StatusHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model'
        ], silent=False)
        # set status to the context
        from torchslime.core.hooks.state import ctx_state
        ctx.hook.state = ctx_state.get(self.state)()
        # change pytorch model mode
        ctx.hook.state.set_model_mode(ctx)
    
    def _get_display_attrs(self) -> dict:
        custom_attrs = {
            'state': 'state'
        }
        return {
            **super()._get_display_attrs(),
            **custom_attrs
        }


class LRDecayHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @CallDebug('LRDecayHandler')
    def handle(self, ctx: BaseContext):
        if ctx.ctx_check(['run.lr_decay']) is True:
            ctx.run.lr_decay.step()

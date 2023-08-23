from torchslime.utils.typing import (
    Dict,
    Sequence,
    Union,
    List,
    Callable,
    Iterable,
    Any,
    Mapping
)
from torchslime.utils import (
    IterTool,
    type_cast,
    terminal as Cursor,
    inf_enumerate
)
from torchslime.utils.bases import BaseList, is_none_or_nothing, Nothing, NOTHING
from torchslime.components.metric import MeterDict
from torchslime.utils.decorators import CallDebug
from torchslime.utils.formatter import progress_format, eta_format
from torchslime.core.context.base import BaseContext
from torchslime.core.handlers import Handler, HandlerContainer
from torchslime.log import logger
from torch import set_grad_enabled
from functools import wraps

# TODO: all
# __all__ = [
#     'TorchGrad'
# ]


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
    
    @CallDebug(module_name='EmptyHandler')
    def handle(self, _: BaseContext):
        """do nothing"""
        pass


class LambdaHandler(Handler, BaseList[Callable[[BaseContext], None]]):
    
    def __init__(self, _lambda: Iterable[Callable[[BaseContext], None]]):
        Handler.__init__(self)
        BaseList.__init__(self, _lambda)
    
    def handle(self, ctx: BaseContext):
        # execute lambda functions
        for _lambda in self:
            _lambda(ctx)


class EpochIterationHandler(HandlerContainer):
    """
    Train Only
    """

    @CallDebug(module_name='EpochIterationHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('iteration_ctx.total', silent=False)
        # epoch loops
        for current in range(ctx.iteration_ctx.start, ctx.iteration_ctx.total):
            # set current epoch to the context
            ctx.iteration_ctx.current = current
            # output epoch info. TODO: change logger operation to a handler?
            logger.log(f'Epoch {ctx.iteration_ctx.current + 1}\n')
            super().handle(ctx)


class IterationHandler(HandlerContainer):

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
            # carry out the subsequent actions
            super().handle(ctx)


class StepIterationHandler(HandlerContainer):
    """
    Train Only
    """
    
    @CallDebug(module_name='StepIterationHandler')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        loader = ctx.hook_ctx.state.get_loader(ctx)
        # loader check
        if is_none_or_nothing(loader):
            logger.warn('Got empty data loader.')
            return
        
        total = ctx.iteration_ctx.total
        for (step, batch), time in IterTool(inf_enumerate(loader, start=ctx.iteration_ctx.start), time=True):
            # current global step increases by 1
            ctx.iteration_ctx.current = step

            ctx.step_ctx.from_dict__({
                'batch': batch,  # original batch data of the dataset
                'progress': (step, total),  # progress of iteration(includes current step and total steps)
                'time': time,  # time of the iter(current time)
                'current': step,  # the current step
                'total': total  # total steps of iteration
            })
            # carry out the subsequent actions
            super().handle(ctx)
            # break if finish
            if step + 1 >= total:
                break


class ForwardHandler(Handler):

    @CallDebug(module_name='ForwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model',
            'device',
            'run_ctx.data_parser',
            'step_ctx'
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
    
    @CallDebug(module_name='LossHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check('run_ctx.loss_func') is True:
            # compute loss
            loss = ctx.run_ctx.loss_func(ctx)
            ctx.step_ctx.loss = loss
            ctx.step_ctx.loss_values = self._parse_float(dict(loss))
    
    def _parse_float(self, loss_dict):
        for key in loss_dict:
            loss_dict[key] = float(loss_dict[key])
        return loss_dict


class BackwardHandler(Handler):

    @CallDebug(module_name='BackwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check(['step_ctx.loss']):
            last = ctx.step_ctx.total % ctx.run_ctx.grad_acc
            grad_acc = ctx.run_ctx.grad_acc if (ctx.step_ctx.total - ctx.step_ctx.current - 1) >= last else last
            # backward
            (ctx.run_ctx.loss_reduction(ctx) / grad_acc).backward()


class OptimizerHandler(HandlerContainer):
    
    @CallDebug(module_name='OptimizerHandler')
    def handle(self, ctx: BaseContext):
        # backward handler
        super().handle(ctx)
        if ctx.ctx_check(['run_ctx.optimizer']) and \
            ((ctx.step_ctx.current + 1) % ctx.run_ctx.grad_acc == 0 or ctx.step_ctx.current + 1 == ctx.step_ctx.total):
            ctx.run_ctx.optimizer.step()
            ctx.run_ctx.optimizer.zero_grad()


class MetricsHandler(Handler):
    
    @CallDebug(module_name='MetricsHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('step_ctx', silent=False)
        if ctx.ctx_check('run_ctx.metrics'):
            ctx.step_ctx.metrics = ctx.run_ctx.metrics(ctx)


class GatherAverageHandler(Handler):
    
    @CallDebug(module_name='GatherAverageHandler')
    def handle(self, ctx: BaseContext):
        dist_comm = ctx.hook_ctx.launch.dist_comm
        # gather data
        gathered_loss_values: List[Dict] = dist_comm.all_gather_object(ctx.step_ctx.loss_values)
        gathered_metrics: List[Dict] = dist_comm.all_gather_object(ctx.step_ctx.metrics)
        
        # Compute average loss values and metrics.
        ctx.step_ctx.loss_values = self._avg_dict(gathered_loss_values)
        ctx.step_ctx.metrics = self._avg_dict(gathered_metrics)
    
    def _avg_dict(self, dicts: List[Mapping]):
        meter_dict = MeterDict()
        for _dict in dicts:
            meter_dict(_dict)
        return meter_dict.get__('mean')


class MeterInitHandler(Handler):
    
    @CallDebug(module_name='MeterInitHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook_ctx.state.init_meter(ctx)


class MeterHandler(Handler):
    
    @CallDebug(module_name='MeterHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook_ctx.state.update_meter(ctx, ctx.step_ctx.loss_values, ctx.step_ctx.metrics)


class DisplayHandler(Handler):

    def __init__(self):
        # set default exec ranks to [0]
        from . import ExecRanks
        self.metadata__ = ExecRanks([0]) | self.metadata__
        super().__init__()
    
    @CallDebug(module_name='DisplayHandler')
    def handle(self, ctx: BaseContext):
        current = ctx.step_ctx.current
        total = ctx.step_ctx.total

        loss_values, metrics = ctx.hook_ctx.state.get_meter(ctx)
        data = {**loss_values.get__('mean'), **metrics.get__('mean')}
        data = ' - '.join(
            list(map(lambda item: f'{item[0]}: {item[1]:.5f}', data.items()))
        )

        with Cursor.cursor_invisible():
            eta_color = Cursor.single_color('b')
            eta_content = eta_format(ctx.step_ctx.time, total - current - 1)
            reset_color = Cursor.reset_style()
            
            Cursor.refresh_print(
                str(ctx.hook_ctx.state),
                # progress bar
                progress_format(ctx.step_ctx.progress, newline=False),
                # eta with color blue
                f'{eta_color}ETA: {eta_content}{reset_color}',
                # loss and metrics output
                data,
                # print new line if progress end
                end='\n' if current + 1 == total else ''
            )


class LRDecayHandler(Handler):
    
    @CallDebug(module_name='LRDecayHandler')
    def handle(self, ctx: BaseContext):
        if ctx.ctx_check(['run_ctx.lr_decay']) is True:
            ctx.run_ctx.lr_decay.step()

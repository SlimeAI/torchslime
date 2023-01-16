from abc import abstractmethod
from typing import Dict, Sequence, Union
from torchslime.util import BaseList, IterTool, NOTHING, is_nothing, safe_divide, type_cast, \
    InvocationDebug, SmartWrapper, terminal as Cursor
from torchslime.util.formatter import progress_format, eta_format
from torchslime.core.context import Context
from torchslime.core import DistributedProxy
from torchslime.log import logger
from torchslime.util.type import INT_SEQ_N
from torch import set_grad_enabled


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @SmartWrapper(func)
    def grad_switch(self, ctx: Context):
        # only when context status is in ['TRAIN'] is the grad enabled
        with set_grad_enabled(str(ctx.status) in ['TRAIN']):
            func(self, ctx)
    return grad_switch


class Handler:
    """Base class for all handlers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, ctx: Context):
        pass

    def __call__(self, ctx: Context):
        self.handle(ctx)


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torchslime.core.handler.Handler): _description_
    """

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EmptyHandler')
    def handle(self, _: Context):
        """do nothing"""
        pass


class DistributedHandler(Handler):

    def __init__(self, exec_ranks: INT_SEQ_N = None):
        super().__init__()
        self.exec_ranks = BaseList.create(exec_ranks)

    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        if self.exec_ranks is None or exec_ranks is None:
            # the exec_ranks are changeable
            self.exec_ranks = BaseList.create(exec_ranks)

    def __call__(self, ctx: DistributedProxy):
        rank = ctx.get_rank()
        if self.exec_ranks is not None and \
            (is_nothing(self.exec_ranks) or rank in self.exec_ranks):
            super().__call__(ctx)


class DistributedHandlerWrapper(DistributedHandler):

    def __init__(
        self,
        wrapped_handler: Handler,
        exec_ranks: INT_SEQ_N = None
    ):
        super().__init__(exec_ranks)
        self._wrapped_handler = wrapped_handler
    
    def handle(self, ctx: Context):
        self._wrapped_handler(ctx)


# handler or sequence of handlers
C_SEQ = Union[Handler, Sequence[Handler]]


class HandlerContainer(Handler, BaseList):

    def __init__(self, handlers: C_SEQ = None):
        Handler.__init__(self)
        BaseList.__init__(self, handlers)
    
    def handle(self, ctx: Context):
        for handler in self:
            handler(ctx)


class DistributedHandlerContainer(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None, default_exec_ranks: INT_SEQ_N = None):
        super().__init__(handlers)
        # the distributed handler container is always executed.
        self.exec_ranks = NOTHING
        # exec ranks that are set to its sub-handlers
        self.default_exec_ranks = BaseList.create(default_exec_ranks)
    
    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        if self.default_exec_ranks is None or exec_ranks is None:
            # the default_exec_ranks are changeable
            self.default_exec_ranks = BaseList.create(exec_ranks)
        for handler in filter(lambda item: isinstance(item, DISTRIBUTED_CLASSES), self):
            handler.set_exec_ranks(self.default_exec_ranks)

    def __call__(self, ctx: DistributedProxy):
        rank = ctx.get_rank()
        if self.exec_ranks is not None and \
            (is_nothing(self.exec_ranks) or rank in self.exec_ranks):
            super().__call__(ctx)


class DistributedHandlerContainerWrapper(DistributedHandlerContainer):

    def __init__(self, wrapped_handler_container: HandlerContainer, default_exec_ranks: INT_SEQ_N = None):
        super().__init__(None, default_exec_ranks)
        self._wrapped_handler_container = wrapped_handler_container
        self._BaseList__list = wrapped_handler_container._BaseList__list

    def handle(self, ctx: Context):
        self._wrapped_handler_container(ctx)


DISTRIBUTED_CLASSES = (DistributedHandler, DistributedHandlerContainer)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('EpochIterationHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check('epoch.total', silent=False)
        # epoch loops
        for current in range(ctx.epoch.total):
            # set current epoch to the context
            ctx.epoch.current = current
            # output epoch info. TODO: change logger operation to a handler?
            logger.log('Epoch %d' % (ctx.epoch.current + 1))
            super().handle(ctx)


class DistributedEpochIterationHandler(DistributedHandlerContainerWrapper):

    def __init__(self, handlers: C_SEQ = None, exec_ranks: INT_SEQ_N = None):
        wrapped_handler_container = EpochIterationHandler(handlers)
        super().__init__(wrapped_handler_container, exec_ranks)


class IterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('IterationHandler')
    @TorchGrad
    def handle(self, ctx: Context):
        # context check
        if ctx.ctx_check('dataset') is True:
            for batch, progress, time, current, total in IterTool(ctx.dataset, True, True, True, True):
                ctx.step.from_dict({
                    'batch': batch, # original batch data of the dataset
                    'progress': progress, # progress of iteration(includes current step and total steps)
                    'time': time, # time of the iter(current time)
                    'current': current, # the current step
                    'total': total # total steps of iteration
                })
                # carry out the subsequent actions
                super().handle(ctx)


class DistributedIterationHandler(DistributedHandlerContainerWrapper):

    def __init__(self, handlers: C_SEQ = None, default_exec_ranks: INT_SEQ_N = None):
        wrapped_handler_container = IterationHandler(handlers)
        super().__init__(wrapped_handler_container, default_exec_ranks)


class ForwardHandler(Handler):
    
    def __init__(self):
        super().__init__()

    @InvocationDebug('ForwardHandler')
    def handle(self, ctx: Context):
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
        ctx.step.from_dict({
            # the result of the forward progress
            'x': x,
            'y_true': y_true,
            'y_pred': y_pred,
            'extra': extra
        })


class LossHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('LossHandler')
    def handle(self, ctx: Context):
        # context check
        if ctx.ctx_check('run.loss') is True:
            # compute loss
            loss = ctx.run.loss(ctx.step.y_pred, ctx.step.y_true)
            ctx.step.loss = loss


class BackwardHandler(Handler):

    def __init__(self):
        super().__init__()

    @InvocationDebug('BackwardHandler')
    def handle(self, ctx: Context):
        # context check
        if ctx.ctx_check(['step.loss']) is True:
            last = ctx.step.total % ctx.run.grad_acc
            grad_acc = ctx.run.grad_acc if (ctx.step.total - ctx.step.current - 1) >= last else last
            # backward
            (ctx.step.loss / grad_acc).backward()


class OptimizerHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)
    
    @InvocationDebug('OptimizerHandler')
    def handle(self, ctx: Context):
        # backward handler
        super().handle(ctx)
        if ctx.ctx_check(['run.optimizer']) is True and \
            ((ctx.step.current + 1) % ctx.run.grad_acc == 0 or ctx.step.current + 1 == ctx.step.total):
            ctx.run.optimizer.step()
            ctx.run.optimizer.zero_grad()


class MetricsHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('MetricsHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check('step', silent=False)
        if ctx.ctx_check('run.metrics') is True:
            ctx.step.metrics = ctx.run.metrics(ctx)


class MetricsGatherHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('MetricsGatherHandler')
    def handle(self, ctx: Context):
        return super().handle(ctx)


# TODO: implementation to be optimized
class AverageHandler(Handler):

    # inner context key
    INNER_KEY = 'AVERAGE_INNER'

    def __init__(self, type: str = 'avg'):
        super().__init__()
        type_supported = ['avg', 'clear']
        if type not in type_supported:
            logger.warn('An unsupported average handler type is set.')
        self.type = type
    
    @InvocationDebug('AverageHandler')
    def handle(self, ctx: Context):
        ctx.status.init_avg_inner_ctx(ctx, self.INNER_KEY)
        if self.type == 'avg':
            self.average(ctx)
        elif self.type == 'clear':
            self.clear(ctx)

    def average(self, ctx: Context):
        # get inner context variables
        summary = ctx.status.get_avg_inner_ctx(ctx, self.INNER_KEY)
        # get average loss and metrics
        avg_loss = self._compute_avg_loss(summary, ctx.step.loss)
        avg_metrics = self._compute_avg_metrics(summary, ctx.step.metrics)
        ctx.status.set_avg_loss_and_metrics(ctx, avg_loss, avg_metrics)

    def clear(self, ctx: Context):
        # reset avg info
        ctx.status.clear_avg_info(ctx, self.INNER_KEY)

    @staticmethod
    def _compute_avg_loss(summary, loss):
        if 'loss' in summary and 'count' in summary and is_nothing(loss) is False:
            summary['loss'] += float(loss)
            summary['count'].setdefault('loss', 0)
            summary['count']['loss'] += 1
            return safe_divide(summary['loss'], summary['count']['loss'])
        else:
            return NOTHING

    @staticmethod
    def _compute_avg_metrics(summary: Dict, metrics: Dict):
        if 'metrics' in summary and 'count' in summary:
            temp = {}
            _metrics = summary['metrics']
            for key, value in metrics.items():
                _metrics.setdefault(key, 0)
                _metrics[key] += value
                summary['count'].setdefault(key, 0)
                summary['count'][key] += 1
            for key, value in _metrics.items():
                temp[key] = safe_divide(value, summary['count'].setdefault(key, 0))
            return temp
        else:
            return NOTHING


class DisplayHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DisplayHandler')
    def handle(self, ctx: Context):
        current = ctx.step.current
        total = ctx.step.total

        data = ' '.join(ctx.status.get_avg_loss_and_metrics(ctx))

        with Cursor.cursor_invisible():
            Cursor.refresh_print(
                str(ctx.status),
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


class DistributedDisplayHandler(DistributedHandlerWrapper):

    def __init__(self, exec_ranks: INT_SEQ_N = None):
        wrapped_handler = DisplayHandler()
        super().__init__(wrapped_handler, exec_ranks)


class DatasetHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DatasetHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check('status', silent=False)
        # get dataset through status
        ctx.status.get_dataset(ctx)


class StatusHandler(Handler):

    def __init__(self, status: str = 'train'):
        super().__init__()
        # get status supported
        from torchslime.core.status import proxy_status
        mode_supported = list(proxy_status.modules.keys())
        if status not in mode_supported:
            logger.warn('An unsupported status is set, this may cause some problems.')
        self.status = status
    
    @InvocationDebug('ModeHandler')
    def handle(self, ctx: Context):
        # context check
        ctx.ctx_check([
            'model'
        ], silent=False)
        # set status to the context
        from torchslime.core.status import proxy_status
        ctx.status = proxy_status.build(self.status)
        # change pytorch model mode
        ctx.status.set_model_mode(ctx)


class LRDecayHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('LRDecayHandler')
    def handle(self, ctx: Context):
        if ctx.ctx_check(['run.lr_decay']) is True:
            ctx.run.lr_decay.step()


class CallbackHandler(Handler):

    def __init__(self, hook: str):
        super().__init__()
        self._hook = hook

    @InvocationDebug('CallbackHandler')
    def handle(self, ctx: Context):
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks._exec_hook(self._hook, ctx)

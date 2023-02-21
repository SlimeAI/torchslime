from abc import abstractmethod
from typing import Dict, Sequence, Union, List, Callable, Any
from torchslime.util import BaseList, IterTool, NOTHING, is_nothing, safe_divide, type_cast, \
    InvocationDebug, SmartWraps, terminal as Cursor
from torchslime.util.formatter import progress_format, eta_format
from torchslime.core.context import BaseContext
from torchslime.core import DistributedContext
from torchslime.log import logger
from torchslime.util.type import INT_SEQ_N
from torch import set_grad_enabled


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @SmartWraps(func)
    def grad_switch(self, ctx: BaseContext):
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
    def handle(self, ctx: BaseContext):
        pass

    def __call__(self, ctx: BaseContext):
        self.handle(ctx)


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torchslime.core.handler.Handler): _description_
    """

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('EmptyHandler')
    def handle(self, _: BaseContext):
        """do nothing"""
        pass


# lambda or sequence of lambdas
L_SEQ = Union[Callable[[BaseContext], Any], Sequence[Callable[[BaseContext], Any]]]


class LambdaHandler(Handler, BaseList):
    
    def __init__(self, _lambda: L_SEQ):
        Handler.__init__(self)
        BaseList.__init__(self, _lambda)
    
    def handle(self, ctx: BaseContext):
        # execute lambda functions
        for _lambda in self:
            _lambda(ctx)


class DistributedHandler(Handler):

    def __init__(self, exec_ranks: INT_SEQ_N = None):
        super().__init__()
        self.exec_ranks = BaseList.create(exec_ranks)

    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        if self.exec_ranks is None or exec_ranks is None:
            # the exec_ranks are changeable
            self.exec_ranks = BaseList.create(exec_ranks)

    def __call__(self, ctx: DistributedContext):
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
    
    def handle(self, ctx: BaseContext):
        self._wrapped_handler(ctx)


class DistributedLambdaHandler(DistributedHandlerWrapper, BaseList):
    
    def __init__(self, _lambda: L_SEQ, exec_ranks: INT_SEQ_N = None):
        wrapped_handler = LambdaHandler(_lambda)
        DistributedHandlerWrapper.__init__(self, wrapped_handler, exec_ranks)
        BaseList.__init__(self, None)
        self.set_list(wrapped_handler.get_list())


# handler or sequence of handlers
C_SEQ = Union[Handler, Sequence[Handler]]


class HandlerContainer(Handler, BaseList):

    def __init__(self, handlers: C_SEQ = None):
        Handler.__init__(self)
        BaseList.__init__(self, handlers)
    
    def handle(self, ctx: BaseContext):
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

    def __call__(self, ctx: DistributedContext):
        rank = ctx.get_rank()
        if self.exec_ranks is not None and \
            (is_nothing(self.exec_ranks) or rank in self.exec_ranks):
            super().__call__(ctx)


class DistributedHandlerContainerWrapper(DistributedHandlerContainer):

    def __init__(self, wrapped_handler_container: HandlerContainer, default_exec_ranks: INT_SEQ_N = None):
        super().__init__(None, default_exec_ranks)
        self._wrapped_handler_container = wrapped_handler_container
        self.set_list(wrapped_handler_container.get_list())

    def handle(self, ctx: BaseContext):
        self._wrapped_handler_container(ctx)


DISTRIBUTED_CLASSES = (DistributedHandler, DistributedHandlerContainer)


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)

    @InvocationDebug('EpochIterationHandler')
    def handle(self, ctx: BaseContext):
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
    def handle(self, ctx: BaseContext):
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

    def __init__(self):
        super().__init__()

    @InvocationDebug('BackwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check(['step.loss']) is True:
            last = ctx.step.total % ctx.run.grad_acc
            grad_acc = ctx.run.grad_acc if (ctx.step.total - ctx.step.current - 1) >= last else last
            # backward
            (ctx.run.loss_reduction(ctx) / grad_acc).backward()


class OptimizerHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None):
        super().__init__(handlers)
    
    @InvocationDebug('OptimizerHandler')
    def handle(self, ctx: BaseContext):
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
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('step', silent=False)
        if ctx.ctx_check('run.metrics') is True:
            ctx.step.metrics = ctx.run.metrics(ctx)


class GatherAverageHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('GatherAverageHandler')
    def handle(self, ctx: BaseContext):
        from torchslime.core import DistributedContext
        from torchslime.metric import LossWrapper
        ctx: DistributedContext = ctx
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
    def handle(self, ctx: BaseContext):
        ctx.status.init_avg_inner_ctx(ctx, self.INNER_KEY)
        if self.type == 'avg':
            self.average(ctx)
        elif self.type == 'clear':
            self.clear(ctx)

    def average(self, ctx: BaseContext):
        from torchslime.metric import LossWrapper
        # get inner context variables
        summary = ctx.status.get_avg_inner_ctx(ctx, self.INNER_KEY)
        
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
        ctx.status.set_avg_loss_value_and_metrics(ctx, avg_loss.decode(), avg_metrics)

    def clear(self, ctx: BaseContext):
        # reset avg info
        ctx.status.clear_avg_info(ctx, self.INNER_KEY)

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

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('DisplayHandler')
    def handle(self, ctx: BaseContext):
        current = ctx.step.current
        total = ctx.step.total

        loss_value, metrics = ctx.status.get_avg_loss_value_and_metrics(ctx)
        data = {**loss_value, **metrics}
        data = ' - '.join(
            list(map(lambda item: '{0}: {1:.5f}'.format(*item), data.items()))
        )

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
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('status', silent=False)
        # get dataset through status
        ctx.status.get_dataset(ctx)


class StatusHandler(Handler):

    def __init__(self, status: str = 'train'):
        super().__init__()
        # get status supported
        from torchslime.core.status import context_status
        mode_supported = list(context_status.modules.keys())
        if status not in mode_supported:
            logger.warn('An unsupported status is set, this may cause some problems.')
        self.status = status
    
    @InvocationDebug('ModeHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model'
        ], silent=False)
        # set status to the context
        from torchslime.core.status import context_status
        ctx.status = context_status.build(self.status)
        # change pytorch model mode
        ctx.status.set_model_mode(ctx)


class LRDecayHandler(Handler):

    def __init__(self):
        super().__init__()
    
    @InvocationDebug('LRDecayHandler')
    def handle(self, ctx: BaseContext):
        if ctx.ctx_check(['run.lr_decay']) is True:
            ctx.run.lr_decay.step()


class CallbackHandler(Handler):

    def __init__(self, hook: str):
        super().__init__()
        self._hook = hook

    @InvocationDebug('CallbackHandler')
    def handle(self, ctx: BaseContext):
        ctx.ctx_check([
            'run.callbacks'
        ])
        ctx.run.callbacks._exec_hook(self._hook, ctx)

from abc import abstractmethod
from typing import Dict, Sequence, Union, List, Callable, Any, Iterable, Tuple
from torchslime.utils import BaseList, IterTool, NOTHING, is_nothing, safe_divide, type_cast, \
    InvocationDebug, SmartWraps, Count, is_none_or_nothing, Nothing, terminal as Cursor
from torchslime.utils.formatter import progress_format, eta_format
from torchslime.core.context import BaseContext
from torchslime.log import logger
from torchslime.utils.tstype import INT_SEQ_N
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


OPTIONAL_HANDLER = Union['Handler', None, Nothing]


class Handler:
    """Base class for all handlers.
    """
    
    _handler_id_gen = Count()
    id_attrs = ['name', 'phase']
    tab = ' ' * 4  # tab is equal to 4 spaces
    def __init__(
        self,
        *args,
        _id: Union[str, None, Nothing] = None,
        exec_ranks: INT_SEQ_N = ...
    ):
        super().__init__()
        # TODO: thread-safe and process-safe
        self.__id = _id if _id is not None else 'handler_{}'.format(self._handler_id_gen)
        self.__parent: Union[HandlerContainer, Nothing] = NOTHING
        self.set_exec_ranks(exec_ranks)

    @abstractmethod
    def handle(self, ctx: BaseContext):
        pass

    def __call__(self, ctx: BaseContext):
        ctx.hook.launch.handler_call(self, ctx)
    
    def replace_self(self, handler: 'Handler') -> bool:
        if self._verify_parent() is not True:
            return False
        parent = self.get_parent()
        index = parent.index(self)
        parent[index] = handler
        return True
    
    def insert_before_self(self, handler: 'Handler') -> bool:
        if self._verify_parent() is not True:
            return False
        parent = self.get_parent()
        index = parent.index(self)
        parent.insert(index, handler)
        return True
    
    def insert_after_self(self, handler: 'Handler') -> bool:
        if self._verify_parent() is not True:
            return False
        parent = self.get_parent()
        index = parent.index(self)
        parent.insert(index + 1, handler)
        return True
    
    def remove_self(self) -> bool:
        if self._verify_parent() is not True:
            return False
        parent = self.get_parent()
        parent.remove(self)
        return True
    
    def _verify_parent(self) -> bool:
        if is_nothing(self.get_parent()) or self not in self.get_parent():
            # root node, wild pointer or unmatched parent
            logger.warn('')
            self.del_parent()
            return False
        return True
    
    def get_by_id(self, _id: str, result: Union[list, None, Nothing] = NOTHING) -> 'Handler':
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        if self.__id == _id:
            self._append_search_result(self, result, allow_multiple=False)
        return NOTHING if len(result) < 1 else result[0]
    
    def get_by_class(self, __class: Union[type, Tuple[type]], result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        if isinstance(self, __class):
            self._append_search_result(self, result)
        return result
    
    def get_by_filter(self, __function: Callable, result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        if __function(self) is True:
            self._append_search_result(self, result)
        return result
    
    def _append_search_result(
        self,
        item,
        result: list,
        allow_duplicate: bool = False,
        allow_multiple: bool = True
    ):
        if item in result and allow_duplicate is False:
            # duplicate node
            logger.warn('')
            return
        # append matched item
        result.append(item)
        # ``len(result) == 2``: warn only once
        if allow_multiple is False and len(result) == 2:
            # multiple matched nodes
            logger.warn('')
    
    def get_id(self) -> Union[str, Nothing]:
        return self.__id

    def set_id(self, _id: Union[str, Nothing]):
        self.__id = _id
    
    def get_parent(self):
        return self.__parent

    def set_parent(self, _parent):
        if is_nothing(self.__parent) is False:
            # duplicate parent
            logger.warn('')
        self.__parent = _parent
    
    def del_parent(self):
        self.__parent = NOTHING
    
    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        self.__exec_ranks = BaseList.create(exec_ranks)

    def get_exec_ranks(self):
        return self.__exec_ranks

    def is_distributed(self) -> bool:
        return False
    
    def display(self):
        logger.info('Handler Structure:\n{content}'.format(
            content=self.get_display_str()
        ))
    
    def get_display_str(self) -> str:
        return '\n'.join(self._get_display_list(indent=0))

    def display_error(self, error_handler: OPTIONAL_HANDLER):
        logger.error('Handler Error Traceback:\n{content}'.format(
            content=self.get_display_error_str(error_handler=error_handler)
        ))
    
    def get_display_error_str(self, error_handler: OPTIONAL_HANDLER) -> str:
        return Cursor.single_color('w') + \
            '\n'.join(self._get_display_list(indent=0, error_handler=error_handler))

    def _get_display_list(self, indent=0, *args, error_handler: OPTIONAL_HANDLER = NOTHING) -> list:
        indent_str = indent * self.tab
        content = self.__str__()
        # error wrap
        if is_none_or_nothing(error_handler) is False and \
            self._is_error_handler(error_handler=error_handler):
            content = self._error_wrap(content)

        display_list = [
            '{indent_str}{content}'.format(
                indent_str=indent_str,
                content=content
            )
        ]
        return display_list
    
    def _error_wrap(self, item):
        return Cursor.single_color('r') + item + '  ' + self._error_indicator() + Cursor.single_color('w')
    
    def _error_indicator(self) -> str:
        _separator_len = 10
        # ×  <---------- ERROR Here ----------
        return chr(0x00D7) + '  ' + '<' + '-' * _separator_len + ' ERROR Here ' + '-' * _separator_len

    def _is_error_handler(self, error_handler: OPTIONAL_HANDLER = NOTHING):
        return self is error_handler

    def __str__(self) -> str:
        class_name = self._get_class_str()
        attrs = self._get_attr_str()
        return '{class_name}({attrs})'.format(class_name=class_name, attrs=attrs)
    
    def _get_attr_str(self) -> str:
        attr_dict = self._get_attr_dict()
        return ', '.join([
            '{key}={value}'.format(key=str(key), value=str(value)) \
            for key, value in attr_dict.items()
        ])
    
    def _get_class_str(self) -> str:
        return type(self).__name__
    
    def _get_display_attrs(self) -> dict:
        return {
            '_Handler__id': 'id',
            '_Handler__exec_ranks': 'exec_ranks'
        }

    def _get_attr_dict(self) -> dict:
        display_attrs = self._get_display_attrs()
        return {
            display_attrs[key]:value \
            for key, value in vars(self).items() \
            if key in display_attrs
        }


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torchslime.core.handler.Handler): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @InvocationDebug('EmptyHandler')
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


# handler or sequence of handlers
C_SEQ = Union[Handler, Sequence[Handler]]


class HandlerContainer(Handler, BaseList):

    def __init__(self, handlers: C_SEQ = None, *args, **kwargs):
        Handler.__init__(self, *args, **kwargs)
        # remove None and NOTHING
        BaseList.__init__(
            self,
            list(filter(lambda item: is_none_or_nothing(item) is not True, handlers if isinstance(handlers, Iterable) else []))
        )
        # set parent
        for handler in self:
            handler: Handler = handler
            handler.set_parent(self)
    
    def handle(self, ctx: BaseContext):
        for handler in self:
            handler(ctx)
    
    def get_by_id(self, _id: str, result: Union[list, None, Nothing] = NOTHING) -> 'Handler':
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_id(_id, result)
        for handler in self:
            handler: Handler = handler
            handler.get_by_id(_id, result)
        return NOTHING if len(result) < 1 else result[0]
    
    def get_by_class(self, __class: Union[type, Tuple[type]], result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_class(__class, result)
        for handler in self:
            handler: Handler = handler
            handler.get_by_class(__class, result)
        return result

    def get_by_filter(self, __function: Callable, result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_filter(__function, result)
        for handler in self:
            handler: Handler = handler
            handler.get_by_filter(__function, result)
        return result
    
    def append(self, handler: Handler):
        result = super().append(handler)
        handler.set_parent(self)
        return result
    
    def clear(self):
        for handler in self:
            handler: Handler = handler
            handler.del_parent()
        return super().clear()
    
    def extend(self, handlers: Iterable[Handler]):
        result = super().extend(handlers)
        for handler in handlers:
            handler.set_parent(self)
        return result
    
    def insert(self, __index, handler: Handler):
        result = super().insert(__index, handler)
        handler.set_parent(self)
        return result
    
    def pop(self, __index=...):
        item: Handler = super().pop(__index)
        item.del_parent()
        return item
    
    def remove(self, handler: Handler):
        result = super().remove(handler)
        handler.del_parent()
        return result
    
    def __setitem__(self, __i_s, handler: Union[Handler, Iterable[Handler]]):
        # TODO: del_parent to the replaced handlers
        result = super().__setitem__(__i_s, handler)
        if isinstance(__i_s, slice):
            for _handler in handler:
                _handler: Handler = _handler
                _handler.set_parent(self)
        else:
            handler.set_parent(self)
        return result
    
    def __delitem__(self, __i) -> None:
        handler: Union[Handler, Iterable[Handler]] = super().__getitem__(__i)
        if isinstance(__i, slice):
            for _handler in handler:
                _handler: Handler = _handler
                _handler.del_parent()
        else:
            handler.del_parent()
        return super().__delitem__(__i)
    
    def _get_display_list(self, indent=0, *args, error_handler: OPTIONAL_HANDLER = NOTHING) -> list:
        display_list = []
        indent_str = indent * self.tab
        prefix_content = self._get_class_str() + '(['
        # error wrap
        if is_none_or_nothing(error_handler) is False and \
            self._is_error_handler(error_handler=error_handler):
            prefix_content = self._error_wrap(prefix_content)
        # prefix
        display_list.append(indent_str + prefix_content)
        # handler
        for handler in self:
            display_list.extend(handler._get_display_list(indent + 1, error_handler=error_handler))
        # suffix
        display_list.append(indent_str + '], ' + self._get_attr_str() + ')')
        return display_list


class EpochIterationHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)

    @InvocationDebug('EpochIterationHandler')
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

    def __init__(self, handlers: C_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)

    @InvocationDebug('IterationHandler')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check('run.dataset') is True:
            for batch, progress, time, current, total in IterTool(ctx.run.dataset, True, True, True, True):
                ctx.step.from_dict({
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @InvocationDebug('BackwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check(['step.loss']) is True:
            last = ctx.step.total % ctx.run.grad_acc
            grad_acc = ctx.run.grad_acc if (ctx.step.total - ctx.step.current - 1) >= last else last
            # backward
            (ctx.run.loss_reduction(ctx) / grad_acc).backward()


class OptimizerHandler(HandlerContainer):

    def __init__(self, handlers: C_SEQ = None, *args, **kwargs):
        super().__init__(handlers, *args, **kwargs)
    
    @InvocationDebug('OptimizerHandler')
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
    
    @InvocationDebug('MetricsHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('step', silent=False)
        if ctx.ctx_check('run.metrics') is True:
            ctx.step.metrics = ctx.run.metrics(ctx)


class GatherAverageHandler(Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @InvocationDebug('GatherAverageHandler')
    def handle(self, ctx: BaseContext):
        from torchslime.core import Context
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
    
    @InvocationDebug('AverageInitHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook.state.init_avg_inner_ctx(ctx, self.INNER_KEY)
        # reset avg info
        ctx.hook.state.clear_avg_info(ctx, self.INNER_KEY)


class AverageHandler(Handler):

    # inner context key
    INNER_KEY = 'AVERAGE_INNER'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @InvocationDebug('AverageHandler')
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
    
    @InvocationDebug('DisplayHandler')
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
    
    @InvocationDebug('DatasetHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('status', silent=False)
        # get dataset through status
        ctx.hook.state.get_dataset(ctx)


class StateHandler(Handler):

    def __init__(self, state: str = 'train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get status supported
        from torchslime.core.hooks.state import context_status
        mode_supported = list(context_status.modules.keys())
        if state not in mode_supported:
            logger.warn('An unsupported status is set, this may cause some problems.')
        self.state = state
    
    @InvocationDebug('StatusHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model'
        ], silent=False)
        # set status to the context
        from torchslime.core.hooks.state import context_status
        ctx.hook.state = context_status.build(self.state)
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
    
    @InvocationDebug('LRDecayHandler')
    def handle(self, ctx: BaseContext):
        if ctx.ctx_check(['run.lr_decay']) is True:
            ctx.run.lr_decay.step()

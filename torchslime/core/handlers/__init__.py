from typing import Sequence, Union, List, Callable, Iterable, Tuple
from torchslime.core.handlers.common import H_SEQ, BaseContext
from torchslime.utils import Count, terminal as Cursor
from torchslime.core.context.base import BaseContext
from torchslime.log import logger
from torchslime.utils.bases import NOTHING, BaseList, Nothing, is_none_or_nothing, is_nothing
from torchslime.utils.tstype import INT_SEQ_N
from torchslime.components.registry import Registry
from torchslime.components.exception import HandlerException, HandlerTerminate


OPTIONAL_HANDLER = Union['Handler', Sequence['Handler'], None, Nothing]


class Handler:
    """Base class for all handlers.
    """
    
    _handler_id_gen = Count()
    id_attrs = ['name', 'phase']
    tab = ' ' * 4  # tab is equal to 4 spaces
    def __init__(
        self,
        *,
        _id: Union[str, None, Nothing] = None,
        exec_ranks: INT_SEQ_N = ...
    ):
        super().__init__()
        # TODO: thread-safe and process-safe
        self.__id = _id if _id is not None else 'handler_{}'.format(self._handler_id_gen)
        self.__parent: Union[HandlerContainer, Nothing] = NOTHING
        self.set_exec_ranks(exec_ranks)

    def handle(self, ctx: BaseContext): pass

    def __call__(self, ctx: BaseContext):
        try:
            ctx.hook_ctx.launch.handler_call(self, ctx)
        except HandlerTerminate as ht:
            # set ``raise_handler`` to the nearest handler
            if is_none_or_nothing(ht.raise_handler) is True:
                ht.raise_handler = self
            raise ht
        except HandlerException as he:
            raise he
        except Exception as e:
            raise HandlerException(exception_handler=self, exception=e)
    
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

    def display_traceback(self, target_handlers: OPTIONAL_HANDLER, wrap_func: Union[str, Callable] = 'exception', level: str = 'error'):
        wrap_func = wrap_func if callable(wrap_func) is True else display_wrap_func.get(wrap_func, lambda x: x)

        getattr(logger, level, logger.error)('Handler Traceback:\n{content}'.format(
            content=self.get_display_traceback_str(target_handlers=target_handlers, wrap_func=wrap_func)
        ))
    
    def get_display_traceback_str(self, target_handlers: OPTIONAL_HANDLER, wrap_func: Callable) -> str:
        return Cursor.single_color('w') + \
            '\n'.join(self._get_display_list(indent=0, target_handlers=target_handlers, wrap_func=wrap_func))

    def _get_display_list(self, indent=0, *, target_handlers: OPTIONAL_HANDLER = NOTHING, wrap_func: Callable = NOTHING) -> list:
        indent_str = indent * self.tab
        content = self.__str__()
        # error wrap
        if is_none_or_nothing(target_handlers) is False and \
            self._is_target_handler(target_handlers=target_handlers):
            content = wrap_func(content)

        display_list = [
            '{indent_str}{content}'.format(
                indent_str=indent_str,
                content=content
            )
        ]
        return display_list

    def _is_target_handler(self, target_handlers: OPTIONAL_HANDLER = NOTHING):
        return self in BaseList.create(
            target_handlers,
            return_none=False,
            return_nothing=False,
            return_ellipsis=False
        )

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


display_wrap_func = Registry('display_wrap_func')


@display_wrap_func.register('exception')
def _exception_wrap(item) -> str:
    _separator_len = 10
    # ×  <---------- EXCEPTION Here ----------
    _exception_indicator = chr(0x00D7) + '  ' + '<' + '-' * _separator_len + ' EXCEPTION Here ' + '-' * _separator_len
    return Cursor.single_color('r') + item + '  ' + _exception_indicator + Cursor.single_color('w')


@display_wrap_func.register('terminate')
def _terminate_wrap(item) -> str:
    _separator_len = 10
    # ||---------- Handler TERMINATE ----------||
    _terminate_indicator = '||' + '-' * _separator_len + ' Handler TERMINATE ' + '-' * _separator_len + '||'
    return Cursor.single_color('g') + item + '  ' + _terminate_indicator + Cursor.single_color('w')


# handler or sequence of handlers
H_SEQ = Union[Handler, Sequence[Handler]]


class HandlerContainer(Handler, BaseList):

    def __init__(self, handlers: H_SEQ = None, *args, **kwargs):
        Handler.__init__(self, *args, **kwargs)
        # remove None and NOTHING
        BaseList.__init__(
            self,
            list(filter(lambda item: is_none_or_nothing(item) is not True, handlers if isinstance(handlers, Iterable) else []))
        )
        # set parent
        for handler in self:
            handler: Handler
            handler.set_parent(self)
    
    def handle(self, ctx: BaseContext):
        for handler in self.get_list__():
            handler(ctx)
    
    def get_by_id(self, _id: str, result: Union[list, None, Nothing] = NOTHING) -> 'Handler':
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_id(_id, result)
        for handler in self:
            handler: Handler
            handler.get_by_id(_id, result)
        return NOTHING if len(result) < 1 else result[0]
    
    def get_by_class(self, __class: Union[type, Tuple[type]], result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_class(__class, result)
        for handler in self:
            handler: Handler
            handler.get_by_class(__class, result)
        return result

    def get_by_filter(self, __function: Callable, result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_filter(__function, result)
        for handler in self:
            handler: Handler
            handler.get_by_filter(__function, result)
        return result
    
    def append(self, handler: Handler):
        result = super().append(handler)
        handler.set_parent(self)
        return result
    
    def clear(self):
        for handler in self:
            handler: Handler
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
                _handler: Handler
                _handler.set_parent(self)
        else:
            handler.set_parent(self)
        return result
    
    def __delitem__(self, __i) -> None:
        handler: Union[Handler, Iterable[Handler]] = super().__getitem__(__i)
        if isinstance(__i, slice):
            for _handler in handler:
                _handler: Handler
                _handler.del_parent()
        else:
            handler.del_parent()
        return super().__delitem__(__i)
    
    def _get_display_list(self, indent=0, *, target_handlers: OPTIONAL_HANDLER = NOTHING, wrap_func: Callable = NOTHING) -> list:
        display_list = []
        indent_str = indent * self.tab
        prefix_content = self._get_class_str() + '(['
        # error wrap
        if is_none_or_nothing(target_handlers) is False and \
            self._is_target_handler(target_handlers=target_handlers):
            prefix_content = wrap_func(prefix_content)
        # prefix
        display_list.append(indent_str + prefix_content)
        # handler
        for handler in self.get_list__():
            display_list.extend(handler._get_display_list(indent + 1, target_handlers=target_handlers, wrap_func=wrap_func))
        # suffix
        display_list.append(indent_str + '], ' + self._get_attr_str() + ')')
        return display_list


from .common import *
from .wrappers import *

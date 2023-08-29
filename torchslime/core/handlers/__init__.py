from torchslime.utils.typing import (
    Sequence,
    Union,
    List,
    Callable,
    Iterable,
    Tuple,
    SupportsIndex
)
from torchslime.utils import Count, cli as Cursor
from torchslime.core.context.base import BaseContext
from torchslime.log import logger
from torchslime.utils.bases import (
    NOTHING,
    PASS,
    BaseList,
    Nothing,
    is_none_or_nothing,
    Pass
)
from torchslime.utils.meta import Meta, Metadata
from torchslime.utils.typing import INT_SEQ_N
from torchslime.components.registry import Registry
from torchslime.components.exception import (
    HandlerException,
    HandlerTerminate,
    HandlerBreak,
    HandlerContinue,
    HandlerWrapperException
)
from torchslime.utils.formatter import dict_to_key_value_str_list, concat_format


@Meta
class HandlerMetaclass:
    """
    Metadata initialization and operations
    """
    
    metadata__: Metadata
    default_metadata__: 'HandlerMetadata'
    # for generating unique id
    _handler_id_gen = Count()
    
    def __init__(self) -> None:
        super().__init__()
        # set default metadata and apply default value
        self.default_metadata__ = HandlerMetadata()
        self.metadata__ = self.default_metadata__ | self.metadata__
        # set default id if id is not specified
        # TODO: thread-safe and process-safe
        if is_none_or_nothing(self.get_id()):
            self.set_id(f'handler_{self._handler_id_gen}')
        # bind self to wrappers after initialization
        wrappers = self.get_wrappers()
        if not is_none_or_nothing(wrappers):
            wrappers.bind(self)
    
    def get_id(self) -> Union[str, Nothing]:
        return self.metadata__.get('id', NOTHING)

    def set_id(self, _id: str) -> None:
        self.metadata__ |= ID(_id)
    
    def get_exec_ranks(self) -> Union[Iterable[int], None, Nothing, Pass]:
        return self.metadata__.get('exec_ranks', NOTHING)
    
    def set_exec_ranks(self, exec_ranks: Union[Iterable[int], None, Nothing, Pass]) -> None:
        self.metadata__ |= ExecRanks(exec_ranks)

    def reset_exec_ranks(self) -> None:
        self.metadata__['exec_ranks'] = self.default_metadata__['exec_ranks']

    def get_wrappers(self) -> Union['HandlerWrapperContainer', Nothing]:
        return self.metadata__.get('wrappers', NOTHING)
    
    def set_wrappers(self, *wrappers) -> None:
        # set metadata
        self.metadata__ |= Wrappers(*wrappers)
        # bind self to wrappers
        self.get_wrappers().bind(self)
    
    def reset_wrappers(self) -> None:
        self.metadata__['wrappers'] = self.default_metadata__['wrappers']
    
    def get_lifecycle(self):
        pass
    
    def set_lifecycle(self):
        pass


class Handler(HandlerMetaclass):
    """Base class for all handlers.
    """
    
    def __init__(self):
        super().__init__()
        # parent initialized to NOTHING
        self.__parent: Union[HandlerContainer, Nothing] = NOTHING

    def handle(self, ctx: BaseContext): pass

    def __call__(self, ctx: BaseContext):
        try:
            wrappers = self.get_wrappers()
            # call wrapper if wrapper is not empty
            handler = self if is_none_or_nothing(wrappers) else wrappers
            ctx.hook_ctx.launch.handler_handle(handler, ctx)
        #
        # Handler Interrupt
        #
        except HandlerTerminate as ht:
            # set ``raise_handler`` to the nearest handler
            if is_none_or_nothing(ht.raise_handler):
                ht.raise_handler = self
            raise ht
        except (HandlerBreak, HandlerContinue) as hi:
            raise hi
        #
        # Handler Wrapper Exception (should be in front of ``HandlerException``)
        #
        except HandlerWrapperException as hwe:
            # output the original exception handler, and raise it as a normal handler exception
            logger.error(str(hwe))
            raise HandlerException(exception_handler=self, exception=hwe.exception)
        #
        # Handler Exception
        #
        except HandlerException as he:
            raise he
        #
        # other Exception(s)
        #
        except Exception as e:
            raise HandlerException(exception_handler=self, exception=e)
    
    def replace_self(self, handler: 'Handler') -> bool:
        if not self._verify_parent():
            return False
        parent = self.get_parent()
        index = parent.index(self)
        parent[index] = handler
        return True
    
    def insert_before_self(self, handler: 'Handler') -> bool:
        if not self._verify_parent():
            return False
        parent = self.get_parent()
        index = parent.index(self)
        parent.insert(index, handler)
        return True
    
    def insert_after_self(self, handler: 'Handler') -> bool:
        if not self._verify_parent():
            return False
        parent = self.get_parent()
        index = parent.index(self)
        parent.insert(index + 1, handler)
        return True
    
    def remove_self(self) -> bool:
        if not self._verify_parent():
            return False
        parent = self.get_parent()
        parent.remove(self)
        return True
    
    def _verify_parent(self) -> bool:
        if self.get_parent() is NOTHING or self not in self.get_parent():
            # root node, wild pointer or unmatched parent
            logger.warn('')
            self.del_parent()
            return False
        return True
    
    def get_by_id(self, _id: str, result: Union[list, None, Nothing] = NOTHING) -> 'Handler':
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        if self.get_id() == _id:
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
        
        if __function(self):
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
    
    def get_parent(self):
        return self.__parent

    def set_parent(self, _parent):
        if not is_none_or_nothing(self.__parent):
            # duplicate parent
            logger.warn('')
        self.__parent = _parent
    
    def del_parent(self):
        self.__parent = NOTHING
    
    def display(self):
        logger.info(f'Handler Structure:\n{str(self)}')

    def display_traceback(
        self,
        target_handlers: Union[List['Handler'], None, Nothing],
        wrap_func: Union[str, Callable] = 'exception',
        level: str = 'error'
    ):
        wrap_func = wrap_func if callable(wrap_func) is True else display_wrap_func.get(wrap_func)

        content = self.get_display_traceback_str(target_handlers=target_handlers, wrap_func=wrap_func)
        getattr(logger, level, logger.error)(f'Handler Traceback:\n{content}')
    
    def get_display_traceback_str(
        self,
        target_handlers: Union[List['Handler'], None, Nothing],
        wrap_func: Callable
    ) -> str:
        display_list = str(self).split('\n')
        if self._is_target_handler(target_handlers):
            display_list[0] = wrap_func(display_list[0])
            print(display_list[0], '114514')
        
        return Cursor.single_color('w') + \
            '\n'.join(
                display_list
            )

    def _is_target_handler(
        self,
        target_handlers: Union[List['Handler'], None, Nothing] = NOTHING
    ):
        return self in BaseList.create__(
            target_handlers,
            return_none=False,
            return_nothing=False,
            return_pass=False
        )

    def __str__(self) -> str:
        class_name = self._get_class_name()
        
        metadata_display_list = dict_to_key_value_str_list(self._get_metadata_dict())
        metadata = concat_format('[', metadata_display_list, ']', break_line=False, item_sep=', ')
        
        attr_display_list = dict_to_key_value_str_list(self._get_attr_dict())
        attr = concat_format('(', attr_display_list, ')', break_line=False, item_sep=', ')
        
        return f'{class_name}{metadata}{attr}'
    
    def _get_class_name(self) -> str:
        return type(self).__name__

    def _get_attr_dict(self) -> dict:
        return {}
    
    def _get_metadata_dict(self) -> dict:
        return {
            'id': self.get_id(),
            'exec_ranks': self.get_exec_ranks(),
            'wrappers': self.get_wrappers(),
            'lifecycle': self.get_lifecycle()
        }


display_wrap_func = Registry('display_wrap_func')


@display_wrap_func(name='exception')
def _exception_wrap(item) -> str:
    _separator_len = 10
    # ×  <---------- EXCEPTION Here ----------
    _exception_indicator = chr(0x00D7) + '  ' + '<' + '-' * _separator_len + ' EXCEPTION Here ' + '-' * _separator_len
    return Cursor.single_color('r') + item + '  ' + _exception_indicator + Cursor.single_color('w')


@display_wrap_func(name='terminate')
def _terminate_wrap(item) -> str:
    _separator_len = 10
    # ||---------- Handler TERMINATE ----------||
    _terminate_indicator = '||' + '-' * _separator_len + ' Handler TERMINATE ' + '-' * _separator_len + '||'
    return Cursor.single_color('g') + item + '  ' + _terminate_indicator + Cursor.single_color('w')


class HandlerContainer(Handler, BaseList[Handler]):

    def __init__(
        self,
        handlers: Union[Iterable[Handler], None, Nothing] = None,
    ):
        Handler.__init__(self)
        # remove ``None`` and ``NOTHING`` in ``handlers``
        handlers: List[Handler] = list(filter(
            lambda item: not is_none_or_nothing(item),
            handlers if isinstance(handlers, Iterable) and not is_none_or_nothing(handlers) else []
        ))
        BaseList.__init__(
            self,
            handlers
        )
    
    def handle(self, ctx: BaseContext):
        try:
            for handler in self:
                handler(ctx)
        except HandlerContinue:
            # continue in the container
            pass
    
    def __call__(self, ctx: BaseContext):
        try:
            super().__call__(ctx)
        except HandlerBreak:
            # break out of the container
            pass
    
    def get_by_id(self, _id: str, result: Union[list, None, Nothing] = NOTHING) -> 'Handler':
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_id(_id, result)
        for handler in self:
            handler.get_by_id(_id, result)
        return NOTHING if len(result) < 1 else result[0]
    
    def get_by_class(self, __class: Union[type, Tuple[type]], result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_class(__class, result)
        for handler in self:
            handler.get_by_class(__class, result)
        return result

    def get_by_filter(self, __function: Callable, result: Union[list, None, Nothing] = NOTHING) -> List['Handler']:
        # initialize
        result = [] if is_none_or_nothing(result) else result
        
        super().get_by_filter(__function, result)
        for handler in self:
            handler.get_by_filter(__function, result)
        return result
    
    def __setitem__(
        self,
        __key: Union[SupportsIndex, slice],
        __value: Union[Handler, Iterable[Handler]]
    ) -> None:
        replaced = self[__key]
        result = super().__setitem__(__key, __value)
        # delete parents of the replaced handlers and set parents to the replacing handlers
        if isinstance(__key, slice):
            for _replaced in replaced:
                _replaced.del_parent()
            
            for _handler in __value:
                _handler.set_parent(self)
        else:
            replaced.del_parent()
            __value.set_parent(self)
        return result
    
    def __delitem__(
        self,
        __key: Union[SupportsIndex, slice]
    ) -> None:
        __value = self[__key]
        if isinstance(__key, slice):
            for _handler in __value:
                _handler.del_parent()
        else:
            __value.del_parent()
        return super().__delitem__(__key)
    
    def insert(self, __index: SupportsIndex, __handler: Handler) -> None:
        __handler.set_parent(self)
        return super().insert(__index, __handler)
    
    def __str__(self) -> str:
        class_name = self._get_class_name()
        
        metadata_display_list = dict_to_key_value_str_list(self._get_metadata_dict())
        metadata = concat_format('[', metadata_display_list, ']', break_line=False, item_sep=', ')
        
        handlers = concat_format('([', [str(handler) for handler in self], '])')
        attr_display_list = dict_to_key_value_str_list(self._get_attr_dict())
        attr = concat_format('(', attr_display_list, ')', break_line=False, item_sep=', ')
        
        return f'{class_name}{metadata}{handlers}{attr}'


from .common import *
from .wrappers import *

#
# Metadata
#

class HandlerMetadata(Metadata):
    
    def __init__(self):
        super().__init__()
        self.update(
            id=NOTHING,
            exec_ranks=PASS,
            wrappers=NOTHING,
            lifecycle=NOTHING
        )


class ID(Metadata):
    
    def __init__(self, _id: str):
        super().__init__('id', _id)


class ExecRanks(Metadata):
    
    def __init__(
        self,
        exec_ranks: Union[Iterable[int], None, Nothing, Pass] = PASS
    ):
        super().__init__('exec_ranks', BaseList.create__(exec_ranks))


class Wrappers(Metadata):
    
    def __init__(self, *wrappers):
        super().__init__('wrappers', HandlerWrapperContainer(list(wrappers)))


class Lifecycle(Metadata):
    
    pass

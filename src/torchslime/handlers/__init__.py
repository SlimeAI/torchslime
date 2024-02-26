from torchslime.handlers.riching import HandlerTreeProfiler
from torchslime.utils.common import (
    Count,
    dict_to_key_value_str_list,
    concat_format
)
from torchslime.utils.typing import (
    NOTHING,
    Nothing,
    NoneOrNothing,
    Union,
    List,
    Callable,
    Iterable,
    Tuple,
    is_none_or_nothing,
    TypeVar,
    Pass,
    PASS,
    TYPE_CHECKING
)
from torchslime.logging.logger import logger
from torchslime.logging.rich import (
    yield_console
)
from torchslime.utils.bases import (
    CompositeStructure,
    CompositeDFS,
    MutableBiListItem,
    BiList,
    BaseList
)
from torchslime.utils.exception import (
    HandlerException,
    HandlerTerminate,
    HandlerBreak,
    HandlerContinue,
    HandlerWrapperException
)
from functools import partial

if TYPE_CHECKING:
    from torchslime.logging.rich import Group
    from torchslime.context import Context


class Handler(CompositeStructure, MutableBiListItem):
    """
    Base class for all handlers.
    """
    # for generating unique id
    _handler_id_gen = Count()
    
    def __init__(
        self,
        *,
        id: Union[str, NoneOrNothing] = NOTHING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass] = PASS,
        wrappers: Union[Iterable['HandlerWrapper'], NoneOrNothing] = NOTHING,
        lifecycle=NOTHING
    ):
        CompositeStructure.__init__(self)
        MutableBiListItem.__init__(self)
        
        self.set_id(id)
        self.set_exec_ranks(exec_ranks)
        self.set_wrappers(wrappers)
        # TODO: lifecycle
        # self.set_lifecycle()

    def handle(self, ctx: "Context") -> None: pass

    def __call__(self, ctx: "Context") -> None:
        try:
            wrappers = self.get_wrappers()
            exec_ranks = self.get_exec_ranks()
            
            if is_none_or_nothing(wrappers):
                ctx.hook_ctx.launch.call(partial(self.handle, ctx), exec_ranks=exec_ranks)
            else:
                wrappers: HandlerWrapperContainer
                ctx.hook_ctx.launch.call(partial(wrappers.handle, ctx, self), exec_ranks=exec_ranks)
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
    
    #
    # Handler Search Operations
    #
    
    def composite_iterable__(self) -> Nothing: return NOTHING
    
    def get_by_id(self, __id: str) -> 'Handler':
        results = CompositeDFS(self, lambda handler: handler.get_id() == __id)
        if len(results) > 1:
            logger.warning(f'Duplicate id found in the Handler: {str(self)}.')
        return NOTHING if len(results) < 1 else results[0]
    
    def get_by_class(self, __class: Union[type, Tuple[type]]) -> List['Handler']:
        return CompositeDFS(self, lambda handler: isinstance(handler, __class))
    
    def get_by_filter(self, __func: Callable[["Handler"], bool]) -> List['Handler']:
        return CompositeDFS(self, __func)
    
    def display(
        self,
        display_attr: bool = True,
        target_handlers: Union[Iterable["Handler"], NoneOrNothing] = NOTHING,
        wrap_func: Union[str, Callable[["Group", "Handler"], "Group"], NoneOrNothing] = NOTHING,
        enable_console: bool = True,
        enable_alt_console: bool = False,
        handler_tree_profiler: Union[HandlerTreeProfiler, NoneOrNothing] = NOTHING
    ):
        if is_none_or_nothing(handler_tree_profiler):
            handler_tree_profiler = HandlerTreeProfiler()
        root = handler_tree_profiler.profile(
            self,
            display_attr=display_attr,
            target_handlers=target_handlers,
            wrap_func=wrap_func
        )
        for console in yield_console(
            enable_console=enable_console,
            enable_alt_console=enable_alt_console
        ):
            console.print('[bold]Handler Tree Profile: [/bold]')
            console.print(root)
    
    def __str__(self) -> str:
        class_name = self.get_class_name()
        
        display_attr_list = dict_to_key_value_str_list(self.get_display_attr_dict())
        attr = concat_format('(', display_attr_list, ')', break_line=False, item_sep=', ')
        
        return f'{class_name}{attr}'
    
    def get_class_name(self) -> str:
        return type(self).__name__

    def get_id(self) -> Union[str, Nothing]:
        return self.__id

    def set_id(self, __id: Union[str, NoneOrNothing]) -> None:
        if is_none_or_nothing(__id):
            self.__id = f'handler_{self._handler_id_gen}'
        else:
            self.__id = __id
    
    def get_exec_ranks(self) -> Union[Iterable[int], NoneOrNothing, Pass]:
        return self.__exec_ranks
    
    def set_exec_ranks(self, exec_ranks: Union[Iterable[int], NoneOrNothing, Pass]) -> None:
        self.__exec_ranks = BaseList.create__(exec_ranks)

    def get_wrappers(self) -> Union['HandlerWrapperContainer', NoneOrNothing]:
        return self.__wrappers
    
    def set_wrappers(self, wrappers: Union[Iterable['HandlerWrapper'], NoneOrNothing]) -> None:
        if is_none_or_nothing(wrappers):
            self.__wrappers = NOTHING
        else:
            self.__wrappers = HandlerWrapperContainer(wrappers)
    
    def get_lifecycle(self):
        pass
    
    def set_lifecycle(self):
        pass
    
    def get_display_attr_dict(self) -> dict:
        return {
            'id': self.get_id(),
            'exec_ranks': self.get_exec_ranks(),
            'wrappers': self.get_wrappers(),
            'lifecycle': self.get_lifecycle()
        }


_T_Handler = TypeVar('_T_Handler', bound=Handler)

class HandlerContainer(Handler, BiList[_T_Handler]):

    def __init__(
        self,
        handlers: Union[Iterable[_T_Handler], NoneOrNothing] = NOTHING,
        *,
        id: Union[str, NoneOrNothing] = NOTHING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass] = PASS,
        wrappers: Union[Iterable['HandlerWrapper'], NoneOrNothing] = NOTHING,
        lifecycle=NOTHING
    ):
        Handler.__init__(
            self,
            id=id,
            exec_ranks=exec_ranks,
            wrappers=wrappers,
            lifecycle=lifecycle
        )
        # remove ``None`` and ``NOTHING`` in ``handlers``
        handlers = filter(
            lambda item: not is_none_or_nothing(item),
            BaseList(handlers)
        )
        BiList.__init__(
            self,
            handlers
        )
    
    def handle(self, ctx: "Context") -> None:
        try:
            for handler in self:
                handler(ctx)
        except HandlerContinue:
            # continue in the container
            pass
    
    def __call__(self, ctx: "Context") -> None:
        try:
            super().__call__(ctx)
        except HandlerBreak:
            # break out of the container
            pass
    
    def composite_iterable__(self) -> Iterable[_T_Handler]: return self


from .common import *
from .wrappers import *

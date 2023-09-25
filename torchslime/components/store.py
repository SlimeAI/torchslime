from torchslime.utils.typing import (
    NOTHING,
    Any,
    TypeVar,
    is_none_or_nothing,
    overload,
    is_slime_naming,
    Dict,
    List,
    Callable,
    TextIO,
    Union,
    NoneOrNothing,
    Set
)
from torchslime.utils.bases import Base
from torchslime.utils.decorators import Singleton, ItemAttrBinding, ContextDecoratorBinding, RemoveOverload, DecoratorCall
import threading
import os
import sys
import re

_T = TypeVar('_T')

LISTEN_FUNC_SUFFIX = '_listen__'
LISTEN_FUNC_SUFFIX_PATTERN = re.compile(f'{LISTEN_FUNC_SUFFIX}$')
LISTEN_FLAG = 'store_listen__'

class StoreListener:
    
    def listen_inspect__(self) -> Set[str]:
        return set(
            map(
                # get the real listened attribute name
                lambda name: LISTEN_FUNC_SUFFIX_PATTERN.sub('', name),
                filter(
                    # filter out store listen function and check if it's callable
                    lambda name: LISTEN_FUNC_SUFFIX_PATTERN.search(name) is not None and getattr(getattr(self, name), LISTEN_FLAG, NOTHING),
                    dir(self)
                )
            )
        )


@overload
def StoreListen(_func: NoneOrNothing = NOTHING, *, flag: bool = True) -> Callable[[_T], _T]: pass
@overload
def StoreListen(_func: _T, *, flag: bool = True) -> _T: pass

@DecoratorCall(index=0, keyword='_func')
def StoreListen(_func=NOTHING, *, flag: bool = True):
    def decorator(func: _T) -> _T:
        try:
            setattr(func, LISTEN_FLAG, flag)
        except Exception:
            from torchslime.logging.logger import logger
            logger.warning(f'Set ``{LISTEN_FLAG}`` attribute failed. Callable: {str(func)}. Please make sure it supports attribute set.')
        return func
    return decorator


class ScopedStore(Base):
    
    def __init__(self) -> None:
        super().__init__()
        # attr name to listeners
        self.__listen: Dict[str, List[StoreListener]] = {}
        # listener id to attr names
        self.__listen_names: Dict[str, Set[str]] = {}
    
    def add_listener__(self, __listener: StoreListener, *, init: bool = True) -> None:
        listener_id = self.get_listener_id__(__listener)
        names = __listener.listen_inspect__()
        
        if listener_id in self.__listen_names:
            # inspect new listen names
            # use a copy of ``listen_names`` to avoid value change during iteration
            names = names - set(self.__listen_names[listener_id])
        
        for name in names:
            self.add_listen_name__(__listener, name, init=init)
    
    def add_listen_name__(self, __listener: StoreListener, __name: str, *, init: bool = True):
        listener_id = self.get_listener_id__(__listener)
        if listener_id not in self.__listen_names:
            self.__listen_names[listener_id] = set()
        
        if __name not in self.__listen:
            self.__listen[__name] = []
        
        self.__listen_names[listener_id].add(__name)
        self.__listen[__name].append(__listener)
        
        if init:
            value = getattr(self, __name, NOTHING)
            self.notify__(__listener, __name, value, NOTHING)
    
    def notify__(self, __listener: StoreListener, __name: str, __new_value: Any, __old_value: Any) -> None:
        func: Callable[[Any, Any], None] = getattr(__listener, f'{__name}{LISTEN_FUNC_SUFFIX}')
        return func(__new_value, __old_value)
    
    def remove_listener__(self, __listener: StoreListener) -> None:
        listener_id = self.get_listener_id__(__listener)
        if listener_id not in self.__listen_names:
            return
        
        # use a copy of ``listen_names`` to avoid value change during iteration
        for name in list(self.__listen_names[listener_id]):
            self.remove_listen_name__(__listener, name)
    
    def remove_listen_name__(self, __listener: StoreListener, __name: str) -> None:
        listener_id = self.get_listener_id__(__listener)
        if listener_id in self.__listen_names:
            names = self.__listen_names[listener_id]
            if __name in names:
                names.remove(__name)
            if len(names) < 1:
                del self.__listen_names[listener_id]

        if __name in self.__listen:
            listeners = self.__listen[__name]
            if __listener in listeners:
                listeners.remove(__listener)
            if len(listeners) < 1:
                del self.__listen[__name]
    
    @staticmethod
    def get_listener_id__(__listener: StoreListener) -> str:
        # this behavior may change through different torchslime versions
        return str(id(__listener))
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in self.__listen:
            return super().__setattr__(__name, __value)
        else:
            old_value = getattr(self, __name, NOTHING)
            super().__setattr__(__name, __value)
            # listener is called only when the new value is different from the old value
            if __value is not old_value:
                for listener in self.__listen[__name]:
                    self.notify__(listener, __name, __value, old_value)


@Singleton
class BuiltinScopedStore(ScopedStore):
    
    def __init__(self) -> None:
        """
        set ``builtin__`` store config
        """
        super().__init__()
        # whether to save log metadata (e.g., exec_name, lineno, etc.) to cache.
        self.use_log_cache = True
        self.log_cache = Base()
        # flag to log only once. For example, some warnings may appear only once.
        self.log_once = Base()
        # whether to use call debug
        self.use_call_debug = False
        self.call_debug_cache = Base()
        # indent str for CLI display
        self.indent_str = ' ' * 4  # default is 4 spaces
        # cli flag
        self.prev_refresh = False
        self.refresh_state = False
        # std out / err
        self.stdout: Union[TextIO, NoneOrNothing] = sys.stdout
        self.stderr: Union[TextIO, NoneOrNothing] = sys.stderr
        # log template
        self.log_template: Union[str, NoneOrNothing] = NOTHING
        self.log_rich_template: Union[str, NoneOrNothing] = NOTHING
        self.log_dateformat: Union[str, NoneOrNothing] = NOTHING
        # launch
        from torchslime.utils.launch import LaunchUtil
        self.launch: Union[str, LaunchUtil] = 'vanilla'

_builtin_scoped_store = BuiltinScopedStore()

_scoped_store_dict = {}

@ItemAttrBinding
@Singleton
@RemoveOverload(checklist=[
    'add_listener__',
    'add_listen_name__',
    'remove_listener__',
    'remove_listen_name__'
])
class Store:
    
    def scope__(self, __key) -> Union[ScopedStore, BuiltinScopedStore]:
        if __key == 'builtin__':
            return _builtin_scoped_store
        elif __key in _scoped_store_dict:
            return _scoped_store_dict[__key]
        else:
            return _scoped_store_dict.setdefault(__key, ScopedStore())

    def current__(self) -> ScopedStore:
        return self.scope__(self.get_current_key__())

    def builtin__(self) -> BuiltinScopedStore:
        return self.scope__('builtin__')

    def destroy__(self, __key=NOTHING):
        if is_none_or_nothing(__key):
            __key = self.get_current_key__()
        
        if __key in _scoped_store_dict:
            del _scoped_store_dict[__key]

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.current__(), __name)

    def __getattribute__(self, __name: str) -> Any:
        # slime naming
        if is_slime_naming(__name) is True:
            return super().__getattribute__(__name)
        # else get from ScopedStore object
        return getattr(self.current__(), __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        setattr(self.current__(), __name, __value)
    
    def __delattr__(self, __name: str) -> None:
        delattr(self.current__(), __name)
    
    @staticmethod
    def get_current_key__() -> str:
        pid = os.getpid(),
        tid = threading.get_ident()
        return f'p{pid}-t{tid}'
    
    @overload
    def add_listener__(self, __listener: StoreListener, *, init: bool = True) -> None: pass
    @overload
    def add_listen_name__(self, __listener: StoreListener, __name: str, *, init: bool = True): pass
    @overload
    def remove_listener__(self, __listener: StoreListener) -> None: pass
    @overload
    def remove_listen_name__(self, __listener: StoreListener, __name: str) -> None: pass


@ContextDecoratorBinding
@RemoveOverload(checklist=['__call__'])
class StoreSet:

    def __init__(self, __name: str, __value: Any, *, restore: bool = True, key=NOTHING) -> None:
        self.name = __name
        self.value = __value
        self.restore = restore
        self.key = store.get_current_key__() if is_none_or_nothing(key) is True else key
        self._store = store.scope__(self.key)
    
    # just for type hint
    @overload
    def __call__(self, func: _T) -> _T: pass

    def __enter__(self) -> 'StoreSet':
        self._set_value()
        return self

    def __exit__(self, *args, **kwargs):
        self._restore_value()
    
    def _set_value(self):
        # cache the store value before ``StoreSet``
        self.prev_value = self._store[self.name]
        # set value
        self._store[self.name] = self.value

    def _restore_value(self):
        if self.restore is True:
            self._store[self.name] = self.prev_value
        else:
            del self._store[self.name]
        del self.prev_value


class BuiltinStoreSet(StoreSet):
    
    def __init__(self, __name: str, __value: Any, *, restore: bool = True) -> None:
        super().__init__(__name, __value, restore=restore, key='builtin__')


store = Store()

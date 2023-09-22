from typing import Any
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
    NoneOrNothing
)
import threading
import os
import sys
from torchslime.utils.bases import Base
from torchslime.utils.decorators import Singleton, ItemAttrBinding, ContextDecoratorBinding, RemoveOverload

_T = TypeVar('_T')


class StoreListener:
    def value_change__(self, new_value: Any, old_value: Any, key: str) -> None: pass

class SimpleStoreListener(StoreListener):
    
    def __init__(self, __func: Callable[[Any, Any, str], None]) -> None:
        super().__init__()
        self.func = __func
    
    def value_change__(self, new_value: Any, old_value: Any, key: str) -> None:
        return self.func(new_value, old_value, key)


class ScopedStore(Base):
    
    def __init__(self) -> None:
        super().__init__()
        self.__listener_dict: Dict[str, List[StoreListener]] = {}
    
    def add_listener__(self, __key: str, __listener: StoreListener, *, init: bool = False) -> None:
        if __key in self.__listener_dict:
            return self.__listener_dict[__key].append(__listener)
        else:
            self.__listener_dict[__key] = [__listener]
        
        if init:
            value = getattr(self, __key)
            __listener.value_change__(value, NOTHING, __key)
    
    def remove_listener__(self, __key: str, __listener: StoreListener) -> None:
        if __key not in self.__listener_dict:
            return
        
        listeners = self.__listener_dict[__key]
        if __listener not in listeners:
            return
        
        listeners.remove(__listener)
        if len(listeners) < 1:
            del self.__listener_dict[__key]
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in self.__listener_dict:
            return super().__setattr__(__name, __value)
        else:
            old_value = getattr(self, __name)
            super().__setattr__(__name, __value)
            # listener is called only when the new value is different from the old value
            if __value is not old_value:
                for listener in self.__listener_dict[__name]:
                    listener.value_change__(__value, old_value, __name)


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
        self.log_template_with_color: Union[str, NoneOrNothing] = NOTHING
        self.log_dateformat: Union[str, NoneOrNothing] = NOTHING

_builtin_scoped_store = BuiltinScopedStore()

_scoped_store_dict = {}

@ItemAttrBinding
@Singleton
@RemoveOverload(checklist=['add_listener__', 'remove_listener__'])
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
    def add_listener__(self, __key: str, __listener: StoreListener, *, init: bool = False) -> None: pass
    @overload
    def remove_listener__(self, __key: str, __listener: StoreListener) -> None: pass


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

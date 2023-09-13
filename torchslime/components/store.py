from torchslime.utils.typing import (
    NOTHING,
    Any,
    TypeVar,
    is_none_or_nothing,
    overload,
    is_slime_naming
)
import threading
import os
from torchslime.utils.bases import Base
from torchslime.utils.decorators import Singleton, ItemAttrBinding, ContextDecoratorBinding

_T = TypeVar('_T')


class ScopedStore(Base):
    
    def __init__(self) -> None:
        super().__init__()


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

_builtin_scoped_store = BuiltinScopedStore()

_scoped_store_dict = {}

@ItemAttrBinding
@Singleton
class Store:
    
    def scope__(self, __key) -> ScopedStore:
        if __key in _scoped_store_dict:
            return _scoped_store_dict[__key]
        else:
            return _scoped_store_dict.setdefault(__key, ScopedStore())

    def current__(self) -> ScopedStore:
        return self.scope__(self.get_current_key__())

    def builtin__(self) -> BuiltinScopedStore:
        return _builtin_scoped_store

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


@ContextDecoratorBinding
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


store = Store()

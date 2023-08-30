from torchslime.utils.typing import (
    Any,
    TypeVar,
    overload
)
import threading
import os
from torchslime.utils.bases import Base, NOTHING, is_none_or_nothing
from torchslime.utils.decorators import Singleton, ItemAttrBinding, ObjectAttrBinding, ContextDecoratorBinding
from torchslime.utils import is_slime_naming

_T = TypeVar('_T')


class ScopedStore(Base):

    def __init__(self, key__) -> None:
        super().__init__()
        self.key__ = key__

# 
# Store, builtin ScopedStore and store_dict
#

_builtin_scoped_store = ScopedStore('builtin__')
_store_dict = {
    'builtin__': _builtin_scoped_store
}

@ObjectAttrBinding
@ItemAttrBinding
@Singleton
class Store:
    
    def scope__(self, __key) -> ScopedStore:
        if __key in _store_dict:
            return _store_dict[__key]
        else:
            return _store_dict.setdefault(__key, ScopedStore(key__=__key))

    def current__(self) -> ScopedStore:
        return self.scope__(self.get_current_key__())

    def builtin__(self) -> ScopedStore:
        return self.scope__('builtin__')

    def destroy__(self, __key=NOTHING):
        if is_none_or_nothing(__key):
            __key = self.get_current_key__()
        
        if __key in _store_dict:
            del _store_dict[__key]

    # original object operation, just for type hint
    @overload
    def object_set__(self, __name: str, __value: Any) -> None: pass
    @overload
    def object_get__(self, __name: str) -> Any: pass
    @overload
    def object_del__(self, __name: str) -> None: pass

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
        self.restore_value = self._store[self.name]
        # set value
        self._store[self.name] = self.value

    def _restore_value(self):
        if self.restore is True:
            self._store[self.name] = self.restore_value
        else:
            del self._store[self.name]
        del self.restore_value


store = Store()

"""set ``builtin__`` store config"""
# whether to save log metadata (e.g., exec_name, lineno, etc.) to cache.
store.builtin__().use_log_cache = True
store.builtin__().log_cache = Base()
# flag to log only once. For example, some warnings may appear only once.
store.builtin__().log_once = Base()
# whether to use call debug
store.builtin__().use_call_debug = False
store.builtin__().call_debug_cache = Base()
# indent str for CLI display
store.builtin__().indent_str = ' ' * 4  # default is 4 spaces
# TODO: cli flag

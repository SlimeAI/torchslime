from torchslime.utils.bases import Base
from functools import wraps
from typing import Any, Union, TypeVar, Callable
import threading
import os
from torchslime.utils.bases import NOTHING, is_none_or_nothing

from torchslime.utils.decorators import Singleton


class StoreScope(Base):

    def __init__(self, key__) -> None:
        super().__init__()
        self.key__ = key__


@Singleton
class Store:
    
    def scope__(self, __key) -> StoreScope:
        if __key in store_dict:
            return store_dict[__key]
        else:
            return store_dict.setdefault(__key, StoreScope(key__=__key))

    def current__(self) -> StoreScope:
        return self.scope__(self.get_current_key__())

    def destroy__(self, __key=NOTHING):
        if is_none_or_nothing(__key):
            __key = self.get_current_key__()
        
        if __key in store_dict:
            del store_dict[__key]

    def __getitem__(self, __name: str):
        return self.current__()[__name]
    
    def __setitem__(self, __name: str, __value: Any):
        self.current__()[__name] = __value
    
    def __delitem__(self, __name: str):
        del self.current__()[__name]

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.current__(), __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        setattr(self.current__(), __name, __value)
    
    def __delattr__(self, __name: str) -> None:
        delattr(self.current__(), __name)
    
    @staticmethod
    def get_current_key__() -> str:
        return 'p{pid}-t{tid}'.format(
            pid=os.getpid(),
            tid=threading.get_ident()
        )


StoreSetSelf = TypeVar('StoreSetSelf', bound='StoreSet')


class StoreSet:

    def __init__(self, __name: str, __value: Any, *, restore: bool = True, key=NOTHING) -> None:
        self.name = __name
        self.value = __value
        self.restore = restore
        self.key = store.get_current_key__() if is_none_or_nothing(key) is True else key
        self._store = store.scope__(self.key)
        # get the store value before ``StoreSet``
        self.restore_value = self._store[__name]
    
    def __call__(self, func: Callable) -> Any:
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._set_value()
            result = func(*args, **kwargs)
            self._restore_value()
            return result
        return wrapper

    def __enter__(self) -> Union['StoreSet', StoreSetSelf]:
        self._set_value()
        return self

    def __exit__(self, *args, **kwargs):
        self._restore_value()
    
    def _set_value(self):
        self._store[self.name] = self.value

    def _restore_value(self):
        if self.restore is True:
            self._store[self.name] = self.restore_value
        else:
            del self._store[self.name]


# outer storage
store_dict = {}

store = Store()

"""set ``inner__`` store config"""
# whether to save log metadata (e.g., exec_name, lineno, etc.) to cache.
store.scope__('inner__').use_log_cache = True
store.scope__('inner__').log_cache = Base()
# flag to log only once. For example, some warnings may appear only once.
store.scope__('inner__').log_once = Base()
# whether to use call debug
store.scope__('inner__').use_call_debug = False
store.scope__('inner__').call_debug_cache = Base()

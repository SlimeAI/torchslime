from torchslime.utils import Singleton, Base
from typing import Any
import threading
import os


@Singleton
class Store:
    
    def __getitem__(self, __key) -> Base:
        if __key in store_dict:
            return store_dict[__key]
        else:
            return store_dict.setdefault(__key, Base())

    def current__(self) -> Base:
        return self[_get_store_key()]

    def __getattr__(self, __name: str) -> Any:
        return getattr(self[_get_store_key()], __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        setattr(self[_get_store_key()], __name, __value)
    
    def __delattr__(self, __name: str) -> None:
        delattr(self[_get_store_key()], __name)


# outer storage
store_dict = {}


def _get_store_key() -> str:
    return 'p{pid}-t{tid}'.format(
        pid=os.getpid(),
        tid=threading.get_ident()
    )


store = Store()

"""set ``inner__`` store config"""
# whether to save log metadata (e.g., exec_name, lineno, etc.) to cache.
store['inner__'].use_log_cache = True
store['inner__'].log_cache = Base()
# flag to log only once. For example, some warnings may appear only once.
store['inner__'].log_once = Base()

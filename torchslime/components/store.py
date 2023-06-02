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

    def __current__(self) -> Base:
        return self[_get_store_key()]

    def __getattr__(self, __name: str) -> Any:
        return getattr(self[_get_store_key()], __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        setattr(self[_get_store_key()], __name, __value)


# outer storage
store_dict = {}


def _get_store_key() -> str:
    return 'p{pid}-t{tid}'.format(
        pid=os.getpid(),
        tid=threading.get_ident()
    )


store = Store()

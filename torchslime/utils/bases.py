from torchslime.utils import dict_merge
import traceback
from typing import Any, Dict, Union
import threading
import multiprocessing


class Base:
    """
    Base class, making its subclasses be able to use '[]' operations(just like python dict).
    Return 'Nothing' if the object does not have the property being retrieved, without throwing Errors.
    What's more, it allows its subclasses assign properties using a dict.
    """

    def from_kwargs__(self, **kwargs):
        self.from_dict__(kwargs)

    def from_dict__(self, _dict: Dict):
        """assign properties to the object using a dict.
        Args:
            kwargs (Dict): property dict.
        """
        self.__dict__ = dict_merge(self.__dict__, _dict)

    def check__(self, item: str):
        """check whether the object has a specific attribute.
        dot operator supported.
        Args:
            items (str): _description_
        """
        attrs = item.split('.')
        temp = self
        for attr in attrs:
            try:
                temp = temp[attr]
                # if the value is NOTHING, then return False directly.
                if is_nothing(temp):
                    return False
            except Exception:
                # output error information
                self.process_exc__()
                return False
        return True

    def hasattr__(self, __name: str) -> bool:
        return str(__name) in self.__dict__

    @staticmethod
    def process_exc__():
        from torchslime.log import logger
        # output error
        logger.error(
            'Python exception raised:\n' +
            traceback.format_exc()
        )
        return NOTHING

    def pop__(self, __name: str):
        attr = getattr(self, __name)
        delattr(self, __name)
        return attr

    def __getattr__(self, *_):
        return NOTHING

    def __getattribute__(self, __name: str):
        return super().__getattribute__(str(__name))

    def __setattr__(self, __name: str, __value: Any) -> None:
        try:
            super().__setattr__(str(__name), __value)
        except Exception:
            return

    def __delattr__(self, __name: str) -> None:
        # safe delete
        try:
            super().__delattr__(str(__name))
        except Exception:
            return

    def __getitem__(self, __name: str):
        try:
            return getattr(self, __name)
        except Exception:
            return self.process_exc__()

    def __setitem__(self, __name: str, __value: Any):
        try:
            return setattr(self, __name, __value)
        except Exception:
            return self.process_exc__()

    def __delitem__(self, __name: str):
        try:
            return delattr(self, __name)
        except Exception:
            return


class BaseList:

    def __init__(self, list_like: Any = None):
        if is_none_or_nothing(list_like):
            self.__list = []
        else:
            # TODO: Iterable WARNING, BaseList only supports list or tuple expansion, other iterable items will be processed as ``[item]``
            self.__list = list(list_like) if isinstance(list_like, (list, tuple)) else [list_like]

    @classmethod
    def create__(
        cls,
        list_like: Any = None,
        return_none: bool = True,
        return_nothing: bool = True,
        return_ellipsis: bool = True
    ):
        """
        If the ``list_like`` object is ``None``, ``NOTHING`` or ``...`` and the corresponding return config is True, then
        return itself, otherwise return ``BaseList`` object.
        WARNING: This changes the default behavior of ``BaseList``, which creates an empty list when the list_like object is 
        ``None`` or ``NOTHING`` and creates ``[...]`` when the list_like object is ``...``.
        """
        if (is_nothing(list_like) and return_nothing is True) or \
            (list_like is None and return_none is True) or \
            (list_like is ... and return_ellipsis is True):
            return list_like
        else:
            return cls(list_like)

    def set_list__(self, _list: list):
        self.__list = _list

    def get_list__(self):
        return self.__list


class BaseDict:

    def __init__(self, _dict: Union[Dict, None, 'Nothing']):
        self.__dict = _dict if isinstance(_dict, (dict, Dict)) else {}

    def set_dict__(self, _dict: dict):
        self.__dict = _dict

    def get_dict__(self):
        return self.__dict

#
# Nothing class, NOTHING instance and related operations.
#

class _NothingSingleton(type):
    """
    Nothing Singleton should be implemented independently, because the ``Singleton`` decorator relies on the basic NOTHING object, which may cause circular reference.
    """

    __t_lock = threading.Lock()
    __p_lock = multiprocessing.Lock()
    __instance = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.__instance is None:
            with self.__t_lock, self.__p_lock:
                if self.__instance is None:
                    self.__instance = super().__call__(*args, **kwargs)
        return self.__instance

class Nothing(metaclass=_NothingSingleton):
    """
    'Nothing' object, different from python 'None'.
    It often comes from getting properties or items that the object does not have, or simply represents a default value.
    'Nothing' allows any attribute-get or method-call operations without throwing Errors, making the program more stable.
    It will show Warnings in the console instead.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self

    def __getattribute__(self, *_):
        return self

    def __getitem__(self, *_):
        return self

    def __setattr__(self, *_):
        pass

    def __setitem__(self, *_):
        pass

    def __len__(self):
        return 0

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __str__(self) -> str:
        return 'NOTHING'

    def __repr__(self) -> str:
        return 'NOTHING'

    def __format__(self, __format_spec: str) -> str:
        return 'NOTHING'

    def __contains__(self) -> bool:
        return False

    def __eq__(self, obj) -> bool:
        if is_nothing(obj):
            return True
        return False

    def __add__(self, _):
        return self

    def __sub__(self, _):
        return self

    def __mul__(self, _):
        return self

    def __truediv__(self, _):
        return self

    def __radd__(self, _):
        return self

    def __rsub__(self, _):
        return self

    def __rmul__(self, _):
        return self

    def __rtruediv__(self, _):
        return self

    def __float__(self):
        return 0.0

    def __bool__(self) -> bool:
        return False

NOTHING = Nothing()

def is_nothing(obj):
    """Check whether an object is an instance of 'Nothing'
    Args:
        obj (Any): object
    Returns:
        bool: whether the object is instance of 'Nothing'
    """
    return NOTHING is obj

def is_none_or_nothing(obj):
    """Check whether an object is None, Nothing or neither.
    Args:
        obj (Any): object
    Returns:
        bool: check result.
    """
    return obj is None or is_nothing(obj)

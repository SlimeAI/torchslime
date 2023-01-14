# TODO: refactor the util package
from typing import Dict, Union, Tuple, Sequence
from collections.abc import Iterator, Iterable
from .type import T_M_SEQ, T_M
from torch import Tensor
from torch.nn import Module
import threading
from functools import wraps
from time import time
import traceback
import inspect


def SmartWrapper(cls):
    """
    Smart wrapper that wraps functions and classes when using decorator.
    It is smarter than functools.wraps, for it can recognize whether the decorated item is a class or a function and then applies
    class wrapper or function wrapper respectively.
    When it is used to a function, the result is the same as functools.wraps,
    while when it is used to a class, you can get the original class by accessing the '_class' attribute,
    so you can use this feature to do other useful things, such as 'isinstance', etc.
    """
    def decorator(func):
        if inspect.isclass(cls):
            class Wrapper:
                def __init__(self, _class) -> None:
                    self._class = _class
                
                def __call__(self, *args, **kwargs) -> None:
                    return func(*args, **kwargs)
                
                def __repr__(self):
                    return "Smart wrapper object: {}. (You can get the original decorated class by accessing the attribute '_class')".format(super().__repr__())
                
                def __str__(self):
                    return "Smart wrapper object: {}. (You can get the original decorated class by accessing the attribute '_class')".format(super().__repr__())
            return wraps(cls)(Wrapper(cls))
        elif inspect.isfunction(cls) or inspect.ismethod(cls):
            @wraps(cls)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
    return decorator


def Singleton(cls):
    """
    Decorator that makes decorated classes singleton.
    It makes the creation of the singleton object thread-safe by using double-checked locking.
    """
    _lock = threading.Lock()
    _instance = {}
    
    @SmartWrapper(cls)
    def wrapper(*args, **kwargs):
        if cls not in _instance:
            with _lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return wrapper


# set import here to avoid import error
from ..log import logger


def InvocationDebug(module_name):
    """A decorator that output debug information before and after a method is invoked.

    Args:
        func (_type_): _description_
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(module_name, 'begin.')
            result = func(*args, **kwargs)
            logger.debug(module_name, 'end.')
            return result
        return wrapper
    return decorator


@Singleton
class Nothing:
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


NOTHING = Nothing()


def is_nothing(obj):
    """Check whether an object is an instance of 'Nothing'

    Args:
        obj (Any): object

    Returns:
        bool: whether the object is instance of 'Nothing'
    """
    return NOTHING is obj


def check_nothing(obj, x, y=NOTHING):
    return x if is_nothing(obj) is False else y


def dict_merge(dict1: Dict, dict2: Dict):
    return { **dict1, **dict2 }


def safe_divide(dividend, divisor):
    return dividend / divisor if divisor != 0 else 0


class Base:
    """
    Base class, making its subclasses be able to use '[]' operations(just like python dict).
    Return 'Nothing' if the object does not have the property being retrieved, without throwing Errors.
    What's more, it allows its subclasses assign properties using a dict.
    """

    def from_dict(self, kwargs: Dict):
        """assign properties to the object using a dict.

        Args:
            kwargs (Dict): property dict.
        """
        self.__dict__ = dict_merge(self.__dict__, kwargs)

    def check(self, item: str):
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
                # output error infomation
                self.process_exc()
                return False
        return True

    @staticmethod
    def process_exc():
        # output error
        logger.error(
            'Python exception raised:\n' +
            traceback.format_exc()
        )
        return NOTHING

    def __getattr__(self, *_):
        return NOTHING

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except Exception:
            return self.process_exc()
    
    def __setitem__(self, key, value):
        try:
            return setattr(self, key, value)
        except Exception:
            return self.process_exc()

    def __getattribute__(self, key):
        return super().__getattribute__(key)
    
    def __delattr__(self, __name: str) -> None:
        # safe delete
        try:
            super().__delattr__(__name)
        except Exception:
            return


class SingleConst:
    """
    A class that defines a const value that cannot be changed.
    Show Warnings when the value is attempted to be changed, and the change won't actually take effect.
    Allow assigning the value to 'Nothing' when initialized, which means that you will assign the value later.
    Once the value is not 'Nothing', it will never be changed.

    *****
    Note that the class property cannot be changed means every instance of the class shares the same property.
    Not suitable for the property that varies from instance to instance.
    *****
    """

    def __init__(self, value=NOTHING):
        # the default value will refer to the same 'Nothing'.
        super().__init__()
        self.value = value

    def __set__(self, _, value):
        # the value can be changed only when it's 'Nothing'
        if is_nothing(self.value):
            self.value = value
        else:
            # TODO: show warnings
            print('the value cannot be changed.')

    def __get__(self, *_):
        return self.value


class MultiConst:
    """
    Const constraint that varies from instances. Similar to 'SingleConst'.

    *****
    WARNING:
    MultiConst is not in strict mode. Actually, for convenience, MultiConst will create a private property of
    the same name using prefix '_' in the object. Thus it won't be able to prevent the direct assignment to 
    the private property it creates.

    e.g.
    class Example:
        attr = MultiConst()
    
    ex = Example()
    ex.attr = 'a' # OK
    ex.attr = 'b' # FAIL, cannot assign new value to a const value.
    ex.attr # Now the property value is still 'a'.
    ex._attr = 'c' # **** WARNING: OK, because the MultiConst cannot prevent assignment to this property. ****
    ex.attr # Now the property value has become 'c'.
    *****

    *****
    WARNING:
    Do not use MultiConst to a property named already with prefix '_', for MultiConst will create a new property
    with '__' as a prefix, making the attribute unaccessible through '__foo' outside the class.
    *****
    """
    def __init__(self):
        super().__init__()
    
    def __set_name__(self, _, name):
        self.private_name = '_%s' % str(name)

    def __set__(self, instance, value):
        temp = getattr(instance, self.private_name, NOTHING)
        if is_nothing(temp):
            setattr(instance, self.private_name, value)
        else:
            # TODO: show warnings
            print('the value cannot be changed')

    def __get__(self, instance, _):
        return getattr(instance, self.private_name, NOTHING)


class Count(SingleConst):

    def __init__(self):
        super().__init__(0)

    def __get__(self, *_):
        tmp = self.value
        self.value += 1
        return tmp


class BaseList(list):

    def __init__(self, list_like: Iterable=None):
        if list_like is None or is_nothing(list_like):
            super().__init__()
        else:
            super().__init__(list_like if isinstance(list_like, Iterable) else [list_like])


def get_device(obj: T_M):
    """Get the device of the model or tensor.

    Args:
        obj (T_M): model or tensor

    Returns:
        device: the device
    """
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.device if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.device
    else:
        return None


def get_dtype(obj: T_M):
    """Get the data type of the model or tensor

    Args:
        obj (T_M): model or tensor

    Returns:
        data type: the data type
    """
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.dtype if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.dtype
    else:
        return None


def type_cast(obj: T_M_SEQ, device=None, dtype=None) -> Union[Tuple[Tensor, Module], Tensor, Module, None]:
    """Apply type cast to the model or tensor.

    Args:
        obj (T_M_SEQ): tensor, model, list of tensor or list of model
        device ([type], optional): device. Defaults to None.
        dtype ([type], optional): dtype. Defaults to None.

    Returns:
        Union[Tuple[Tensor, Module], Tensor, Module, None]: [description]
    """
    obj = obj if isinstance(obj, (list, tuple)) else ((obj, ) if isinstance(obj, (Tensor, Module)) else obj)
    if isinstance(obj, (list, tuple)) is False:
        return obj
    if device is not None:
        obj = [item.to(device=device) for item in obj]
    if dtype is not None:
        obj = [item.to(dtype=dtype) for item in obj]
    obj = tuple(obj)
    return obj if len(obj) > 1 else obj[0]


def list_take(list_like, index: Union[Sequence[int], int]):
    """Get item or sub list of the list_like object through index(es).

    Args:
        list_like (_type_): list-like object
        index (Union[Sequence[int], int]): the index(es) to be taken.

    Returns:
        _type_: single item or list.
    """
    if index is None:
        return NOTHING
    # convert non-list item to list.
    if isinstance(list_like, (list, tuple)) is False:
        list_like = (list_like,)
    
    list_len = len(list_like)
    # take item(s).
    if isinstance(index, int):
        # return nothing if the index is out of bounds.
        return list_like[index] if index < list_len else NOTHING
    elif isinstance(index, (list, tuple)):
        return tuple(list_like[i] if i < list_len else NOTHING for i in index)


def MethodChaining(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper


class Iter:

    def __init__(self, _iterable):
        # iterable item
        self._iterable = _iterable
        # iterator
        self._iterator = None

    def __iter__(self):
        # get iterator
        self._iterator = iter(self._iterable)
        return self

    def __next__(self):
        if isinstance(self._iterator, Iterator):
            # get next
            return next(self._iterator)
        else:
            raise StopIteration


class IterTool(Iter):

    def __init__(self, _iterable, progress=False, time=False, index=False, total=False):
        super().__init__(_iterable)
        # iteration index
        self._index = 0
        # additional information in iteration
        self.items = [progress, time, index, total]
        self.func_set = [self.progress, self.time, self.index, self.total]

    def __iter__(self):
        super().__iter__()
        # set index to 0
        self._index = 0
        return self

    def __next__(self):
        # get the next item
        item = super().__next__()
        # get needed information indexes
        indexes = [index for index, value in enumerate(self.items) if value is True]
        # func set result
        func_set_res = [func() for func in list_take(self.func_set, indexes)]
        # index increases by 1(this should be done after the current index is accessed)
        self._index += 1
        return item if len(func_set_res) == 0 else (item, *func_set_res)

    def __len__(self):
        try:
            return len(self._iterable)
        except Exception:
            logger.error('The iterable item has no __len__.')
            return 0

    def progress(self):
        return self._index, self.__len__()

    def time(self):
        return time()

    def index(self):
        return self._index

    def total(self):
        return self.__len__()


def count_params(model: Module, format: str = None, decimal: int = 2):
    format_dict = {
        None: 1,
        'K': 1000,
        'M': 1000000
    }
    divisor = format_dict.get(format, 1)

    num = 0
    for param in model.parameters():
        num += param.numel()
    result = num / divisor
    return result if format is None else ('{0:.' + str(decimal) + 'f}{1}').format(result, format)

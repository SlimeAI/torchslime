# TODO: refactor the util package
from .typing import (
    Dict,
    Union,
    Tuple,
    Sequence,
    Iterator,
    Iterable,
    Any,
    TypeVar,
    TYPE_CHECKING
)
from time import time
import inspect
import os
if TYPE_CHECKING:
    from .typing import (
        TorchModule,
        TorchTensor,
        TorchTensorOrModule,
        TorchTensorOrModuleOrSequence
    )

_T = TypeVar('_T')


def get_exec_info(obj):
    exec_name = inspect.getmodule(obj).__name__
    full_exec_name = os.path.abspath(inspect.getfile(obj))
    lineno = inspect.getsourcelines(obj)[1]
    _exec = {
        'exec_name': exec_name,
        'full_exec_name': full_exec_name,
        'lineno': lineno
    }
    return _exec


def bound_clip(value, _min, _max):
    """clip ``value`` between ``_min`` and ``_max``(including the boundary). Return ``NOTHING`` if ``_min > _max``

    Args:
        value: value to be clipped.
        _min: min value.
        _max: max value.

    Returns:
        ``Number`` or ``NOTHING``
    """
    if _min > _max:
        return NOTHING
    return _min if value < _min else \
        _max if value > _max \
        else value


def dict_merge(dict1: Dict, dict2: Dict):
    return { **dict1, **dict2 }


def safe_divide(dividend, divisor, default=0):
    return dividend / divisor if divisor != 0 else default


def inf_range(start: int = 0, step: int = 1):
    value = start
    while True:
        yield value
        value += step


def inf_iter(__iterable: Iterable):
    while True:
        for item in __iterable:
            yield item


def inf_enumerate(__iterable: Iterable, start: int = 0):
    for item in enumerate(inf_iter(__iterable), start=start):
        yield item


def get_device(obj: "TorchTensorOrModule"):
    """Get the device of the model or tensor.

    Args:
        obj (T_M): model or tensor

    Returns:
        device: the device
    """
    from torch.nn import Module
    from torch import Tensor
    
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.device if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.device
    else:
        return None


def get_dtype(obj: "TorchTensorOrModule"):
    """Get the data type of the model or tensor

    Args:
        obj (T_M): model or tensor

    Returns:
        data type: the data type
    """
    from torch.nn import Module
    from torch import Tensor
    
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.dtype if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.dtype
    else:
        return None


def type_cast(
    obj: "TorchTensorOrModuleOrSequence",
    device=None,
    dtype=None
) -> Union[Tuple["TorchTensor", "TorchModule"], "TorchTensor", "TorchModule", None]:
    """Apply type cast to the model or tensor.

    Args:
        obj (T_M_SEQ): tensor, model, list of tensor or list of model
        device ([type], optional): device. Defaults to None.
        dtype ([type], optional): dtype. Defaults to None.

    Returns:
        Union[Tuple[Tensor, Module], Tensor, Module, None]: [description]
    """
    from torch.nn import Module
    from torch import Tensor
    
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
            from torchslime.logging.logger import logger
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


def count_params(model: "TorchModule", format: str = None, decimal: int = 2):
    format_dict = {
        None: 1,
        'K': 1e3,
        'M': 1e6,
        'B': 1e9
    }
    divisor = format_dict.get(format, 1)

    num = 0
    for param in model.parameters():
        num += param.numel()
    result = num / divisor
    return result if format is None else f'{result:.{decimal}f}{format}'


def is_torch_distributed_ready():
    """
    Check whether the torch distributed settings are ready.
    """
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


class StrTemplate:

    pass


def xor__(__x, __y) -> bool:
    return bool((__x and not __y) or (not __x and __y))


class LessThanAnything:
    
    def __lt__(self, __value: Any) -> bool: return True
    def __le__(self, __value: Any) -> bool: return True
    def __eq__(self, __value: Any) -> bool: return False
    def __gt__(self, __value: Any) -> bool: return False
    def __ge__(self, __value: Any) -> bool: return False


class GreaterThanAnything:
    
    def __lt__(self, __value: Any) -> bool: return False
    def __le__(self, __value: Any) -> bool: return False
    def __eq__(self, __value: Any) -> bool: return False
    def __gt__(self, __value: Any) -> bool: return True
    def __ge__(self, __value: Any) -> bool: return True


def window_iter(__sequence: Sequence[_T], window_size: int = 1, step: int = 1) -> Iterator[Tuple[_T]]:
    if window_size < 1 or step < 1:
        raise ValueError('``window_size`` and ``step`` should be integers not less than 1.')
    
    import math
    max_index = math.floor((len(__sequence) - window_size) / step)
    # index start from 0, so it should be ``max_index + 1``
    for i in range(max_index + 1):
        yield tuple(__sequence[i * step : i * step + window_size])


from torchslime.utils.typing import NOTHING


def get_len(__obj: Any, *, default: _T = NOTHING) -> Union[int, _T]:
    try:
        return len(__obj)
    except TypeError:
        return default

import os
import inspect
from .typing import (
    NOTHING,
    Sequence,
    Union,
    Tuple,
    Dict,
    Iterator,
    TypeVar,
    NoneOrNothing,
    is_none_or_nothing,
    Any
)
from slime_core.utils.common import *
from torch.nn import Module
from torch import Tensor

_T = TypeVar("_T")

#
# dict and list formatter
#

def concat_format(
    __left: str,
    __content: Sequence[str],
    __right: str,
    *,
    item_sep: str = ',',
    indent_prefix: Union[str, NoneOrNothing] = NOTHING,
    break_line: bool = True
) -> str:
    if len(__content) < 1:
        # empty content: simply concat ``__left`` and ``__right``
        return __left + __right

    break_line_sep = '\n'
    if not break_line:
        indent_prefix = ''
    elif is_none_or_nothing(indent_prefix):
        from .store import store
        indent_prefix: str = store.builtin__().indent_str
    # format content
    content_sep = item_sep + (break_line_sep if break_line else '')
    __content = indent(content_sep.join(__content), prefix=indent_prefix)
    # format concat
    concat_sep = break_line_sep if break_line else ''
    return concat_sep.join([__left, __content, __right])


def iterable(__obj: Any) -> bool:
    try:
        iter(__obj)
    except Exception:
        return False
    else:
        return True


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
    return (
        _min if value < _min else 
        _max if value > _max 
        else value
    )


def dict_merge(dict1: Dict, dict2: Dict):
    return { **dict1, **dict2 }


def safe_divide(dividend, divisor, default=0):
    return dividend / divisor if divisor != 0 else default


def inf_range(start: int = 0, step: int = 1):
    value = start
    while True:
        yield value
        value += step


def get_device(obj: Union[Tensor, Module]):
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


def get_dtype(obj: Union[Tensor, Module]):
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


def type_cast(
    obj: Union[Tensor, Module, Sequence[Tensor], Sequence[Module]],
    device=None,
    dtype=None
) -> Union[Tuple[Tensor, Module], Tensor, Module, None]:
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


def count_params(model: Module, format: str = None, decimal: int = 2):
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


class StrTemplate:

    pass


def xor__(__x, __y) -> bool:
    return bool((__x and not __y) or (not __x and __y))


# TODO: refactor it to ``typing`` and make it a flag constant?
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


def window_iter(
    __sequence: Sequence[_T],
    window_size: int = 1,
    step: int = 1
) -> Iterator[Tuple[_T, ...]]:
    if window_size < 1 or step < 1:
        raise ValueError('``window_size`` and ``step`` should be integers not less than 1.')

    import math
    max_index = math.floor((len(__sequence) - window_size) / step)
    # index start from 0, so it should be ``max_index + 1``
    for i in range(max_index + 1):
        yield tuple(__sequence[i * step : i * step + window_size])


def get_len(__obj: Any, *, default: _T = NOTHING) -> Union[int, _T]:
    try:
        return len(__obj)
    except TypeError:
        return default

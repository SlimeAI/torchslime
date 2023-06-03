# TODO: refactor the util package
from typing import Dict, Union, Tuple, Sequence, MutableSequence, Generic, TypeVar, \
    overload, Iterator, Iterable, Any, List
from torch import Tensor
import torch
from torch.nn import Module
import threading
import multiprocessing
from functools import wraps
from time import time
import traceback
import inspect
import pickle
import io
import os


def SmartWraps(cls):
    """
    Smart wrapper that wraps functions and classes when using decorator.
    It is smarter than functools.wraps, for it can recognize whether the decorated item is a class or a
    function and then applies class wrapper or function wrapper respectively.
    When it is used to a function, the result is the same as functools.wraps,
    while when it is used to a class, you can get the original class by accessing the '_wrapped_class' attribute,
    so you can use this feature to do other useful things, such as 'isinstance', etc.

    WARNING: DO NOT use ``_wrapped`` as attribute name in the decorated class.
    """
    def decorator(func):
        if inspect.isclass(cls):
            _wrapped = cls._wrapped if hasattr(cls, '_wrapped') else cls
            class Wrapper(metaclass=_get_wrapper_type(_wrapped)):
                def __new__(_cls: type, *args, **kwargs):
                    obj = func(*args, **kwargs)
                    return obj
            return _update_class_wrapper(_wrapped)(Wrapper)
        elif inspect.isfunction(cls) or inspect.ismethod(cls):
            @wraps(cls)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
    return decorator


def _update_class_wrapper(_wrapped: type):
    WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__doc__')
    WRAPPER_UPDATES = ('__dict__',)

    def _super_setattr(__wrapper, __name, __value):
        super(__wrapper.__class__, __wrapper).__setattr__(__name, __value)

    def partial_func(_wrapper):
        for attr in WRAPPER_ASSIGNMENTS:
            if hasattr(_wrapped, attr):
                _super_setattr(_wrapper, attr, getattr(_wrapped, attr))
        for attr in WRAPPER_UPDATES:
            for key, value in getattr(_wrapped, attr, {}).items():
                if key == '__dict__':
                    continue
                _super_setattr(_wrapper, key, value)
        return _wrapper
    return partial_func


def _get_wrapper_type(_wrapped: type):
    
    class WrapperType(type):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            super().__setattr__('_wrapped', _wrapped)
        
        def __setattr__(self, __name: str, __value: Any) -> None:
            setattr(getattr(self, '_wrapped', NOTHING), __name, __value)
            return super().__setattr__(__name, __value)
        
        def __new__(cls, name, bases, attrs):
            # copy bases from the wrapped class
            bases = _wrapped.__bases__
            return super().__new__(cls, name, bases, attrs)
    
    return WrapperType


def Singleton(cls):
    """
    Decorator that makes decorated classes singleton.
    It makes the creation of the singleton object thread-safe by using double-checked locking.
    """
    t_lock = threading.Lock()
    p_lock = multiprocessing.Lock()
    _instance = {}
    
    @SmartWraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in _instance:
            with t_lock, p_lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return wrapper


def InvocationDebug(module_name):
    """A decorator that output debug information before and after a method is invoked.

    Args:
        func (_type_): _description_
    """
    def decorator(func):
        # cache inspect result
        from torchslime.log import logger
        _exec_info = get_exec_info(func)

        @SmartWraps(func)
        def wrapper(*args, **kwargs):
            logger.debug('{} begins.'.format(module_name), _exec_info=_exec_info)
            result = func(*args, **kwargs)
            logger.debug('{} ends.'.format(module_name), _exec_info=_exec_info)
            return result
        return wrapper
    return decorator


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


def dict_merge(dict1: Dict, dict2: Dict):
    return { **dict1, **dict2 }


def safe_divide(dividend, divisor, default=0):
    return dividend / divisor if divisor != 0 else default


class Base:
    """
    Base class, making its subclasses be able to use '[]' operations(just like python dict).
    Return 'Nothing' if the object does not have the property being retrieved, without throwing Errors.
    What's more, it allows its subclasses assign properties using a dict.
    """

    def update__(self, **kwargs):
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

    def __getattribute__(self, __name: str):
        return super().__getattribute__(__name)
    
    def __delattr__(self, __name: str) -> None:
        # safe delete
        try:
            super().__delattr__(__name)
        except Exception:
            return


class Count:
    """
    Count times of variable-get.
    """

    def __init__(self):
        super().__init__()
        self.value = 0

    def __set__(self, *_):
        pass

    def __get__(self, *_):
        tmp = self.value
        self.value += 1
        return tmp


class BaseList:

    def __init__(self, list_like: Any = None):
        if is_none_or_nothing(list_like):
            self.__list = []
        else:
            # TODO: Iterable WARNING, BaseList only supports list or tuple expansion, other iterable items will be processed as ``[item]``
            self.__list = list(list_like) if isinstance(list_like, (list, tuple)) else [list_like]

    @classmethod
    def create(
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

    def set_list(self, _list: list):
        self.__list = _list
    
    def get_list(self):
        return self.__list

    """
    List operation adapter.
    """
    @overload
    def index(self, __value) -> int: pass
    @overload
    def index(self, __value, __start) -> int: pass
    @overload
    def index(self, __value, __start, __stop) -> int: pass
    @overload
    def pop(self): pass
    @overload
    def pop(self, __index): pass
    
    def append(self, __object) -> None:
        return self.__list.append(__object)
    
    def clear(self) -> None:
        return self.__list.clear()
    
    def copy(self):
        return self.__list.copy()
    
    def count(self, __value) -> int:
        return self.__list.count(__value)
    
    def extend(self, __iterable: Iterable) -> None:
        return self.__list.extend(__iterable)
    
    def index(self, *args) -> int:
        return self.__list.index(*args)
    
    def insert(self, __index, __object) -> None:
        return self.__list.insert(__index, __object)
    
    def pop(self, *args):
        return self.__list.pop(*args)
    
    def remove(self, __value) -> None:
        return self.__list.remove(__value)
    
    def reverse(self) -> None:
        return self.__list.reverse()
    
    def __setitem__(self, __i_s, __o):
        return self.__list.__setitem__(__i_s, __o)
    
    def __getitem__(self, __i_s):
        return self.__list.__getitem__(__i_s)

    def __contains__(self, __o: object) -> bool:
        return self.__list.__contains__(__o)

    def __len__(self) -> int:
        return self.__list.__len__()
    
    def __delitem__(self, __i) -> None:
        return self.__list.__delitem__(__i)
    
    def __iadd__(self, __x: Iterable):
        return self.__list.__iadd__(__x)
    
    def __imul__(self, __n):
        return self.__list.__imul__(__n)
    
    def __iter__(self) -> Iterator:
        return self.__list.__iter__()
    
    def __add__(self, __x: list) -> list:
        return self.__list.__add__(__x)
    
    def __reversed__(self) -> Iterator:
        return self.__list.__reversed__()

    def __mul__(self, __n) -> list:
        return self.__list.__mul__(__n)

    def __rmul__(self, __n):
        return self.__list.__rmul__(__n)

    def __str__(self) -> str:
        return self.__list.__str__()

    def __repr__(self) -> str:
        return self.__list.__repr__()


class BaseDict:

    def __init__(self, _dict: Union[Dict, None, Nothing]):
        self.__dict = _dict if isinstance(_dict, (dict, Dict)) else {}

    def set_dict(self, _dict: dict):
        self.__dict = _dict
    
    def get_dict(self):
        return self.__dict

    """
    Dict operation adapter.
    """
    @overload
    def setdefault(self, __key): pass
    @overload
    def setdefault(self, __key, __default): pass
    @overload
    def get(self, __key): pass
    @overload
    def get(self, __key, __default): pass
    @overload
    def pop(self, __key): pass
    @overload
    def pop(self, __key, __default): pass

    def copy(self):
        return self.__dict.copy()

    def keys(self):
        return self.__dict.keys()

    def values(self):
        return self.__dict.values()

    def items(self):
        return self.__dict.items()
    
    def clear(self):
        return self.__dict.clear()

    def update(self, *args, **kwargs):
        return self.__dict.update(*args, **kwargs)
    
    def setdefault(self, *args):
        return self.__dict.setdefault(*args)

    def get(self, *args):
        return self.__dict.get(*args)

    def pop(self, *args):
        return self.__dict.pop(*args)

    def __len__(self) -> int:
        return self.__dict.__len__()
    
    def __getitem__(self, __key):
        return self.__dict.__getitem__(__key)
    
    def __setitem__(self, __key, __value) -> None:
        return self.__dict.__setitem__(__key, __value)
    
    def __delitem__(self, __key) -> None:
        return self.__dict.__delitem__(__key)

    def __iter__(self) -> Iterator:
        return self.__dict.__iter__()

    def __contains__(self, __o: object) -> bool:
        return self.__dict.__contains__(__o)
    
    def __str__(self) -> str:
        return self.__dict.__str__()
    
    def __repr__(self) -> str:
        return self.__dict.__repr__()


def inf_range(start: int = 0, step: int = 1):
    value = start
    while True:
        yield value
        value += step


class TorchComm:

    def __init__(self) -> None:
        self._pickler = pickle.Pickler
        self._unpickler = pickle.Unpickler

    def gather(self, tensor: Tensor, dst=0, group=None, async_op=False):
        import torch.distributed as dist
        device = self._get_device(group=group)
        group_size = dist.get_world_size(group=group)
        # get GLOBAL RANK here
        rank = dist.get_rank()
        # get ``tensor_size``
        tensor_size = tuple(tensor.size())
        tensor_list: List[Tensor] = self._make_tensor_group_list(
            tensor_size, group_size, tensor.dtype, device
        ) if rank == dst else None
        work = dist.gather(tensor.to(device), tensor_list, dst=dst, group=group, async_op=async_op)
        if async_op is True:
            return tensor_list, work
        return tensor_list
    
    def gather_object(self, obj, dst=0, group=None):
        # code modified from torch.distributed.gather_object in PyTorch 1.13
        import torch.distributed as dist
        device = self._get_device(group=group)
        object_tensor, local_size = self._object_to_tensor(obj, device)
        group_size = dist.get_world_size(group=group)
        # get GLOBAL RANK here
        rank = dist.get_rank()
        # object sizes
        object_size_list = self._all_gather_size(local_size, group_size, device, group)
        # get max object size
        max_object_size = int(max(object_size_list).item())
        # resize object tensor to max size
        object_tensor.resize_(max_object_size)
        # output object tensors
        output_tensors = self._make_tensor_group_list(
            max_object_size, group_size, dtype=torch.uint8, device=device
        ) if rank == dst else None
        dist.gather(object_tensor, gather_list=output_tensors, dst=dst, group=group)
        # return ``None`` if current rank is not destination rank
        if rank != dst:
            return
        return self._transfer_objects(output_tensors, object_size_list, group_size)

    def all_gather(self, tensor: Tensor, group=None, async_op=False):
        import torch.distributed as dist
        device = self._get_device(group=group)
        group_size = dist.get_world_size(group=group)
        # get ``tensor_size``
        tensor_size = tuple(tensor.size())
        tensor_list: List[Tensor] = self._make_tensor_group_list(tensor_size, group_size, tensor.dtype, device)
        work = dist.all_gather(tensor_list, tensor.to(device), group=group, async_op=async_op)
        if async_op is True:
            return tensor_list, work
        return tensor_list

    def all_gather_object(self, obj, group=None):
        # code modified from torch.distributed.all_gather_object in PyTorch 1.13
        import torch.distributed as dist
        device = self._get_device(group=group)
        object_tensor, local_size = self._object_to_tensor(obj, device)
        group_size = dist.get_world_size(group=group)
        # object sizes
        object_size_list = self._all_gather_size(local_size, group_size, device, group)
        # get max object size
        max_object_size = int(max(object_size_list).item())
        # resize object tensor to max size
        object_tensor.resize_(max_object_size)
        # output object tensors
        output_tensors = self._make_tensor_group_list(
            max_object_size, group_size, dtype=torch.uint8, device=device
        )
        # all gather object tensors
        dist.all_gather(output_tensors, object_tensor, group=group)
        return self._transfer_objects(output_tensors, object_size_list, group_size)

    def broadcast(self, tensor, src=0, group=None, async_op=False):
        # this API is simple enough that does not need more adaptation
        import torch.distributed as dist
        return dist.broadcast(tensor, src, group=group, async_op=async_op)

    def broadcast_object(self, obj, src=0, group=None):
        # code modified from torch.distributed.broadcast_object_list in PyTorch 1.13
        import torch.distributed as dist
        device = self._get_device(group=group)
        # get GLOBAL RANK here
        rank = dist.get_rank()
        if rank == src:
            object_tensor, local_size = self._object_to_tensor(obj, device)
        else:
            object_tensor, local_size = None, torch.zeros(1, dtype=torch.long, device=device)
        # broadcast object size to all ranks
        dist.broadcast(local_size, src=src, group=group)
        if rank != src:
            object_tensor = torch.zeros(local_size.item(), dtype=torch.uint8, device=device)
        # broadcast object tensor to all ranks
        dist.broadcast(object_tensor, src=src, group=group)
        return self._tensor_to_object(object_tensor, object_tensor.numel())

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        # this API is simple enough that does not need more adaptation
        import torch.distributed as dist
        return dist.scatter(tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)

    def scatter_object(self, objs, src=0, group=None):
        # code modified from torch.distributed.scatter_object_list in PyTorch 1.13
        import torch.distributed as dist
        device = self._get_device(group=group)
        # get GLOBAL RANK here
        rank = dist.get_rank()
        if rank == src:
            object_tensors, local_sizes = zip(
                *[self._object_to_tensor(obj, device) for obj in objs]
            )
            object_tensors, local_sizes = list(object_tensors), list(local_sizes)
        
        if rank == src:
            # get max object size
            max_object_size: Tensor = max(local_sizes)
            for tensor in object_tensors:
                tensor.resize_(int(max_object_size.item()))
        else:
            max_object_size = torch.LongTensor([0]).to(device=device)
        dist.broadcast(max_object_size, src=src, group=group)

        local_size = torch.LongTensor([0]).to(device=device)
        dist.scatter(
            local_size,
            scatter_list=local_sizes if rank == src else None,
            src=src,
            group=group
        )

        object_tensor = torch.zeros(int(max_object_size.item()), dtype=torch.uint8, device=device)
        dist.scatter(
            object_tensor,
            scatter_list=object_tensors if rank == src else None,
            src=src,
            group=group
        )
        return self._tensor_to_object(object_tensor, local_size)

    def _all_gather_size(self, size_tensor, group_size: int, device, group):
        import torch.distributed as dist
        size_list = self._make_tensor_group_list(1, group_size, dtype=torch.long, device=device)
        # gather object sizes into ``object_size_list``
        dist.all_gather(size_list, size_tensor.type(torch.long).to(device), group=group)
        return size_list

    def _transfer_objects(self, output_tensors, object_size_list, group_size):
        # The unpickled objects are gathered in ``object_list``
        object_list = [NOTHING for _ in range(group_size)]
        for i, tensor in enumerate(output_tensors):
            object_list[i] = self._tensor_to_object(tensor, object_size_list[i].item())
        return object_list

    def _object_to_tensor(self, obj, device):
        f = io.BytesIO()
        self._pickler(f).dump(obj)
        byte_tensor = torch.ByteTensor(list(f.getvalue())).to(device)
        local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
        return byte_tensor, local_size
    
    def _tensor_to_object(self, tensor, tensor_size):
        # cast the object tensor into uint8 type and cpu device
        # cast the object uint8 list into bytes
        byte_data = bytes(tensor.type(torch.uint8).cpu().tolist()[:tensor_size])
        return self._unpickler(io.BytesIO(byte_data)).load()
    
    def _make_tensor_group_list(
        self,
        size: Union[list, tuple, int],
        group_size: int,
        dtype,
        device
    ):
        assert isinstance(size, (list, tuple, int)), 'size must be list, tuple or int, but not {}'.format(type(size).__qualname__)
        tensor_size = (group_size,) + (
            tuple(size) if isinstance(size, (list, tuple)) else (size,)
        )
        tensor_placeholder = torch.zeros(tensor_size, dtype=dtype, device=device)
        return [
            tensor_placeholder[i, :] for i in range(group_size)
        ]
    
    def _get_device(self, group=None):
        import torch.distributed as dist
        backend_dict = {
            'nccl': torch.device('cuda', torch.cuda.current_device()),
            'mpi': torch.device('cpu'),
            'gloo': torch.device('cpu')
        }
        backend = dist.get_backend(group=group)
        return backend_dict.get(backend, torch.device('cpu'))


from torchslime.utils.tstype import T_M_SEQ, T_M


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
    @SmartWraps(func)
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
            from torchslime.log import logger
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
        'K': 1e3,
        'M': 1e6,
        'B': 1e9
    }
    divisor = format_dict.get(format, 1)

    num = 0
    for param in model.parameters():
        num += param.numel()
    result = num / divisor
    return result if format is None else ('{0:.' + str(decimal) + 'f}{1}').format(result, format)


def is_torch_distributed_ready():
    """
    Check whether the torch distributed settings are ready.
    """
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


class GenericID:

    def __init__(self, attrs: Union[List, Tuple]) -> None:
        self.attrs = list(attrs)

    def __call__(self, *args, **kwargs):
        return GIDValue(self.attrs, *args, **kwargs)


class GIDValue:
    
    def __init__(self, attrs: Union[List, Tuple], *args, **kwargs) -> None:
        self.attrs = attrs

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

# TODO: refactor the util package
from typing import Dict, Union, Tuple, Sequence, MutableSequence, Generic, TypeVar, \
    overload, Iterator, Iterable, Any, List
from torch import Tensor
import torch
from torch.nn import Module
from time import time
import inspect
import pickle
import io
import os
from types import MethodType, FunctionType
import re


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


def is_function_or_method(__item: Any) -> bool:
    return isinstance(__item, (MethodType, FunctionType))


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
            'nccl': torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else NOTHING,
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


class ContextDecorator:

    pass


class StrTemplate:

    pass


MAGIC_PATTERN = re.compile('^_{2}[^_](?:.*[^_])?_{2}$')

def is_magic_naming(__name: str) -> bool:
    return re.match(MAGIC_PATTERN, str(__name)) is not None


SLIME_PATTERN = re.compile('^[^_](?:.*[^_])?_{2}$')

def is_slime_naming(__name: str) -> bool:
    return re.match(SLIME_PATTERN, str(__name)) is not None


def xor__(__x, __y) -> bool:
    return bool((__x and not __y) or (not __x and __y))


from torchslime.utils.bases import NOTHING

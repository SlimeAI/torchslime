import io
import pickle
from .registry import Registry
from .typing import (
    NOTHING,
    List,
    Union,
    Callable,
    TypeVar,
    Sequence,
    NoneOrNothing,
    Pass,
    PASS,
    is_none_or_nothing,
    Type,
    Iterable,
    Missing,
    MISSING
)
from .base import BaseList, AttrObserver, AttrObserve, AttrObservable
from slime_core.utils.launch import (
    CoreLaunchUtil,
    CoreDistComm
)
import torch.distributed as dist
from torch import Tensor
import torch

_T = TypeVar("_T")
launch_util_registry = Registry[Type["LaunchUtil"]]('launch_util_registry')


class LaunchUtil(CoreLaunchUtil):
    
    def __init__(self) -> None:
        self.dist_comm: Union[DistComm, NoneOrNothing] = NOTHING

    def call(self, __caller: Callable[[], _T], *, exec_ranks: Union[Sequence[int], NoneOrNothing, Pass] = PASS) -> Union[_T, None]: pass
    def is_exec(self, exec_ranks: Union[Sequence[int], NoneOrNothing, Pass] = PASS) -> bool: pass
    def is_distributed(self) -> bool: pass
    def is_distributed_ready(self) -> bool: pass
    def get_rank(self, group=None) -> Union[int, NoneOrNothing]: pass
    def get_world_size(self, group=None) -> Union[int, NoneOrNothing]: pass


def is_torch_distributed_ready():
    """
    Check whether the torch distributed settings are ready.
    """
    return dist.is_available() and dist.is_initialized()


@launch_util_registry(name='vanilla')
class VanillaLaunchUtil(LaunchUtil):
    
    def call(
        self,
        __caller: Callable[[], _T],
        *,
        exec_ranks: Union[Sequence[int], NoneOrNothing, Pass] = PASS
    ) -> _T:
        return __caller()
    
    def is_exec(self, exec_ranks: Union[Sequence[int], NoneOrNothing, Pass] = PASS) -> bool:
        return True
    
    def is_distributed(self) -> bool:
        return False
    
    def is_distributed_ready(self) -> bool:
        ready = is_torch_distributed_ready()
        if ready is True:
            from torchslime.logging.logger import logger
            logger.warning('Trying to run torch distributed in the vanilla launch, where TorchSlime will not have the distributed behavior.')
        return ready
    
    def get_rank(self, group=None):
        return NOTHING
    
    def get_world_size(self, group=None):
        return NOTHING


@launch_util_registry(name='distributed')
class DistributedLaunchUtil(LaunchUtil):
    
    def __init__(self) -> None:
        super().__init__()
        self.dist_comm = TorchComm()
    
    def call(
        self,
        __caller: Callable[[], _T],
        *,
        exec_ranks: Union[Sequence[int], NoneOrNothing, Pass] = PASS
    ) -> Union[_T, None]:
        if self.is_exec(exec_ranks):
            return __caller()

    def is_exec(self, exec_ranks: Union[Sequence[int], NoneOrNothing, Pass] = PASS) -> bool:
        # PASS: always exec
        # None or Nothing: never exec
        # Others: ``rank`` in ``exec_ranks`` <=> exec
        return (exec_ranks is PASS) or (not is_none_or_nothing(exec_ranks) and self.get_rank() in exec_ranks)

    def is_distributed(self) -> bool:
        return True

    def is_distributed_ready(self) -> bool:
        ready = is_torch_distributed_ready()
        return ready

    def get_rank(self, group=None):
        return dist.get_rank(group=group)
    
    def get_world_size(self, group=None):
        return dist.get_world_size(group=group)


class Launcher(AttrObserver):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING
    ) -> None:
        AttrObserver.__init__(self)
        self.bind_launch_to_builtin_store__ = launch is MISSING
        if launch is MISSING:
            # bind launch to builtin store
            from .store import store
            store.builtin__().attach__(self, namespaces=['builtin_store_launch__'])
        else:
            self.set_launch__(launch)
        
        if exec_ranks is MISSING:
            exec_ranks = [0]
        self.set_exec_ranks__(exec_ranks)
    
    def set_launch__(self, launch: Union[str, LaunchUtil]):
        if isinstance(launch, str):
            launch = launch_util_registry.get(launch)()
        self.launch__ = launch
    
    @AttrObserve(namespace='builtin_store_launch__')
    def launch_observe__(self, new_value, old_value, observable: AttrObservable):
        self.set_launch__(new_value)
    
    def set_exec_ranks__(self, exec_ranks: Union[Iterable[int], NoneOrNothing, Pass]):
        self.exec_ranks__ = BaseList.create__(exec_ranks)
    
    def is_exec__(self):
        return self.launch__.is_exec(self.exec_ranks__)


class DistComm(CoreDistComm):

    def gather(self, tensor: Tensor, dst=0, group=None, async_op=False) -> None: pass
    def gather_object(self, obj, dst=0, group=None) -> None: pass
    def all_gather(self, tensor: Tensor, group=None, async_op=False) -> None: pass
    def all_gather_object(self, obj, group=None) -> None: pass
    def broadcast(self, tensor: Tensor, src=0, group=None, async_op=False) -> None: pass
    def broadcast_object(self, obj, src=0, group=None) -> None: pass
    def scatter(self, tensor: Tensor, scatter_list=None, src=0, group=None, async_op=False) -> None: pass
    def scatter_object(self, objs, src=0, group=None) -> None: pass


class TorchComm(DistComm):

    def __init__(self) -> None:
        self._pickler = pickle.Pickler
        self._unpickler = pickle.Unpickler

    def gather(self, tensor: Tensor, dst=0, group=None, async_op=False):
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

    def broadcast(self, tensor: Tensor, src=0, group=None, async_op=False):
        # this API is simple enough that does not need more adaptation
        return dist.broadcast(tensor, src, group=group, async_op=async_op)

    def broadcast_object(self, obj, src=0, group=None):
        # code modified from torch.distributed.broadcast_object_list in PyTorch 1.13
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
        return dist.scatter(tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)

    def scatter_object(self, objs, src=0, group=None):
        # code modified from torch.distributed.scatter_object_list in PyTorch 1.13
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

    def _all_gather_size(self, size_tensor: Tensor, group_size: int, device, group):
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

    def _tensor_to_object(self, tensor: Tensor, tensor_size):
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
        assert isinstance(size, (list, tuple, int)), f'size must be list, tuple or int, but not {type(size).__qualname__}'
        tensor_size = (group_size,) + (
            tuple(size) if isinstance(size, (list, tuple)) else (size,)
        )
        tensor_placeholder = torch.zeros(tensor_size, dtype=dtype, device=device)
        return [
            tensor_placeholder[i, :] for i in range(group_size)
        ]

    def _get_device(self, group=None):
        backend_dict = {
            'nccl': torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else NOTHING,
            'mpi': torch.device('cpu'),
            'gloo': torch.device('cpu')
        }
        backend = dist.get_backend(group=group)
        return backend_dict.get(backend, torch.device('cpu'))

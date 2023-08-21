"""
Distributed Launch Hook
"""
import torch
from torch import Tensor
import io
import pickle
from torchslime.utils.typing import List, Union
from torchslime.core.context import BaseContext
from torchslime.core.handlers import Handler
from torchslime.core.hooks.build import _BuildInterface
from torchslime.utils import is_torch_distributed_ready
from torchslime.log import logger
from torchslime.utils.bases import NOTHING, is_none_or_nothing, is_pass
from torchslime.components.registry import Registry

launch_registry = Registry('launch_registry')


class LaunchHook(_BuildInterface):

    def __init__(self) -> None:
        super().__init__()
        self.dist_comm: DistComm = NOTHING

    def handler_call(self, handler: Handler, ctx: BaseContext): pass
    def is_distributed(self) -> bool: pass
    def is_distributed_ready(self) -> bool: pass
    def get_rank(self, group=None): pass
    def get_world_size(self, group=None): pass
    def get_device_info(self, ctx: BaseContext): pass


@launch_registry.register(name='vanilla')
class VanillaLaunch(LaunchHook):
    
    def handler_call(self, handler: Handler, ctx: BaseContext):
        handler.handle(ctx)
    
    def is_distributed(self) -> bool:
        return False
    
    def is_distributed_ready(self) -> bool:
        ready = is_torch_distributed_ready()
        if ready is True:
            logger.warn('Trying to run torch distributed in the vanilla launch, where TorchSlime will not have the distributed behavior.')
        return ready
    
    def get_rank(self, group=None):
        return NOTHING
    
    def get_world_size(self, group=None):
        return NOTHING
    
    def get_device_info(self, ctx: BaseContext):
        return super().get_device_info(ctx)


@launch_registry.register(name='distributed')
class DistributedLaunch(LaunchHook):
    
    def __init__(self) -> None:
        super().__init__()
        self.dist_comm = TorchComm()
    
    def handler_call(self, handler: Handler, ctx: BaseContext):
        exec_ranks = handler.get_exec_ranks()
        # always exec
        if is_pass(exec_ranks):
            handler.handle(ctx)
            return
        # never exec
        if is_none_or_nothing(exec_ranks):
            return
        # exec in the specific ranks
        rank = self.get_rank()
        if rank in exec_ranks:
            handler.handle(ctx)

    def is_distributed(self) -> bool:
        return True

    def is_distributed_ready(self) -> bool:
        ready = is_torch_distributed_ready()
        return ready

    def get_rank(self, group=None):
        import torch.distributed as dist
        return dist.get_rank(group=group)
    
    def get_world_size(self, group=None):
        import torch.distributed as dist
        return dist.get_world_size(group=group)

    def after_build_train(self, ctx: BaseContext) -> None:
        handler = ctx.handler_ctx
        average_handlers = ctx.run_ctx.train.get_by_class(handler.Meter)
        for a_handler in average_handlers:
            state = a_handler.get_id().split('_')[-1]
            a_handler.insert_before_self(handler.GatherAverage(_id='gather_average_{state}'.format(state=state)))

    def after_build_eval(self, ctx: BaseContext) -> None:
        handler = ctx.handler_ctx
        average_handlers = ctx.run_ctx.eval.get_by_class(handler.Meter)
        for a_handler in average_handlers:
            state = a_handler.get_id().split('_')[0]
            a_handler.insert_before_self(handler.GatherAverage(_id='gather_average_{state}'.format(state=state)))
    
    def get_device_info(self, ctx: BaseContext):
        return super().get_device_info(ctx)


class DistComm:
    
    def gather(self, tensor: Tensor, dst=0, group=None, async_op=False) -> None: pass
    def gather_object(self, obj, dst=0, group=None) -> None: pass
    def all_gather(self, tensor: Tensor, group=None, async_op=False) -> None: pass
    def all_gather_object(self, obj, group=None) -> None: pass
    def broadcast(self, tensor, src=0, group=None, async_op=False) -> None: pass
    def broadcast_object(self, obj, src=0, group=None) -> None: pass
    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False) -> None: pass
    def scatter_object(self, objs, src=0, group=None) -> None: pass


class TorchComm(DistComm):

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

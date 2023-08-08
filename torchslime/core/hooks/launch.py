"""
Distributed Launch Hook
"""
from torchslime.core.context import BaseContext
from torchslime.core.handlers import Handler
from torchslime.core.hooks.build import _BuildInterface
from torchslime.utils import is_torch_distributed_ready
from torchslime.log import logger
from torchslime.utils.bases import NOTHING, is_none_or_nothing
from torchslime.components.registry import Registry

launch_registry = Registry('launch_registry')


class LaunchHook(_BuildInterface):

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
    
    def handler_call(self, handler: Handler, ctx: BaseContext):
        # always exec
        if exec_ranks is ...:
            handler.handle(ctx)
            return
        # never exec
        if is_none_or_nothing(exec_ranks):
            return
        # exec in the specific ranks
        rank = ctx.distributed_ctx.get_rank()
        exec_ranks = handler.get_exec_ranks()
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
        metric_handlers = ctx.run_ctx.train.get_by_class(handler.Metrics)
        for m_handler in metric_handlers:
            m_handler.insert_after_self(handler.GatherAverage(_id=''))

    def after_build_eval(self, ctx: BaseContext) -> None:
        handler = ctx.handler_ctx
        metric_handlers = ctx.run_ctx.train.get_by_class(handler.Metrics)
        for m_handler in metric_handlers:
            m_handler.insert_after_self(handler.GatherAverage(_id=''))
    
    def get_device_info(self, ctx: BaseContext):
        return super().get_device_info(ctx)

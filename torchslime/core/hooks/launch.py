"""
Distributed Launch Hook
"""
from torchslime.core import Context
from torchslime.core.handler import Handler
from torchslime.utils import is_none_or_nothing, NOTHING, is_torch_distributed_ready
from torchslime.log import logger


class LaunchHook:

    def handler_call(self, handler: Handler, ctx: Context): pass
    def is_distributed(self) -> bool: pass
    def is_distributed_ready(self) -> bool: pass
    def get_rank(self, group=None): pass
    def get_world_size(self, group=None): pass
    def after_build_train(self, ctx: Context): pass
    def after_build_predict(self, ctx: Context): pass
    def after_build_eval(self, ctx: Context): pass
    def get_device(self, ctx: Context): pass


class VanillaLaunch(LaunchHook):
    
    def handler_call(self, handler: Handler, ctx: Context):
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
    
    def get_device(self, ctx: Context):
        return super().get_device(ctx)


class DistributedLaunch(LaunchHook):
    
    def handler_call(self, handler: Handler, ctx: Context):
        rank = ctx.distributed.get_rank()
        exec_ranks = handler.get_exec_ranks()
        if is_none_or_nothing(exec_ranks) is False and \
            (exec_ranks is ... or rank in exec_ranks):
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

    def after_build_train(self, ctx: Context):
        handler = ctx.handler
        metric_handlers = ctx.run.train.get_by_class(handler.Metrics)
        for m_handler in metric_handlers:
            m_handler.insert_after_self(handler.GatherAverage(_id=''))

    def after_build_eval(self, ctx: Context):
        handler = ctx.handler
        metric_handlers = ctx.run.train.get_by_class(handler.Metrics)
        for m_handler in metric_handlers:
            m_handler.insert_after_self(handler.GatherAverage(_id=''))
    
    def get_device(self, ctx: Context):
        return super().get_device(ctx)

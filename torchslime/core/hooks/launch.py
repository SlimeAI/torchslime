"""
Distributed Launch Hook
"""
from torchslime.utils.launch import LaunchUtil, VanillaLaunchUtil, DistributedLaunchUtil
from torchslime.core.context import BaseContext
from .build import _BuildInterface
from torchslime.components.registry import Registry

launch_registry = Registry('launch_registry')


class LaunchHook(LaunchUtil, _BuildInterface):

    def get_device_info(self, ctx: BaseContext): pass


@launch_registry(name='vanilla')
class VanillaLaunch(LaunchHook, VanillaLaunchUtil):
    
    def get_device_info(self, ctx: BaseContext):
        return super().get_device_info(ctx)


@launch_registry(name='distributed')
class DistributedLaunch(LaunchHook, DistributedLaunchUtil):

    def after_build_train(self, ctx: BaseContext) -> None:
        handler = ctx.handler_ctx
        average_handlers = ctx.run_ctx.train.get_by_class(handler.MeterHandler)
        for a_handler in average_handlers:
            state = a_handler.get_id().split('_')[-1]
            a_handler.insert_before_self(handler.GatherAverageHandler.m__(id=f'gather_average_{state}')())

    def after_build_eval(self, ctx: BaseContext) -> None:
        handler = ctx.handler_ctx
        average_handlers = ctx.run_ctx.eval.get_by_class(handler.MeterHandler)
        for a_handler in average_handlers:
            state = a_handler.get_id().split('_')[-1]
            a_handler.insert_before_self(handler.GatherAverageHandler.m__(id=f'gather_average_{state}')())
    
    def get_device_info(self, ctx: BaseContext):
        return super().get_device_info(ctx)

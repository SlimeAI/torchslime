"""
State Pattern for model state management.
"""
from torchslime.utils.bases import is_nothing, NOTHING
from torchslime.components.registry import Registry
from torchslime.components.metric import MeterDict
from torchslime.core.context.base import BaseContext
from typing import Tuple, Mapping

from torch.utils.data import DataLoader

state_registry = Registry('state_registry')


class StateHook:

    def __init__(self) -> None: pass
    def set_model_mode(self, ctx: BaseContext): pass
    def get_loader(self, ctx: BaseContext) -> DataLoader: pass
    # meter operations
    def init_meter(self, ctx: BaseContext) -> None: pass
    def update_meter(self, ctx: BaseContext, loss_value: Mapping, metrics: Mapping) -> None: pass
    def get_meter(self, ctx: BaseContext) -> Tuple[MeterDict, MeterDict]: pass

    def __str__(self) -> str:
        return 'BASE STATUS'


@state_registry.register(name='train')
class TrainState(StateHook):

    def __init__(self) -> None:
        super().__init__()

    def set_model_mode(self, ctx: BaseContext):
        ctx.model.train()

    def get_loader(self, ctx: BaseContext) -> DataLoader:
        ctx.ctx_check('run_ctx.train_provider', silent=False)
        ctx.run_ctx.train_loader = ctx.run_ctx.train_provider(ctx)
        return ctx.run_ctx.train_loader

    def init_meter(self, ctx: BaseContext) -> None:
        ctx.iteration_ctx.train_loss_values.clear()
        ctx.iteration_ctx.train_metrics.clear()
    
    def update_meter(self, ctx: BaseContext, loss_value: Mapping, metrics: Mapping) -> None:
        ctx.iteration_ctx.train_loss_values(loss_value)
        ctx.iteration_ctx.train_metrics(metrics)
    
    def get_meter(self, ctx: BaseContext) -> Tuple[MeterDict, MeterDict]:
        return ctx.iteration_ctx.train_loss_values, ctx.iteration_ctx.train_metrics

    def __str__(self) -> str:
        return 'TRAIN'


@state_registry.register(name='eval')
class EvalState(StateHook):

    def __init__(self) -> None:
        super().__init__()
    
    def set_model_mode(self, ctx: BaseContext):
        ctx.model.eval()

    def get_loader(self, ctx: BaseContext) -> DataLoader:
        ctx.ctx_check('run_ctx.eval_provider', silent=False)
        ctx.run_ctx.eval_loader = ctx.run_ctx.eval_provider(ctx)
        return ctx.run_ctx.eval_loader

    def init_meter(self, ctx: BaseContext) -> None:
        ctx.iteration_ctx.eval_loss_values.clear()
        ctx.iteration_ctx.eval_metrics.clear()
    
    def update_meter(self, ctx: BaseContext, loss_value: Mapping, metrics: Mapping) -> None:
        ctx.iteration_ctx.eval_loss_values(loss_value)
        ctx.iteration_ctx.eval_metrics(metrics)
    
    def get_meter(self, ctx: BaseContext) -> Tuple[MeterDict, MeterDict]:
        return ctx.iteration_ctx.eval_loss_values, ctx.iteration_ctx.eval_metrics

    def __str__(self) -> str:
        return 'EVAL'


@state_registry.register(name='val')
class ValState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def update_meter(self, ctx: BaseContext, loss_value: Mapping, metrics: Mapping) -> None:
        loss_value = {'val_{}'.format(str(k)):v for k, v in loss_value.items()}
        metrics = {'val_{}'.format(str(k)):v for k, v in metrics.items()}
        super().update_meter(ctx, loss_value, metrics)

    def __str__(self) -> str:
        return 'VAL'


@state_registry.register(name='predict')
class PredictState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'PREDICT'

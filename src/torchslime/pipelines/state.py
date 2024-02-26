"""
State Pattern for model state management.
"""
from torchslime.utils.registry import Registry
from torchslime.pipelines.metric import MeterDict
from torchslime.utils.typing import (
    Tuple,
    Mapping,
    Type,
    TYPE_CHECKING
)
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from torchslime.context import Context

state_registry = Registry[Type["ModelState"]]('state_registry')


class ModelState:

    def __init__(self) -> None: pass
    def set_model_mode(self, ctx: "Context"): pass
    def get_loader(self, ctx: "Context") -> DataLoader: pass
    # meter operations
    def init_meter(self, ctx: "Context") -> None: pass
    def update_meter(self, ctx: "Context", loss_value: Mapping, metrics: Mapping) -> None: pass
    def get_meter(self, ctx: "Context") -> Tuple[MeterDict, MeterDict]: pass

    def __str__(self) -> str:
        return 'BASE STATE'


@state_registry(name='train')
class TrainState(ModelState):

    def __init__(self) -> None:
        super().__init__()

    def set_model_mode(self, ctx: "Context"):
        ctx.model.train()

    def get_loader(self, ctx: "Context") -> DataLoader:
        ctx.ctx_check('pipeline_ctx.train_provider', silent=False)
        ctx.pipeline_ctx.train_loader = ctx.pipeline_ctx.train_provider(ctx)
        return ctx.pipeline_ctx.train_loader

    def init_meter(self, ctx: "Context") -> None:
        ctx.iteration_ctx.train_loss_values.clear()
        ctx.iteration_ctx.train_metrics.clear()
    
    def update_meter(self, ctx: "Context", loss_value: Mapping, metrics: Mapping) -> None:
        ctx.iteration_ctx.train_loss_values(loss_value)
        ctx.iteration_ctx.train_metrics(metrics)
    
    def get_meter(self, ctx: "Context") -> Tuple[MeterDict, MeterDict]:
        return ctx.iteration_ctx.train_loss_values, ctx.iteration_ctx.train_metrics

    def __str__(self) -> str:
        return 'TRAIN'


@state_registry(name='eval')
class EvalState(ModelState):

    def __init__(self) -> None:
        super().__init__()
    
    def set_model_mode(self, ctx: "Context"):
        ctx.model.eval()

    def get_loader(self, ctx: "Context") -> DataLoader:
        ctx.ctx_check('pipeline_ctx.eval_provider', silent=False)
        ctx.pipeline_ctx.eval_loader = ctx.pipeline_ctx.eval_provider(ctx)
        return ctx.pipeline_ctx.eval_loader

    def init_meter(self, ctx: "Context") -> None:
        ctx.iteration_ctx.eval_loss_values.clear()
        ctx.iteration_ctx.eval_metrics.clear()
    
    def update_meter(self, ctx: "Context", loss_value: Mapping, metrics: Mapping) -> None:
        ctx.iteration_ctx.eval_loss_values(loss_value)
        ctx.iteration_ctx.eval_metrics(metrics)
    
    def get_meter(self, ctx: "Context") -> Tuple[MeterDict, MeterDict]:
        return ctx.iteration_ctx.eval_loss_values, ctx.iteration_ctx.eval_metrics

    def __str__(self) -> str:
        return 'EVAL'


@state_registry(name='val')
class ValState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def update_meter(self, ctx: "Context", loss_value: Mapping, metrics: Mapping) -> None:
        loss_value = {f'val_{str(k)}':v for k, v in loss_value.items()}
        metrics = {f'val_{str(k)}':v for k, v in metrics.items()}
        super().update_meter(ctx, loss_value, metrics)

    def __str__(self) -> str:
        return 'VAL'


@state_registry(name='predict')
class PredictState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'PREDICT'

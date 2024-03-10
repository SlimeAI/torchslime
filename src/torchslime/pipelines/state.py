"""
State Pattern for model state management.
"""
from torchslime.utils.registry import Registry
from torchslime.pipelines.metric import MeterDict
from torchslime.utils.decorator import FuncSetAttr
from torchslime.utils.typing import (
    Tuple,
    Mapping,
    Type,
    Dict,
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
    
    @FuncSetAttr(attr_dict=dict(
        profile_keys__=()
    ))
    def format_logging_profile(self, ctx: "Context", profile_dict: Dict[str, str]) -> str:
        """
        Method that formats logging profile. ``profile_keys__`` can be set to specify 
        the needed profile values in order to improve efficiency.
        """
        pass
    
    def is_grad_enabled(self) -> bool:
        return False

    def __str__(self) -> str:
        return 'BASE STATE'


@state_registry(key='train')
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

    @FuncSetAttr(attr_dict=dict(
        profile_keys__=('logging_point', 'meter')
    ))
    def format_logging_profile(self, ctx: "Context", profile_dict: Dict[str, str]) -> str:
        return f'{profile_dict["logging_point"]} | {profile_dict["meter"]}'

    def is_grad_enabled(self) -> bool:
        return True

    def __str__(self) -> str:
        return 'TRAIN'


@state_registry(key='eval')
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

    @FuncSetAttr(attr_dict=dict(
        profile_keys__=('meter',)
    ))
    def format_logging_profile(self, ctx: "Context", profile_dict: Dict[str, str]) -> str:
        # In eval state, logging point should be ignored.
        return f'{profile_dict["meter"]}'

    def is_grad_enabled(self) -> bool:
        return False

    def __str__(self) -> str:
        return 'EVAL'


@state_registry(key='val')
class ValState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def update_meter(self, ctx: "Context", loss_value: Mapping, metrics: Mapping) -> None:
        loss_value = {f'val_{str(k)}':v for k, v in loss_value.items()}
        metrics = {f'val_{str(k)}':v for k, v in metrics.items()}
        super().update_meter(ctx, loss_value, metrics)

    @FuncSetAttr(attr_dict=dict(
        profile_keys__=('logging_point', 'meter')
    ))
    def format_logging_profile(self, ctx: "Context", profile_dict: Dict[str, str]) -> str:
        return f'{profile_dict["logging_point"]} | {profile_dict["meter"]}'

    def __str__(self) -> str:
        return 'VAL'


@state_registry(key='predict')
class PredictState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    # NOTE: Predict state is not involved in meter computing.
    def init_meter(self, ctx: "Context") -> None: pass
    def update_meter(self, ctx: "Context", loss_value: Mapping, metrics: Mapping) -> None: pass

    def get_meter(self, ctx: "Context") -> Tuple[MeterDict, MeterDict]:
        """
        Should always return empty ``MeterDict``, because predict state doesn't 
        compute metrics or loss.
        """
        return MeterDict(), MeterDict()

    @FuncSetAttr(attr_dict=dict(
        profile_keys__=()
    ))
    def format_logging_profile(self, ctx: "Context", profile_dict: Dict[str, str]) -> str:
        # In predict state, there is nothing to be logged.
        return f'[{str(self)} Profile] (Empty)'

    def __str__(self) -> str:
        return 'PREDICT'

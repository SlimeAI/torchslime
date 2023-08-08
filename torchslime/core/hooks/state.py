"""
State Pattern for model state management.
"""
from torchslime.utils.bases import is_nothing, NOTHING
from torchslime.components.registry import Registry
from torchslime.core.context.base import BaseContext
from typing import Tuple

from torch.utils.data import DataLoader

state_registry = Registry('state_registry')


class StateHook:

    def __init__(self) -> None: pass
    def set_model_mode(self, ctx: BaseContext): pass
    def get_loader(self, ctx: BaseContext) -> DataLoader: pass
    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]: pass
    def set_avg_loss_value_and_metrics(self, ctx: BaseContext, loss_value, metrics): pass
    def get_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY): pass

    def init_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        if is_nothing(ctx.inner_ctx[INNER_KEY]):
            ctx.inner_ctx[INNER_KEY] = {}

    def clear_avg_info(self, ctx: BaseContext, INNER_KEY):
        if is_nothing(ctx.inner_ctx[INNER_KEY]):
            ctx.inner_ctx[INNER_KEY] = {}

    def _get_avg_inner_init_item(self, ctx: BaseContext):
        return {
            'loss_value': ctx.run_ctx.loss_wrapper.get_empty(),
            'loss_value_count': {},
            'metrics': {},
            'metrics_count': {}
        }

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
        return ctx.run_ctx.train_provider(ctx)

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        loss_value = ctx.run_ctx.loss_wrapper.get_copy(ctx.iteration_ctx.train_loss_value)
        metrics = ctx.iteration_ctx.train_metrics
        return loss_value, metrics
    
    def init_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        super().init_avg_inner_ctx(ctx, INNER_KEY)
        if is_nothing(ctx.inner_ctx[INNER_KEY].get('train', NOTHING)):
            ctx.inner_ctx[INNER_KEY]['train'] = self._get_avg_inner_init_item(ctx)
    
    def set_avg_loss_value_and_metrics(self, ctx: BaseContext, loss_value, metrics):
        ctx.iteration_ctx.train_loss_value = loss_value
        ctx.iteration_ctx.train_metrics = metrics

    def get_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        return ctx.inner_ctx[INNER_KEY].get('train', NOTHING)

    def clear_avg_info(self, ctx: BaseContext, INNER_KEY):
        super().clear_avg_info(ctx, INNER_KEY)
        ctx.inner_ctx[INNER_KEY]['train'] = self._get_avg_inner_init_item(ctx)
        ctx.iteration_ctx.train_metrics = NOTHING
        ctx.iteration_ctx.train_loss_value = NOTHING

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
        return ctx.run_ctx.eval_provider(ctx)

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        loss_value = ctx.run_ctx.loss_wrapper.get_copy(ctx.iteration_ctx.eval_loss_value)
        metrics = ctx.iteration_ctx.eval_metrics
        return loss_value, metrics

    def init_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        super().init_avg_inner_ctx(ctx, INNER_KEY)
        if is_nothing(ctx.inner_ctx[INNER_KEY].get('eval', NOTHING)):
            ctx.inner_ctx[INNER_KEY]['eval'] = self._get_avg_inner_init_item(ctx)

    def set_avg_loss_value_and_metrics(self, ctx: BaseContext, loss_value, metrics):
        ctx.iteration_ctx.eval_loss_value = loss_value
        ctx.iteration_ctx.eval_metrics = metrics
    
    def get_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        return ctx.inner_ctx[INNER_KEY].get('eval', NOTHING)

    def clear_avg_info(self, ctx: BaseContext, INNER_KEY):
        super().clear_avg_info(ctx, INNER_KEY)
        ctx.inner_ctx[INNER_KEY]['eval'] = self._get_avg_inner_init_item(ctx)
        ctx.iteration_ctx.eval_metrics = NOTHING
        ctx.iteration_ctx.eval_loss_value = NOTHING

    def __str__(self) -> str:
        return 'EVAL'


@state_registry.register(name='val')
class ValState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        loss_value = ctx.run_ctx.loss_wrapper.get_copy(ctx.iteration_ctx.eval_loss_value)
        _loss_value = {}
        for key, value in loss_value.items():
            _loss_value['val_{}'.format(key)] = value
        loss_value.set_dict__(_loss_value)
        
        _metrics = ctx.iteration_ctx.eval_metrics
        metrics = {}
        for key, value in _metrics.items():
            metrics['val_{}'.format(key)] = value
        return loss_value, metrics

    def __str__(self) -> str:
        return 'VAL'


@state_registry.register(name='predict')
class PredictState(EvalState):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'PREDICT'

"""
Status Pattern for model status management.
"""
from torchslime.util import NOTHING, is_nothing
from torchslime.module import Registry
from torchslime.core.context import BaseContext
from typing import Tuple

context_status = Registry('context_status')


class Status:

    def __init__(self) -> None:
        pass

    def set_model_mode(self, ctx: BaseContext):
        pass

    def get_dataset(self, ctx: BaseContext):
        pass

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        pass

    def init_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        if is_nothing(ctx.inner[INNER_KEY]):
            ctx.inner[INNER_KEY] = {}

    def set_avg_loss_value_and_metrics(self, ctx: BaseContext, loss_value, metrics):
        pass

    def get_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        pass

    def clear_avg_info(self, ctx: BaseContext, INNER_KEY):
        if is_nothing(ctx.inner[INNER_KEY]):
            ctx.inner[INNER_KEY] = {}

    def _get_avg_inner_init_item(self, ctx: BaseContext):
        return {
            'loss_value': ctx.run.loss_wrapper.get_empty(),
            'loss_value_count': {},
            'metrics': {},
            'metrics_count': {}
        }

    def __str__(self) -> str:
        return 'BASE STATUS'


@context_status.register('train')
class TrainStatus(Status):

    def __init__(self) -> None:
        super().__init__()

    def set_model_mode(self, ctx: BaseContext):
        ctx.model.train()

    def get_dataset(self, ctx: BaseContext):
        ctx.ctx_check('run.train_provider', silent=False)
        ctx.dataset = ctx.run.train_provider(ctx)

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        loss_value = ctx.run.loss_wrapper.get_copy(ctx.epoch.train_loss_value)
        metrics = ctx.epoch.train_metrics
        return loss_value, metrics
    
    def init_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        super().init_avg_inner_ctx(ctx, INNER_KEY)
        if is_nothing(ctx.inner[INNER_KEY].get('train', NOTHING)):
            ctx.inner[INNER_KEY]['train'] = self._get_avg_inner_init_item(ctx)
    
    def set_avg_loss_value_and_metrics(self, ctx: BaseContext, loss_value, metrics):
        ctx.epoch.train_loss_value = loss_value
        ctx.epoch.train_metrics = metrics

    def get_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        return ctx.inner[INNER_KEY].get('train', NOTHING)

    def clear_avg_info(self, ctx: BaseContext, INNER_KEY):
        super().clear_avg_info(ctx, INNER_KEY)
        ctx.inner[INNER_KEY]['train'] = self._get_avg_inner_init_item(ctx)
        ctx.epoch.train_metrics = NOTHING
        ctx.epoch.train_loss_value = NOTHING

    def __str__(self) -> str:
        return 'TRAIN'


@context_status.register('eval')
class EvalStatus(Status):

    def __init__(self) -> None:
        super().__init__()
    
    def set_model_mode(self, ctx: BaseContext):
        ctx.model.eval()

    def get_dataset(self, ctx: BaseContext):
        ctx.ctx_check('run.eval_provider', silent=False)
        ctx.dataset = ctx.run.eval_provider(ctx)

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        loss_value = ctx.run.loss_wrapper.get_copy(ctx.epoch.eval_loss_value)
        metrics = ctx.epoch.eval_metrics
        return loss_value, metrics

    def init_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        super().init_avg_inner_ctx(ctx, INNER_KEY)
        if is_nothing(ctx.inner[INNER_KEY].get('eval', NOTHING)):
            ctx.inner[INNER_KEY]['eval'] = self._get_avg_inner_init_item(ctx)

    def set_avg_loss_value_and_metrics(self, ctx: BaseContext, loss_value, metrics):
        ctx.epoch.eval_loss_value = loss_value
        ctx.epoch.eval_metrics = metrics
    
    def get_avg_inner_ctx(self, ctx: BaseContext, INNER_KEY):
        return ctx.inner[INNER_KEY].get('eval', NOTHING)

    def clear_avg_info(self, ctx: BaseContext, INNER_KEY):
        super().clear_avg_info(ctx, INNER_KEY)
        ctx.inner[INNER_KEY]['eval'] = self._get_avg_inner_init_item(ctx)
        ctx.epoch.eval_metrics = NOTHING
        ctx.epoch.eval_loss_value = NOTHING

    def __str__(self) -> str:
        return 'EVAL'


@context_status.register('val')
class ValStatus(EvalStatus):

    def __init__(self) -> None:
        super().__init__()

    def get_avg_loss_value_and_metrics(self, ctx: BaseContext) -> Tuple[dict, dict]:
        loss_value = ctx.run.loss_wrapper.get_copy(ctx.epoch.eval_loss_value)
        _loss_value = {}
        for key, value in loss_value.items():
            _loss_value['val_{}'.format(key)] = value
        loss_value.set_dict(_loss_value)
        
        _metrics = ctx.epoch.eval_metrics
        metrics = {}
        for key, value in _metrics.items():
            metrics['val_{}'.format(key)] = value
        return loss_value, metrics

    def __str__(self) -> str:
        return 'VAL'


@context_status.register('predict')
class PredictStatus(EvalStatus):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'PREDICT'

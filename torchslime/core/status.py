"""
Status Pattern for model status management.
"""
from ..util import NOTHING, is_nothing
from ..module import Registry
from .context import Context

proxy_status = Registry('proxy_status')


class Status:

    def __init__(self) -> None:
        pass

    def set_model_mode(self, ctx: Context):
        pass

    def get_dataset(self, ctx: Context):
        pass

    def get_avg_loss_and_metrics(self, ctx: Context):
        pass

    def init_avg_inner_ctx(self, ctx: Context, INNER_KEY):
        if is_nothing(ctx.inner[INNER_KEY]):
            ctx.inner[INNER_KEY] = {}

    def set_avg_loss_and_metrics(self, ctx: Context, loss, metrics):
        pass

    def get_avg_inner_ctx(self, ctx: Context, INNER_KEY):
        pass

    def clear_avg_info(self, ctx: Context, INNER_KEY):
        if is_nothing(ctx.inner[INNER_KEY]):
            ctx.inner[INNER_KEY] = {}

    @staticmethod
    def _get_avg_inner_init_item():
        return {
            'count': {},
            'loss': 0,
            'metrics': {}
        }

    def __str__(self) -> str:
        return 'BASE STATUS'


@proxy_status.register('train')
class TrainStatus(Status):

    def __init__(self) -> None:
        super().__init__()

    def set_model_mode(self, ctx: Context):
        ctx.model.train()

    def get_dataset(self, ctx: Context):
        ctx.ctx_check('run.train_provider', silent=False)
        ctx.dataset = ctx.run.train_provider(ctx)

    def get_avg_loss_and_metrics(self, ctx: Context) -> list:
        data = []
        if is_nothing(ctx.epoch.train_loss) is False:
            data.append('loss: {0:.5f}'.format(ctx.epoch.train_loss))
        for key, value in ctx.epoch.train_metrics.items():
            data.append('{0}: {1:.5f}'.format(key, value))
        return data
    
    def init_avg_inner_ctx(self, ctx: Context, INNER_KEY):
        super().init_avg_inner_ctx(ctx, INNER_KEY)
        if is_nothing(ctx.inner[INNER_KEY].get('train', NOTHING)):
            ctx.inner[INNER_KEY]['train'] = self._get_avg_inner_init_item()
    
    def set_avg_loss_and_metrics(self, ctx: Context, loss, metrics):
        ctx.epoch.train_loss = loss
        ctx.epoch.train_metrics = metrics

    def get_avg_inner_ctx(self, ctx: Context, INNER_KEY):
        return ctx.inner[INNER_KEY].get('train', NOTHING)

    def clear_avg_info(self, ctx: Context, INNER_KEY):
        super().clear_avg_info(ctx, INNER_KEY)
        ctx.inner[INNER_KEY]['train'] = self._get_avg_inner_init_item()
        ctx.epoch.train_metrics = NOTHING
        ctx.epoch.train_loss = NOTHING

    def __str__(self) -> str:
        return 'TRAIN'


@proxy_status.register('eval')
class EvalStatus(Status):

    def __init__(self) -> None:
        super().__init__()
    
    def set_model_mode(self, ctx: Context):
        ctx.model.eval()

    def get_dataset(self, ctx: Context):
        ctx.ctx_check('run.eval_provider', silent=False)
        ctx.dataset = ctx.run.eval_provider(ctx)

    def get_avg_loss_and_metrics(self, ctx: Context):
        data = []
        if is_nothing(ctx.epoch.eval_loss) is False:
            data.append('loss: {0:.5f}'.format(ctx.epoch.eval_loss))
        for key, value in ctx.epoch.eval_metrics.items():
            data.append('{0}: {1:.5f}'.format(key, value))
        return data

    def init_avg_inner_ctx(self, ctx: Context, INNER_KEY):
        super().init_avg_inner_ctx(ctx, INNER_KEY)
        if is_nothing(ctx.inner[INNER_KEY].get('eval', NOTHING)):
            ctx.inner[INNER_KEY]['eval'] = self._get_avg_inner_init_item()

    def set_avg_loss_and_metrics(self, ctx: Context, loss, metrics):
        ctx.epoch.eval_loss = loss
        ctx.epoch.eval_metrics = metrics
    
    def get_avg_inner_ctx(self, ctx: Context, INNER_KEY):
        return ctx.inner[INNER_KEY].get('eval', NOTHING)

    def clear_avg_info(self, ctx: Context, INNER_KEY):
        super().clear_avg_info(ctx, INNER_KEY)
        ctx.inner[INNER_KEY]['eval'] = self._get_avg_inner_init_item()
        ctx.epoch.eval_metrics = NOTHING
        ctx.epoch.eval_loss = NOTHING

    def __str__(self) -> str:
        return 'EVAL'


@proxy_status.register('val')
class ValStatus(EvalStatus):

    def __init__(self) -> None:
        super().__init__()

    def set_avg_loss_and_metrics(self, ctx: Context, loss, metrics):
        ctx.epoch.eval_loss = loss
        _metrics = {}
        for key, value in metrics.items():
            _metrics['val_{0}'.format(key)] = value
        ctx.epoch.eval_metrics = _metrics

    def get_avg_loss_and_metrics(self, ctx: Context):
        data = []
        if is_nothing(ctx.epoch.eval_loss) is False:
            data.append('val_loss: {0:.5f}'.format(ctx.epoch.eval_loss))
        for key, value in ctx.epoch.eval_metrics.items():
            data.append('{0}: {1:.5f}'.format(key, value))
        return data

    def __str__(self) -> str:
        return 'VAL'


@proxy_status.register('predict')
class PredictStatus(EvalStatus):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'PREDICT'

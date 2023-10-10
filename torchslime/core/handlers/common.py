from torchslime.utils.typing import (
    Dict,
    List,
    Callable,
    Iterable,
    Mapping,
    TypeVar,
    Tuple,
    is_none_or_nothing,
    NoneOrNothing,
    Any,
    Sequence,
    Union
)
from torchslime.utils import (
    type_cast,
    get_len
)
from torchslime.utils.bases import BaseList
from torchslime.components.metric import MeterDict
from torchslime.components.store import store
from torchslime.utils.decorators import CallDebug
from torchslime.utils.common import dict_to_key_value_str
from torchslime.core.context.base import BaseContext
from torchslime.core.handlers import Handler, HandlerContainer
from torchslime.core.hooks.state import state_registry, StateHook
from torchslime.logging.logger import logger
from torchslime.logging.rich import HandlerProgress, SlimeLiveLauncher, SlimeGroup, SlimeProgressLauncher
from contextlib import contextmanager
from torch import set_grad_enabled
from functools import wraps
from itertools import cycle

__all__ = [
    'TorchGrad',
    'EmptyHandler',
    'FuncHandler',
    'EpochIterationContainer',
    'IterationContainer',
    'StepIterationContainer',
    'ForwardHandler',
    'LossHandler',
    'BackwardHandler',
    'OptimizerContainer',
    'MetricHandler',
    'GatherAverageHandler',
    'MeterInitHandler',
    'MeterHandler',
    'LRDecayHandler'
]

_T = TypeVar('_T')


def TorchGrad(func):
    """
    Set grad enabled or not according to the context mode.
    """
    @wraps(func)
    def grad_switch(self, ctx: BaseContext):
        # only when context status is in ['TRAIN'] is the grad enabled
        with set_grad_enabled(str(ctx.hook_ctx.state) in ['TRAIN']):
            func(self, ctx)
    return grad_switch


class EmptyHandler(Handler):
    """Empty handler that does nothing when called.

    Args:
        Handler (torchslime.core.handler.Handler): _description_
    """
    
    @CallDebug(module_name='EmptyHandler')
    def handle(self, _: BaseContext):
        """do nothing"""
        pass


class FuncHandler(Handler, BaseList[Callable[[BaseContext], None]]):
    
    def __init__(self, __func_list: Iterable[Callable[[BaseContext], None]]):
        Handler.__init__(self)
        BaseList.__init__(self, __func_list)
    
    def handle(self, ctx: BaseContext):
        # execute lambda functions
        for func in self:
            func(ctx)


class ProgressInterface:
    
    def create_progress__(self, ctx: BaseContext) -> Tuple[Any, Any]: pass
    def progress_update__(self, ctx: BaseContext) -> None: pass
    
    def add_progress__(self, ctx: BaseContext) -> None:
        display_ctx = ctx.display_ctx
        display_ctx.live_group.append(display_ctx.handler_progress)
    
    def remove_progress__(self, ctx: BaseContext) -> None:
        ctx.display_ctx.handler_progress.remove_self__()
    
    @contextmanager
    def progress_context__(self, ctx: BaseContext):
        progress, task_id = self.create_progress__(ctx)
        with ctx.display_ctx.assign__(
            handler_progress=progress,
            progress_task_id=task_id
        ):
            self.add_progress__(ctx)
            yield
            self.remove_progress__(ctx)


class EpochIterationContainer(HandlerContainer, ProgressInterface):
    """
    Train Only
    """

    @CallDebug(module_name='EpochIterationContainer')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('iteration_ctx.total', silent=False)
        
        with self.progress_context__(ctx):
            # epoch iteration
            for current in range(
                ctx.iteration_ctx.start,
                ctx.iteration_ctx.total
            ):
                # set current epoch to the context
                with ctx.iteration_ctx.assign__(
                    current=current
                ):
                    # output epoch info.
                    logger.info(f'Epoch {ctx.iteration_ctx.current + 1} begins.')
                    super().handle(ctx)
                    # update progress
                    self.progress_update__(ctx)
    
    def create_progress__(self, ctx: BaseContext) -> Tuple[Any, Any]:
        progress = SlimeProgressLauncher.create__()
        task_id = progress.add_task('EpochIteration', total=ctx.iteration_ctx.total, completed=ctx.iteration_ctx.start)
        return progress, task_id
    
    def progress_update__(self, ctx: BaseContext) -> None:
        ctx.display_ctx.handler_progress.advance(
            task_id=ctx.display_ctx.progress_task_id,
            advance=1
        )
    
    def remove_progress__(self, ctx: BaseContext) -> None:
        super().remove_progress__(ctx)
        # detach observer
        store.builtin__().detach__(ctx.display_ctx.handler_progress)


class ProfileProgressInterface(ProgressInterface):
    
    def progress_update__(self, ctx: BaseContext) -> None:
        ctx.display_ctx.handler_progress.progress.advance(
            task_id=ctx.display_ctx.progress_task_id,
            advance=1
        )
        ctx.display_ctx.handler_progress.set_text__(
            f'{ctx.hook_ctx.profiler.meter_profile(ctx)}'
        )
    
    def remove_progress__(self, ctx: BaseContext) -> None:
        super().remove_progress__(ctx)
        # detach observer
        store.builtin__().detach__(ctx.display_ctx.handler_progress.progress)


class IterationContainer(HandlerContainer, ProfileProgressInterface):

    @CallDebug(module_name='IterationContainer')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        loader = ctx.hook_ctx.state.get_loader(ctx)
        # loader check
        if is_none_or_nothing(loader):
            logger.warning('Got empty data loader.')
            return
        
        total = get_len(loader)
        
        with ctx.step_ctx.assign__(
            total=total
        ), self.progress_context__(ctx):
            # data iteration
            for current, batch in enumerate(loader):
                with ctx.step_ctx.assign__(
                    batch=batch,  # original batch data of the dataset
                    current=current,  # the current step
                ):
                    # carry out the subsequent actions
                    super().handle(ctx)
                    self.progress_update__(ctx)
    
    def create_progress__(self, ctx: BaseContext) -> tuple[Any, Any]:
        total = ctx.step_ctx.total
        total=total if not is_none_or_nothing(total) else None
        
        handler_progress = HandlerProgress()
        task_id = handler_progress.progress.add_task(
            str(ctx.hook_ctx.state),
            total=total
        )
        return handler_progress, task_id


class StepIterationContainer(HandlerContainer, ProfileProgressInterface):
    """
    Train Only
    """
    
    @CallDebug(module_name='StepIterationContainer')
    @TorchGrad
    def handle(self, ctx: BaseContext):
        loader = ctx.hook_ctx.state.get_loader(ctx)
        # loader check
        if is_none_or_nothing(loader):
            logger.warning('Got empty data loader.')
            return
        
        total = ctx.iteration_ctx.total
        start = ctx.iteration_ctx.start
        
        with self.progress_context__(ctx):
            for current, batch in enumerate(cycle(loader), start=start):
                with ctx.iteration_ctx.assign__(
                    current=current  # current global step
                ), ctx.step_ctx.assign__(
                    batch=batch,  # original batch data of the dataset
                    current=current,  # the current step
                    total=total  # total steps
                ):
                    # carry out the subsequent actions
                    super().handle(ctx)
                    self.progress_update__(ctx)
                # break if finish
                if current + 1 >= total:
                    break
    
    def create_progress__(self, ctx: BaseContext) -> tuple[Any, Any]:
        handler_progress = HandlerProgress()
        task_id = handler_progress.progress.add_task(
            'StepIteration',
            total=ctx.iteration_ctx.total
        )
        return handler_progress, task_id


class ForwardHandler(Handler):

    @CallDebug(module_name='ForwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check([
            'model',
            'device',
            'run_ctx.data_parser',
            'step_ctx'
        ], silent=False)
        # forward
        x, y_true, extra = ctx.run_ctx.data_parser(ctx)
        y_pred = ctx.model(type_cast(x, ctx.device))
        y_true = type_cast(y_true, ctx.device)
        # clone and update context info
        ctx.step_ctx.from_kwargs__(
            # the result of the forward progress
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            extra=extra
        )


class LossHandler(Handler):
    
    @CallDebug(module_name='LossHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check('run_ctx.loss_func') is True:
            # compute loss
            loss = ctx.run_ctx.loss_func(ctx)
            ctx.step_ctx.loss = loss
            ctx.step_ctx.loss_values = self._parse_float(dict(loss))
    
    def _parse_float(self, loss_dict):
        for key in loss_dict:
            loss_dict[key] = float(loss_dict[key])
        return loss_dict


class BackwardHandler(Handler):

    @CallDebug(module_name='BackwardHandler')
    def handle(self, ctx: BaseContext):
        # context check
        if ctx.ctx_check(['step_ctx.loss']):
            last = ctx.step_ctx.total % ctx.run_ctx.grad_acc
            grad_acc = ctx.run_ctx.grad_acc if (ctx.step_ctx.total - ctx.step_ctx.current - 1) >= last else last
            # backward
            (ctx.run_ctx.loss_reduction(ctx) / grad_acc).backward()


class OptimizerContainer(HandlerContainer):
    
    @CallDebug(module_name='OptimizerContainer')
    def handle(self, ctx: BaseContext):
        # backward handler
        super().handle(ctx)
        if ctx.ctx_check(['run_ctx.optimizer']) and \
            ((ctx.step_ctx.current + 1) % ctx.run_ctx.grad_acc == 0 or ctx.step_ctx.current + 1 == ctx.step_ctx.total):
            ctx.run_ctx.optimizer.step()
            ctx.run_ctx.optimizer.zero_grad()


class MetricHandler(Handler):
    
    @CallDebug(module_name='MetricHandler')
    def handle(self, ctx: BaseContext):
        # context check
        ctx.ctx_check('step_ctx', silent=False)
        if ctx.ctx_check('run_ctx.metrics'):
            ctx.step_ctx.metrics = ctx.run_ctx.metrics(ctx)


class GatherAverageHandler(Handler):
    
    @CallDebug(module_name='GatherAverageHandler')
    def handle(self, ctx: BaseContext):
        dist_comm = ctx.hook_ctx.launch.dist_comm
        # gather data
        gathered_loss_values: List[Dict] = dist_comm.all_gather_object(ctx.step_ctx.loss_values)
        gathered_metrics: List[Dict] = dist_comm.all_gather_object(ctx.step_ctx.metrics)
        
        # Compute average loss values and metrics.
        ctx.step_ctx.loss_values = self._avg_dict(gathered_loss_values)
        ctx.step_ctx.metrics = self._avg_dict(gathered_metrics)
    
    def _avg_dict(self, dicts: List[Mapping]):
        meter_dict = MeterDict()
        for _dict in dicts:
            meter_dict(_dict)
        return meter_dict.get__('mean')


class MeterInitHandler(Handler):
    
    @CallDebug(module_name='MeterInitHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook_ctx.state.init_meter(ctx)


class MeterHandler(Handler):
    
    @CallDebug(module_name='MeterHandler')
    def handle(self, ctx: BaseContext):
        ctx.hook_ctx.state.update_meter(ctx, ctx.step_ctx.loss_values, ctx.step_ctx.metrics)


class LRDecayHandler(Handler):
    
    @CallDebug(module_name='LRDecayHandler')
    def handle(self, ctx: BaseContext):
        if ctx.ctx_check(['run_ctx.lr_decay']) is True:
            ctx.run_ctx.lr_decay.step()


class LoggingHandler(Handler):
    
    def __init__(
        self,
        logging_states: Sequence[Union[str, StateHook]]
    ) -> None:
        super().__init__()
        self.logging_states = logging_states
    
    def handle(self, ctx: BaseContext) -> None:
        # get logging point (Epoch/Step, current, total)
        profiler = ctx.hook_ctx.profiler
        logging_point = profiler.logging_point_profile(ctx)
        
        for state in self.logging_states:
            logger.info(
                f'{logging_point} | {profiler.meter_profile(ctx, state)}'
            )


class RootContainer(HandlerContainer):
    
    def handle(self, ctx: BaseContext) -> None:
        live_launcher, live_group = self.create_live__(ctx)
        
        with ctx.display_ctx.assign__(
            live_launcher=live_launcher,
            live_group=live_group
        ), ctx.display_ctx.live_launcher.get__():
            super().handle(ctx)
            # refresh live
            ctx.display_ctx.live_launcher.get__().refresh()
        
        store.builtin__().detach__(live_launcher)
    
    def create_live__(self, ctx: BaseContext) -> Tuple[Any, Any]:
        live_group = SlimeGroup()
        live_launcher = SlimeLiveLauncher(live_group, console=store.builtin__().console_launcher, transient=True)
        return live_launcher, live_group

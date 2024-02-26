from torchslime.utils.typing import (
    Any,
    Dict,
    Union,
    Callable,
    Iterable,
    is_none_or_nothing,
    Nothing,
    NOTHING,
    NoneOrNothing,
    TYPE_CHECKING,
    Missing,
    MISSING,
    TorchLRScheduler
)
from torchslime.utils.common import FuncArgs
from torchslime.logging.logger import logger, LoggerKwargs
from torchslime.pipelines.data import (
    ConstantProvider,
    DataParser,
    DataProvider,
    IndexParser
)
from torchslime.pipelines.metric import (
    MetricContainer,
    LossReductionFactory,
    Metric,
    LossFunc,
    LossFuncContainer
)
from torchslime.hooks.build import BuildHook, build_registry
from torchslime.hooks.launch import LaunchHook, launch_registry
from torchslime.pipelines.profiler import PipelineProfiler, profiler_registry
from torchslime.pipelines.state import ModelState, state_registry
from torchslime.utils.decorator import CallDebug, MethodChaining
from torch import Tensor
from torch.optim import Optimizer
# Type check only
if TYPE_CHECKING:
    from . import Context, AcceptableDataType

COMPILE_FUNC_SUFFIX = '_compile__'


class CompileFuncArgs(FuncArgs): pass


class Compile:
    
    def __init__(
        self,
        ctx: Union["Context", NoneOrNothing] = NOTHING
    ) -> None:
        self.ctx = ctx
    
    def __call__(self, **kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            # Do nothing.
            if value is MISSING:
                continue
            
            func_name = f'{key}{COMPILE_FUNC_SUFFIX}'
            func: Callable[..., None] = getattr(self, func_name, NOTHING)
            if func is NOTHING:
                logger.warning(
                    f'Compile func ``{func_name}`` not found. Compile ``{key}`` attribute failed.',
                    **LoggerKwargs(stacklevel=2)
                )
                continue
            
            if isinstance(value, CompileFuncArgs):
                func(*value.args, **value.kwargs)
            else:
                func(value)
    
    @CallDebug
    @MethodChaining
    def compile_pipeline_ctx(
        self,
        loss_func_list: Union[Iterable[LossFunc], NoneOrNothing, Missing] = MISSING,
        loss_reduction: Union[str, dict, Callable[["Context"], Tensor], NoneOrNothing, Missing] = MISSING,
        metrics: Union[Iterable[Metric], NoneOrNothing, Missing] = MISSING,
        optimizer: Union[Optimizer, NoneOrNothing, Missing] = MISSING,
        lr_scheduler: Union[TorchLRScheduler, NoneOrNothing, Missing] = MISSING,
        data_parser: Union[DataParser, NoneOrNothing, Missing] = MISSING
    ) -> "Compile":
        self(
            loss_func=loss_func_list,
            loss_reduction=loss_reduction,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            data_parser=data_parser
        )
    
    @CallDebug
    @MethodChaining
    def compile_hook_ctx(
        self,
        build_hook: Union[str, BuildHook, Missing] = MISSING,
        launch_hook: Union[str, LaunchHook, Missing] = MISSING
    ) -> "Compile":
        self(
            build_hook=build_hook,
            launch_hook=launch_hook
        )

    @CallDebug
    @MethodChaining
    def loss_func_compile__(self, loss_func_list: Union[Iterable[LossFunc], NoneOrNothing]) -> "Compile":
        if is_none_or_nothing(loss_func_list):
            loss_func = loss_func_list
        else:
            loss_func = LossFuncContainer(loss_func_list)
        self.ctx.pipeline_ctx.loss_func = loss_func

    @CallDebug
    @MethodChaining
    def loss_reduction_compile__(self, loss_reduction: Union[str, dict, Callable[["Context"], Tensor], NoneOrNothing]) -> "Compile":
        if not is_none_or_nothing(loss_reduction):
            loss_reduction = LossReductionFactory.get(loss_reduction)
        self.ctx.pipeline_ctx.loss_reduction = loss_reduction

    @CallDebug
    @MethodChaining
    def metrics_compile__(self, metrics: Union[Iterable[Metric], NoneOrNothing]) -> "Compile":
        if not is_none_or_nothing(metrics):
            metrics = MetricContainer(metrics)
        self.ctx.pipeline_ctx.metrics = metrics

    @CallDebug
    @MethodChaining
    def data_parser_compile__(self, data_parser: Union[DataParser, NoneOrNothing]) -> "Compile":
        self.ctx.pipeline_ctx.data_parser = data_parser if not is_none_or_nothing(data_parser) else IndexParser()

    @CallDebug
    @MethodChaining
    def optimizer_compile__(self, optimizer: Union[Optimizer, NoneOrNothing]) -> "Compile":
        self.ctx.pipeline_ctx.optimizer = optimizer

    @CallDebug
    @MethodChaining
    def lr_scheduler_compile__(self, lr_scheduler: Union[TorchLRScheduler, NoneOrNothing]) -> "Compile":
        self.ctx.pipeline_ctx.lr_scheduler = lr_scheduler

    @CallDebug
    @MethodChaining
    def train_end_compile__(self, train_end: int) -> "Compile":
        if not isinstance(train_end, int):
            classname = type(train_end).__name__
            logger.warning(f'``train_end`` should be ``int``, but ``{classname}`` found.')
        self.ctx.iteration_ctx.total = train_end

    @CallDebug
    @MethodChaining
    def train_start_compile__(self, train_start: int) -> "Compile":
        if not isinstance(train_start, int):
            classname = type(train_start).__name__
            logger.warning(f'``train_start`` should be ``int``, but ``{classname}`` found.')
        self.ctx.iteration_ctx.start = train_start

    @CallDebug
    @MethodChaining
    def train_provider_compile__(self, data: Union["AcceptableDataType", NoneOrNothing]) -> "Compile":
        if not is_none_or_nothing(data):
            data = data if isinstance(data, DataProvider) else ConstantProvider(data)
        self.ctx.pipeline_ctx.train_provider = data
    
    @CallDebug
    @MethodChaining
    def eval_provider_compile__(self, data: Union["AcceptableDataType", NoneOrNothing]) -> "Compile":
        if not is_none_or_nothing(data):
            data = data if isinstance(data, DataProvider) else ConstantProvider(data)
        self.ctx.pipeline_ctx.eval_provider = data

    @CallDebug
    @MethodChaining
    def grad_acc_compile__(self, grad_acc: int) -> "Compile":
        self.ctx.pipeline_ctx.grad_acc = grad_acc

    @CallDebug
    @MethodChaining
    def valid_freq_compile__(self, valid_freq: Union[int, Callable[["Context"], bool]]) -> "Compile":
        self.ctx.pipeline_ctx.valid_freq = valid_freq

    @CallDebug
    @MethodChaining
    def build_hook_compile__(self, build_hook: Union[str, BuildHook]) -> "Compile":
        if isinstance(build_hook, str):
            build_hook = build_registry.get(build_hook)()
        
        if not isinstance(build_hook, BuildHook):
            logger.warning(
                f'Build hook type mismatch. Expected: {BuildHook} -> Actual: {type(build_hook)}'
            )
        
        self.ctx.hook_ctx.build = build_hook

    @CallDebug
    @MethodChaining
    def launch_hook_compile__(self, launch_hook: Union[str, LaunchHook]) -> "Compile":
        if isinstance(launch_hook, str):
            launch_hook = launch_registry.get(launch_hook)()
        
        if not isinstance(launch_hook, LaunchHook):
            logger.warning(
                f'Launch hook type mismatch. Expected: {LaunchHook} -> Actual: {type(launch_hook)}'
            )
        
        self.ctx.hook_ctx.launch = launch_hook
    
    @CallDebug
    @MethodChaining
    def pipeline_profiler_compile__(self, pipeline_profiler: Union[str, PipelineProfiler]) -> "Compile":
        if isinstance(pipeline_profiler, str):
            pipeline_profiler = profiler_registry.get(pipeline_profiler)()
        
        if not isinstance(pipeline_profiler, PipelineProfiler):
            logger.warning(
                f'PipelineProfiler type mismatch. Expected: {PipelineProfiler} -> Actual: {type(pipeline_profiler)}'
            )
        
        self.ctx.pipeline_ctx.pipeline_profiler = pipeline_profiler

    @CallDebug
    @MethodChaining
    def model_state_compile__(self, model_state: Union[str, ModelState, Nothing]) -> "Compile":
        if isinstance(model_state, str):
            model_state = state_registry.get(model_state)()
        
        if not isinstance(model_state, (ModelState, Nothing)):
            logger.warning(
                f'ModelState type mismatch. Expected: {ModelState} or {Nothing} -> Actual: {type(model_state)}'
            )
        
        self.ctx.pipeline_ctx.model_state = model_state
        # Change model mode according to the state hook
        self.ctx.pipeline_ctx.model_state.set_model_mode(self.ctx)

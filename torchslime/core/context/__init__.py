from torchslime.utils.typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Iterable
)
from torchslime.components.data import ConstantProvider, DataParser, DataProvider, IndexParser
from torchslime.components.metric import MetricContainer, LossReductionFactory, Metric, LossFunc, LossFuncContainer
from torchslime.components.exception import APIMisused
from torchslime.utils import get_device, type_cast, count_params
from torchslime.log import logger
from torchslime.utils.bases import NOTHING, is_none_or_nothing
from torchslime.utils.decorators import CallDebug, MethodChaining
from torchslime.utils.typing import NUMBER
from torchslime.core.context.base import BaseContext
from torchslime.core.hooks.build import BuildHook, build_registry
from torchslime.core.hooks.launch import LaunchHook, launch_registry
from torchslime.core.hooks.plugin import PluginHook
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor


DATASET = Union[DataLoader, DataProvider]


class Context(BaseContext):

    def __init__(
        self,
        model,
        device=None,
        build_hook: Union[str, BuildHook] = 'vanilla',
        launch_hook: Union[str, LaunchHook] = 'vanilla'
    ):
        # init context
        super().__init__()
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)
        # build hooks
        self.compile_build_hook(build_hook)
        self.compile_launch_hook(launch_hook)

    @CallDebug(module_name='Context.train')
    @MethodChaining
    def train(
        self,
        train_dataset: DATASET,
        train_end: int = 1,
        val_dataset: DATASET = NOTHING,
        grad_acc: int = 1,
        valid_freq: Union[int, List[int], Callable[[BaseContext], bool]] = 1,
        train_start: int = 0
    ) -> 'Context':
        if is_none_or_nothing(self.run_ctx.train):
            logger.error('``train`` called before train handlers are built. Call ``build_train`` first.')
            raise APIMisused('train')
        
        self.compile_train_end(train_end)
        self.compile_dataset(train_dataset, 'train')
        self.compile_dataset(val_dataset, 'eval')
        self.compile_grad_acc(grad_acc)
        self.compile_valid_freq(valid_freq)
        self.compile_train_start(train_start)

        logger.info(self.hook_ctx.launch.get_device_info(self))

        _handler_call(self.run_ctx.train, self)

    @CallDebug(module_name='Context.build_train')
    @MethodChaining
    def build_train(self) -> 'Context':
        self.hook_ctx.build._build_train(self)
    
    @CallDebug(module_name='Context.display_train')
    @MethodChaining
    def display_train(self) -> 'Context':
        if is_none_or_nothing(self.run_ctx.train):
            logger.warn('``display_train`` called before train handlers are built.')
        self.run_ctx.train.display()

    @CallDebug(module_name='Context.eval')
    @MethodChaining
    def eval(
        self,
        dataset: DATASET
    ) -> 'Context':
        if is_none_or_nothing(self.run_ctx.eval):
            logger.error('``eval`` called before eval handlers are built. Call ``build_eval`` first.')
            raise APIMisused('eval')
        
        self.compile_dataset(dataset, 'eval')

        logger.info(self.hook_ctx.launch.get_device_info(self))
        
        _handler_call(self.run_ctx.eval, self)

    @CallDebug(module_name='Context.build_eval')
    @MethodChaining
    def build_eval(self) -> 'Context':
        self.hook_ctx.build._build_eval(self)
    
    @CallDebug(module_name='Context.display_eval')
    @MethodChaining
    def display_eval(self) -> 'Context':
        if is_none_or_nothing(self.run_ctx.eval):
            logger.warn('``display_eval`` called before eval handlers are built.')
        self.run_ctx.eval.display()

    @CallDebug(module_name='Context.predict')
    @MethodChaining
    def predict(
        self,
        dataset: DATASET
    ) -> 'Context':
        if is_none_or_nothing(self.run_ctx.predict):
            logger.error('``predict`` called before predict handlers are built. Call ``build_predict`` first.')
            raise APIMisused('predict')
        
        self.compile_dataset(dataset, 'eval')

        logger.info(self.hook_ctx.launch.get_device_info(self))
        
        _handler_call(self.run_ctx.predict, self)

    @CallDebug(module_name='Context.build_predict')
    @MethodChaining
    def build_predict(self) -> 'Context':
        self.hook_ctx.build._build_predict(self)
    
    @CallDebug(module_name='Context.display_predict')
    @MethodChaining
    def display_predict(self) -> 'Context':
        if is_none_or_nothing(self.run_ctx.predict):
            logger.warn('``display_predict`` called before predict handlers are built.')
        self.run_ctx.predict.display()

    @CallDebug(module_name='Context.summary')
    @MethodChaining
    def summary(self) -> 'Context':
        # TODO
        pass

    @CallDebug(module_name='Context.install_plugins')
    @MethodChaining
    def install_plugins(self, plugins: Iterable[PluginHook]) -> 'Context':
        self.hook_ctx.plugins.extend(plugins)

    @CallDebug(module_name='Context.count_params')
    @MethodChaining
    def count_params(self, format: str = None, decimal: int = 2, log: bool = True) -> 'Context':
        result = count_params(self.model, format, decimal)
        if log is True:
            logger.info(f'Model parameters: {result}')
        return result

    @CallDebug(module_name='Context.compile')
    @MethodChaining
    def compile(
        self,
        loss_func_list: Union[Iterable[LossFunc], None] = None,
        loss_reduction: Union[str, dict, Callable[[BaseContext], Tensor], None] = None,
        metrics: Union[Iterable[Metric], None] = None,
        optimizer: Union[str, Optimizer] = None,
        lr: NUMBER = None,
        lr_decay: Any = None,
        optimizer_options: Optional[Dict] = None,
        lr_decay_options: Optional[Dict] = None,
        data_parser: Optional[DataParser] = None
    ) -> 'Context':
        self.compile_loss_func(loss_func_list)
        self.compile_loss_reduction(loss_reduction)
        self.compile_metrics(metrics)
        self.compile_data_parser(data_parser)
        self.compile_optimizer(optimizer, lr, optimizer_options)
        self.compile_lr_decay(lr_decay, lr_decay_options)

    @CallDebug(module_name='Context.compile_loss_func')
    @MethodChaining
    def compile_loss_func(self, loss_func_list) -> 'Context':
        if loss_func_list is not None:
            self.run_ctx.loss_func = LossFuncContainer(loss_func_list)

    @CallDebug(module_name='Context.compile_loss_reduction')
    @MethodChaining
    def compile_loss_reduction(self, loss_reduction) -> 'Context':
        if loss_reduction is not None:
            self.run_ctx.loss_reduction = LossReductionFactory.get(loss_reduction)

    @CallDebug(module_name='Context.compile_metrics')
    @MethodChaining
    def compile_metrics(self, metrics) -> 'Context':
        if metrics is not None:
            self.run_ctx.metrics = MetricContainer(metrics)

    @CallDebug(module_name='Context.compile_data_parser')
    @MethodChaining
    def compile_data_parser(self, data_parser) -> 'Context':
        if data_parser is not None:
            self.run_ctx.data_parser = data_parser if data_parser is not NOTHING else IndexParser()

    @CallDebug(module_name='Context.compile_optimizer')
    @MethodChaining
    def compile_optimizer(self, optimizer, lr, optimizer_options) -> 'Context':
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.run_ctx.optimizer = optimizer

    @CallDebug(module_name='Context.compile_lr_decay')
    @MethodChaining
    def compile_lr_decay(self, lr_decay, lr_decay_options) -> 'Context':
        if lr_decay is not None:
            if isinstance(lr_decay, str) is False:
                self.run_ctx.lr_decay = lr_decay

    @CallDebug(module_name='Context.compile_train_end')
    @MethodChaining
    def compile_train_end(self, train_end: int) -> 'Context':
        if not isinstance(train_end, int):
            classname = type(train_end).__name__
            logger.warn(f'``train_end`` should be ``int``, but ``{classname}`` found.')
        self.iteration_ctx.total = train_end

    @CallDebug(module_name='Context.compile_train_start')
    @MethodChaining
    def compile_train_start(self, train_start: int) -> 'Context':
        if not isinstance(train_start, int):
            classname = type(train_start).__name__
            logger.warn(f'``train_start`` should be ``int``, but ``{classname}`` found.')
        self.iteration_ctx.start = train_start

    @CallDebug(module_name='Context.compile_dataset')
    @MethodChaining
    def compile_dataset(self, dataset, mode: str) -> 'Context':
        if dataset is not None:
            if dataset is not NOTHING:
                dataset = dataset if isinstance(dataset, DataProvider) else ConstantProvider(dataset)

            mode_supported = ['train', 'eval']
            if mode not in mode_supported:
                logger.warn('compile_dataset mode not supported.')
            setattr(self.run_ctx, f'{mode}_provider', dataset)

    @CallDebug(module_name='Context.compile_grad_acc')
    @MethodChaining
    def compile_grad_acc(self, grad_acc: int) -> 'Context':
        if grad_acc is not None:
            self.run_ctx.grad_acc = grad_acc

    def is_distributed(self) -> bool:
        return self.hook_ctx.launch.is_distributed()

    @CallDebug(module_name='Context.compile_build_hook')
    @MethodChaining
    def compile_build_hook(self, build_hook: Union[str, BuildHook]) -> 'Context':
        if isinstance(build_hook, str):
            self.hook_ctx.build = build_registry.get(build_hook)()
        elif isinstance(build_hook, BuildHook):
            self.hook_ctx.build = build_hook
        else:
            logger.warn('Build hook type unsupported.')
    
    @CallDebug(module_name='Context.compile_launch_hook')
    @MethodChaining
    def compile_launch_hook(self, launch_hook: Union[str, LaunchHook]) -> 'Context':
        if isinstance(launch_hook, str):
            self.hook_ctx.launch = launch_registry.get(launch_hook)()
        elif isinstance(launch_hook, LaunchHook):
            self.hook_ctx.launch = launch_hook
        else:
            logger.warn('Launch hook type unsupported.')

    @CallDebug(module_name='Context.compile_valid_freq')
    @MethodChaining
    def compile_valid_freq(self, valid_freq: Union[int, Callable[[BaseContext], bool]]) -> 'Context':
        self.run_ctx.valid_freq = valid_freq

#
# Try with handler exceptions
#

from torchslime.core.handlers import Handler
from torchslime.components.exception import HandlerException, HandlerTerminate

def _handler_call(handler: Handler, ctx: Context):
    try:
        handler(ctx)
    except HandlerTerminate as ht:
        handler.display_traceback(target_handlers=[ht.raise_handler], wrap_func='terminate', level='info')
        logger.info(f'Handler terminated with message: {ht.msg}')
    except HandlerException as he:
        handler.display_traceback(target_handlers=[he.exception_handler])
        raise he.exception

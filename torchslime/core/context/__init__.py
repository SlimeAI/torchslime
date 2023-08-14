from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    Callable,
    Iterable
)
from torchslime.components.data import ConstantProvider, DataParser, DataProvider, IndexParser
from torchslime.components.metric import M_SEQ, MetricContainer, LossReductionFactory
from torchslime.utils import get_device, type_cast, count_params
from torchslime.log import logger
from torchslime.utils.bases import NOTHING, BaseList, is_nothing
from torchslime.utils.decorators import CallDebug, MethodChaining
from torchslime.utils.tstype import NUMBER, INT_SEQ_N
from torchslime.core.context.base import BaseContext
from torchslime.core.hooks.build import BuildHook, build_registry
from torchslime.core.hooks.launch import LaunchHook, launch_registry
from torchslime.core.hooks.plugin import PluginHook
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor


ContextSelf = TypeVar('ContextSelf', bound='Context')
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

    @CallDebug(module_name='Context.Train')
    def train(
        self,
        train_dataset: DATASET,
        train_end: int = 1,
        val_dataset: DATASET = NOTHING,
        grad_acc: int = 1,
        valid_freq: Union[int, List[int], Callable[[BaseContext], bool]] = 1,
        train_start: int = 0
    ):
        self.compile_train_end(train_end)
        self.compile_dataset(train_dataset, 'train')
        self.compile_dataset(val_dataset, 'eval')
        self.compile_grad_acc(grad_acc)
        self.compile_valid_freq(valid_freq)
        self.compile_train_start(train_start)

        # build train handler
        self.hook_ctx.build._build_train(self)
        self.run_ctx.train.display()

        logger.info(self.hook_ctx.launch.get_device_info(self))

        from torchslime.components.exception import HandlerException, HandlerTerminate
        try:
            self.run_ctx.train(self)
        except HandlerTerminate as ht:
            self.run_ctx.train.display_traceback(target_handlers=ht.raise_handler, wrap_func='terminate', level='info')
            logger.info('Handler terminated with message: {msg}'.format(msg=ht.msg))
        except HandlerException as he:
            self.run_ctx.train.display_traceback(target_handlers=he.exception_handler)
            raise he.exception

    @CallDebug(module_name='Context.Predict')
    def predict(
        self,
        dataset: DATASET
    ):
        self.compile_dataset(dataset, 'eval')

        # build predict handler
        self.hook_ctx.build._build_predict(self)
        self.run_ctx.predict.display()

        logger.info(self.hook_ctx.launch.get_device_info(self))
        self.run_ctx.predict(self)

    @CallDebug(module_name='Context.Eval')
    def eval(
        self,
        dataset: DATASET
    ):
        self.compile_dataset(dataset, 'eval')

        # build eval handler
        self.hook_ctx.build._build_eval(self)
        self.run_ctx.eval.display()

        logger.info(self.hook_ctx.launch.get_device_info(self))
        self.run_ctx.eval(self)

    @CallDebug(module_name='Context.Summary')
    def summary(self):
        # TODO
        pass

    @CallDebug(module_name='Context.install_plugins')
    def install_plugins(self, plugins: Iterable[PluginHook]):
        self.hook_ctx.plugins.extend(plugins)

    @CallDebug(module_name='Context.CountParams')
    def count_params(self, format: str = None, decimal: int = 2, log: bool = True):
        result = count_params(self.model, format, decimal)
        if log is True:
            logger.info('Model parameters: {0}'.format(result))
        return result

    @CallDebug(module_name='Context.Compile')
    @MethodChaining
    def compile(
        self,
        loss_func = None,
        loss_reduction: Union[str, dict, Callable[[BaseContext], Tensor], None] = None,
        metrics: M_SEQ = None,
        optimizer: Union[str, Optimizer] = None,
        lr: NUMBER = None,
        lr_decay: Any = None,
        optimizer_options: Optional[Dict] = None,
        lr_decay_options: Optional[Dict] = None,
        data_parser: Optional[DataParser] = None
    ) -> Union[ContextSelf, 'Context']:
        self.compile_loss_func(loss_func)
        self.compile_loss_reduction(loss_reduction)
        self.compile_metrics(metrics)
        self.compile_data_parser(data_parser)
        self.compile_optimizer(optimizer, lr, optimizer_options)
        self.compile_lr_decay(lr_decay, lr_decay_options)

    @CallDebug(module_name='Context.compile_loss_func')
    def compile_loss_func(self, loss_func):
        if loss_func is not None:
            self.run_ctx.loss_func = loss_func

    @CallDebug(module_name='Context.compile_loss_reduction')
    def compile_loss_reduction(self, loss_reduction):
        if loss_reduction is not None:
            self.run_ctx.loss_reduction = LossReductionFactory.get(loss_reduction)

    @CallDebug(module_name='Context.compile_metrics')
    def compile_metrics(self, metrics):
        if metrics is not None:
            self.run_ctx.metrics = MetricContainer(metrics) if is_nothing(metrics) is False else NOTHING

    @CallDebug(module_name='Context.compile_data_parser')
    def compile_data_parser(self, data_parser):
        if data_parser is not None:
            self.run_ctx.data_parser = data_parser if is_nothing(data_parser) is False else IndexParser()

    @CallDebug(module_name='Context.compile_optimizer')
    def compile_optimizer(self, optimizer, lr, optimizer_options):
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.run_ctx.optimizer = optimizer

    @CallDebug(module_name='Context.compile_lr_decay')
    def compile_lr_decay(self, lr_decay, lr_decay_options):
        if lr_decay is not None:
            if isinstance(lr_decay, str) is False:
                self.run_ctx.lr_decay = lr_decay

    @CallDebug(module_name='Context.compile_train_end')
    def compile_train_end(self, train_end: int):
        if not isinstance(train_end, int):
            logger.warn('``train_end`` should be ``int``, but ``{}`` found.'.format(type(train_end).__name__))
        self.iteration_ctx.total = train_end

    @CallDebug(module_name='Context.compile_train_start')
    def compile_train_start(self, train_start: int):
        if not isinstance(train_start, int):
            logger.warn('``train_start`` should be ``int``, but ``{}`` found.'.format(type(train_start).__name__))
        self.iteration_ctx.start = train_start

    @CallDebug(module_name='Context.compile_dataset')
    def compile_dataset(self, dataset, mode: str):
        if dataset is not None:
            if is_nothing(dataset):
                dataset = NOTHING
            else:
                dataset = dataset if isinstance(dataset, DataProvider) else ConstantProvider(dataset)

            mode_supported = ['train', 'eval']
            if mode not in mode_supported:
                logger.warn('compile_dataset mode not supported.')
            setattr(self.run_ctx, '{}_provider'.format(mode), dataset)

    @CallDebug(module_name='Context.compile_grad_acc')
    def compile_grad_acc(self, grad_acc: int):
        if grad_acc is not None:
            self.run_ctx.grad_acc = grad_acc

    def is_distributed(self):
        return self.hook_ctx.launch.is_distributed()

    @CallDebug(module_name='Context.compile_build_hook')
    def compile_build_hook(self, build_hook: Union[str, BuildHook]):
        if isinstance(build_hook, str):
            self.hook_ctx.build = build_registry.get(build_hook)()
        elif isinstance(build_hook, BuildHook):
            self.hook_ctx.build = build_hook
        else:
            logger.warn('Build hook type unsupported.')
    
    @CallDebug(module_name='Context.compile_launch_hook')
    def compile_launch_hook(self, launch_hook: Union[str, LaunchHook]):
        if isinstance(launch_hook, str):
            self.hook_ctx.launch = launch_registry.get(launch_hook)()
        elif isinstance(launch_hook, LaunchHook):
            self.hook_ctx.launch = launch_hook
        else:
            logger.warn('Launch hook type unsupported.')

    @CallDebug(module_name='Context.compile_valid_freq')
    def compile_valid_freq(self, valid_freq: Union[int, Callable[[BaseContext], bool]]):
        self.run_ctx.valid_freq = valid_freq

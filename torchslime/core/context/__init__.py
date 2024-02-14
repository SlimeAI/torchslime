from .base import BaseContext
from .compile import Compile
from torchslime.utils.typing import (
    List,
    Union,
    Callable,
    Iterable,
    is_none_or_nothing,
    NOTHING,
    Missing,
    MISSING
)
from torchslime.components.data import DataProvider
from torchslime.components.exception import APIMisused
from torchslime.logging.logger import logger
from torchslime.utils import get_device, type_cast, count_params
from torchslime.utils.decorators import CallDebug, MethodChaining
from torchslime.utils.bases import AttrObserver, AttrObserve
from torchslime.core.hooks.build import BuildHook
from torchslime.core.hooks.launch import LaunchHook
from torchslime.core.hooks.profiler import ProfilerHook
from torchslime.core.hooks.plugin import PluginHook
from torch.utils.data import DataLoader


AcceptableDataType = Union[DataLoader, DataProvider]


class Context(BaseContext, AttrObserver):

    def __init__(
        self,
        model,
        device=None,
        build_hook: Union[str, BuildHook] = 'vanilla',
        launch_hook: Union[str, LaunchHook] = 'vanilla',
        profiler_hook: Union[str, ProfilerHook] = 'vanilla',
        compile: Union[Compile, Missing] = MISSING
    ):
        BaseContext.__init__(self)
        AttrObserver.__init__(self)
        self.compile = Compile() if compile is MISSING else compile
        
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)
        
        # compile hooks
        self.compile.compile_hook_ctx(
            build_hook=build_hook,
            launch_hook=launch_hook,
            profiler_hook=profiler_hook
        )

    @CallDebug(module_name='Context.train')
    @MethodChaining
    def train(
        self,
        train_data: AcceptableDataType,
        train_end: int = 1,
        val_data: AcceptableDataType = NOTHING,
        grad_acc: int = 1,
        valid_freq: Union[int, List[int], Callable[[BaseContext], bool]] = 1,
        train_start: int = 0
    ) -> 'Context':
        if is_none_or_nothing(self.run_ctx.train_container):
            logger.error('``train`` called before train handlers are built. Call ``build_train`` first.')
            raise APIMisused('train')
        
        self.compile(
            train_end=train_end,
            train_provider=train_data,
            eval_provider=val_data,
            grad_acc=grad_acc,
            valid_freq=valid_freq,
            train_start=train_start
        )
        logger.info(self.hook_ctx.launch.get_device_info(self))
        _handler_call(self.run_ctx.train_container, self)

    @CallDebug(module_name='Context.build_train')
    @MethodChaining
    def build_train(self) -> 'Context':
        self.hook_ctx.build._build_train(self)
    
    @CallDebug(module_name='Context.display_train')
    @MethodChaining
    def display_train(self) -> 'Context':
        if is_none_or_nothing(self.run_ctx.train_container):
            logger.warning('``display_train`` called before train handlers are built.')
        self.run_ctx.train_container.display()

    @CallDebug(module_name='Context.eval')
    @MethodChaining
    def eval(
        self,
        data: AcceptableDataType
    ) -> 'Context':
        if is_none_or_nothing(self.run_ctx.eval_container):
            logger.error('``eval`` called before eval handlers are built. Call ``build_eval`` first.')
            raise APIMisused('eval')
        
        self.compile.eval_provider_compile__(data)
        logger.info(self.hook_ctx.launch.get_device_info(self))
        _handler_call(self.run_ctx.eval_container, self)

    @CallDebug(module_name='Context.build_eval')
    @MethodChaining
    def build_eval(self) -> 'Context':
        self.hook_ctx.build._build_eval(self)
    
    @CallDebug(module_name='Context.display_eval')
    @MethodChaining
    def display_eval(self) -> 'Context':
        if is_none_or_nothing(self.run_ctx.eval_container):
            logger.warning('``display_eval`` called before eval handlers are built.')
        self.run_ctx.eval_container.display()

    @CallDebug(module_name='Context.predict')
    @MethodChaining
    def predict(
        self,
        data: AcceptableDataType
    ) -> 'Context':
        if is_none_or_nothing(self.run_ctx.predict_container):
            logger.error('``predict`` called before predict handlers are built. Call ``build_predict`` first.')
            raise APIMisused('predict')
        
        self.compile.eval_provider_compile__(data)
        logger.info(self.hook_ctx.launch.get_device_info(self))
        _handler_call(self.run_ctx.predict_container, self)

    @CallDebug(module_name='Context.build_predict')
    @MethodChaining
    def build_predict(self) -> 'Context':
        self.hook_ctx.build._build_predict(self)
    
    @CallDebug(module_name='Context.display_predict')
    @MethodChaining
    def display_predict(self) -> 'Context':
        if is_none_or_nothing(self.run_ctx.predict_container):
            logger.warning('``display_predict`` called before predict handlers are built.')
        self.run_ctx.predict_container.display()

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

    def is_distributed(self) -> bool:
        return self.hook_ctx.launch.is_distributed()
    
    @AttrObserve
    def launch_observe__(self, new_value, old_value):
        pass

#
# Try with handler exceptions
#

from torchslime.core.handlers import Handler
from torchslime.components.exception import HandlerException, HandlerTerminate

def _handler_call(handler: Handler, ctx: Context):
    try:
        handler(ctx)
    except HandlerTerminate as ht:
        handler.display(target_handlers=[ht.raise_handler], wrap_func='terminate')
        logger.info(f'Handler terminated with message: {ht.msg}')
    except HandlerException as he:
        handler.display(target_handlers=[he.exception_handler], wrap_func='exception')
        raise he.exception

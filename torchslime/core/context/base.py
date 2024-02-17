from torchslime.utils.bases import Base
from torchslime.utils.typing import (
    Any,
    Sequence,
    Union,
    Dict,
    Callable,
    List,
    TYPE_CHECKING,
    NOTHING,
    Nothing,
    is_none_or_nothing
)
from torchslime.components.exception import APIMisused
from torchslime.logging.logger import logger, LoggerKwargs

if TYPE_CHECKING:
    from torchslime.logging.rich import (
        SlimeLiveLauncher,
        SlimeGroup,
        ProfileProgress,
        SlimeProgressLauncher
    )
    from .compile import Compile
    from torchslime.utils.typing import (
        TorchDevice,
        TorchModule,
        TorchTensor,
        TorchOptimizer,
        TorchLRScheduler
    )


class BaseContext(Base):
    """
    Base Context in the whole life time.
    """

    def __init__(self):
        super().__init__()
        
        """
        context attribute placeholders(for code hints)
        """
        # TODO model shard
        # device for pytorch
        self.device: Union[str, "TorchDevice"] = NOTHING
        # model
        self.model: "TorchModule" = NOTHING
        # run context
        self.run_ctx: RunContext = RunContext()
        # information about iteration
        self.iteration_ctx: IterationContext = IterationContext()
        # information in one step
        self.step_ctx: StepContext = StepContext()
        # handler context
        self.handler_ctx: HandlerContext = HandlerContext()
        # custom context
        self.custom_ctx: CustomContext = CustomContext()
        # hook context
        self.hook_ctx: HookContext = HookContext()
        # display context
        self.display_ctx: DisplayContext = DisplayContext()

    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        self.__model = value

    #
    # ``compile`` property
    #

    @property
    def compile(self) -> "Compile":
        from .compile import Compile
        
        if not self.hasattr__('_BaseContext__compile'):
            logger.warning(
                'Property ``compile`` has not been bound to an object yet.'
            )
        compile = self.__compile
        if not isinstance(compile, Compile):
            logger.warning(
                f'Property ``compile`` is not set to a ``Compile`` instance, but {compile}. This may '
                'cause some unknown problems.'
            )
        elif compile.ctx is not self:
            raise APIMisused(
                f'Bindings between ``compile`` and ``Context`` mismatch. ``compile`` is bound to {compile.ctx}, '
                f'while ``Context`` is {self}.'
            )
        return compile

    @compile.setter
    def compile(self, value: "Compile") -> None:
        if not is_none_or_nothing(value.ctx) and \
                value.ctx is not self:
            raise APIMisused(
                f'The property ``compile`` ({value}) being set has already been bound to another ``Context`` object ({value.ctx}). '
                f'You should unbind it from the other ``Context`` ({value.ctx}) using ``del`` first.'
            )
        value.ctx = self
        self.__compile = value

    @compile.deleter
    def compile(self) -> None:
        self.compile.ctx = NOTHING
        del self.__compile

    def ctx_check(self, items: Union[str, Sequence[str]], silent: bool = True):
        # check single item
        def _check(_item):
            _result = super(BaseContext, self).check__(_item)
            if _result is False:
                msg = 'Context check failed: got NOTHING with key \'%s\'.' % _item
                if silent is True:
                    logger.debug(msg, LoggerKwargs(stacklevel=3))
                else:
                    logger.warning(msg, LoggerKwargs(stacklevel=3))
            return _result

        if isinstance(items, (list, tuple)):
            # sequence value
            for item in items:
                if _check(str(item)) is False:
                    return False
            return True
        else:
            # single value
            return _check(str(items))


class TempContext(Base):
    """Temp context that defines a initialize method to quickly reset the context.

    Args:
        Base (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        # initialize
        self.initialize()
    
    def initialize(self):
        pass


class StepContext(TempContext):

    def initialize(self):
        """
        step context attribute placeholders(for code hints)
        """
        # data input to the model
        self.x: Any = NOTHING
        # output of the model
        self.y_pred: Any = NOTHING
        # label of the data(or expected output in unsupervised learning)
        self.y_true: Any = NOTHING
        # metrics of the step
        self.metrics: Dict = NOTHING
        # loss tensor(s) of the step
        self.loss: Union[Dict[str, "TorchTensor"], Nothing] = NOTHING
        # loss value(s) of the step
        self.loss_values: Union[Dict[str, float], Nothing] = NOTHING
        # extra data passed to the context
        self.extra: Any = NOTHING
        # current iteration step
        self.current: int = NOTHING
        # total steps of iteration
        self.total: int = NOTHING
        # original batch data of the iteration of dataloader
        self.batch: Any = NOTHING


class IterationContext(TempContext):

    def initialize(self):
        """
        epoch context attribute placeholders(for code hints)
        """
        # iteration information (epoch/step)
        self.current: int = NOTHING
        self.total: int = NOTHING
        self.start: int = 0
        # average information in one period (e.g. epoch or a specified number of steps)
        from torchslime.components.metric import MeterDict
        # average train metrics
        self.train_metrics: MeterDict = MeterDict()
        # average eval metrics
        self.eval_metrics: MeterDict = MeterDict()
        # average train loss value(s)
        self.train_loss_values: MeterDict = MeterDict()
        # average eval loss value(s)
        self.eval_loss_values: MeterDict = MeterDict()


class RunContext(TempContext):
   
    def initialize(self):
        # handler containers that define the process of training, evaluating and predicting.
        from torchslime.core.handlers import HandlerContainer
        self.train_container: Union[HandlerContainer, Nothing] = NOTHING
        self.eval_container: Union[HandlerContainer, Nothing] = NOTHING
        self.predict_container: Union[HandlerContainer, Nothing] = NOTHING
        
        # data loader
        self.train_loader = NOTHING
        self.eval_loader = NOTHING
        # validation freq
        self.valid_freq: Union[int, List[int], Callable[[BaseContext], bool]] = 1
        # optimizer
        self.optimizer: "TorchOptimizer" = NOTHING
        # loss_func
        from torchslime.components.metric import LossFuncContainer
        self.loss_func: Union[LossFuncContainer, Nothing] = NOTHING
        # gradient accumulation
        self.grad_acc: int = 1
        # learning rate scheduler
        self.lr_scheduler: "TorchLRScheduler" = NOTHING
        # data provider
        from torchslime.components.data import DataProvider
        self.train_provider: DataProvider = NOTHING
        self.eval_provider: DataProvider = NOTHING
        # data parser
        from torchslime.components.data import DataParser, IndexParser
        # the data parser should be set to IndexParser as default
        self.data_parser: DataParser = IndexParser()
        # metric container
        from torchslime.components.metric import MetricContainer
        self.metrics: Union[MetricContainer, Nothing] = NOTHING
        # loss reduction func
        from torchslime.components.metric import LossReductionFactory
        self.loss_reduction: Callable[[BaseContext], "TorchTensor"] = LossReductionFactory.get('mean')


class HandlerContext(TempContext):

    def initialize(self):
        # base handlers
        from torchslime.core import handlers
        self.Handler = handlers.Handler
        self.Container = handlers.HandlerContainer
        
        from torchslime.core.handlers import common
        # the root container, should be used only once in a single container structure
        self.RootContainer = common.RootContainer
        # common handlers
        self.EmptyHandler = common.EmptyHandler
        self.EpochIterationContainer = common.EpochIterationContainer
        self.IterationContainer = common.IterationContainer
        self.StepIterationContainer = common.StepIterationContainer
        self.ForwardHandler = common.ForwardHandler
        self.LossHandler = common.LossHandler
        self.BackwardHandler = common.BackwardHandler
        self.OptimizerContainer = common.OptimizerContainer
        self.MetricHandler = common.MetricHandler
        self.MeterInitHandler = common.MeterInitHandler
        self.MeterHandler = common.MeterHandler
        self.GatherAverageHandler = common.GatherAverageHandler
        self.LoggingHandler = common.LoggingHandler
        self.LRScheduleHandler = common.LRScheduleHandler
        self.FuncHandler = common.FuncHandler
        
        # handler wrappers
        from torchslime.core.handlers import wrappers
        self.Wrapper = wrappers.HandlerWrapper
        self.WrapperContainer = wrappers.HandlerWrapperContainer
        self.StateWrapper = wrappers.StateWrapper
        self.ConditionWrapper = wrappers.ConditionWrapper


class CustomContext(TempContext):

    def initialize(self):
        self.__dict__.clear()
        logger.debug('Custom context has been initialized.')


class HookContext(TempContext):

    def initialize(self):
        # hooks
        from torchslime.core.hooks.plugin import PluginContainer
        self.plugins: PluginContainer = PluginContainer()
        
        from torchslime.core.hooks.launch import LaunchHook
        self.launch: Union[LaunchHook, Nothing] = NOTHING
        
        from torchslime.core.hooks.build import BuildHook
        self.build: Union[BuildHook, Nothing] = NOTHING
        
        from torchslime.core.hooks.state import StateHook
        self.state: Union[StateHook, Nothing] = NOTHING
        
        from torchslime.core.hooks.profiler import ProfilerHook
        self.profiler: Union[ProfilerHook, Nothing] = NOTHING


class DisplayContext(TempContext):
    
    def initialize(self):
        self.live_launcher: Union["SlimeLiveLauncher", Nothing] = NOTHING
        self.live_group: Union["SlimeGroup", Nothing] = NOTHING
        self.handler_progress: Union["ProfileProgress", "SlimeProgressLauncher", Nothing] = NOTHING
        self.progress_task_id: Union[int, Nothing] = NOTHING

from torchslime.utils import TorchComm
from torch.nn import Module
from torch import device, Tensor
from torch.optim.optimizer import Optimizer
from torchslime.utils.bases import NOTHING, Base, Nothing
from torchslime.utils.tstype import NUMBER
from typing import Any, Sequence, Union, Dict, Tuple, Callable, Type
from torchslime.log import logger


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
        self.device: Union[str, device] = NOTHING
        # model
        self.model: Module = NOTHING
        # run context
        self.run_ctx: RunContext = RunContext(ctx=self)
        # information about iteration
        self.iteration_ctx: IterationContext = IterationContext(ctx=self)
        # information in one step
        self.step_ctx: StepContext = StepContext(ctx=self)
        # handler context
        self.handler_ctx: HandlerContext = HandlerContext(ctx=self)
        # custom context
        self.custom_ctx: CustomContext = CustomContext(ctx=self)
        # inner context
        self.inner_ctx: InnerContext = InnerContext(ctx=self)
        # hook context
        self.hook_ctx: HookContext = HookContext(ctx=self)
        # distributed context
        self.distributed_ctx: DistributedContext = DistributedContext(ctx=self)

    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        self.__model = value

    def ctx_check(self, items: Union[str, Sequence[str]], silent: bool = True):
        # check single item
        def _check(_item):
            _result = super(BaseContext, self).check__(_item)
            if _result is False:
                msg = 'Context check failed: got NOTHING with key \'%s\'.' % _item
                if silent is True:
                    logger.debug(msg, _frame_offset=2)
                else:
                    logger.warn(msg, _frame_offset=2)
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
    def __init__(self, ctx: BaseContext = NOTHING):
        super().__init__()
        # get context
        self.ctx = ctx
        # initialize
        self.initialize()
    
    def initialize(self):
        pass


class StepContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
        self.loss: Union[Tensor, Dict[Any, Tensor], Nothing] = NOTHING
        # loss value(s) of the step
        self.loss_value: Union[float, Dict[Any, float], Nothing] = NOTHING
        # extra data passed to the context
        self.extra: Any = NOTHING
        # current iteration step
        self.current: int = NOTHING
        # total steps of iteration
        self.total: int = NOTHING
        # timestamp at the beginning of the step
        self.time: Union[int, float] = NOTHING
        # tuple of current step and total steps, it's used for progress visualization in the console
        self.progress: Tuple[int, int] = NOTHING
        # original batch data of the iteration of dataloader
        self.batch: Any = NOTHING


class IterationContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self):
        """
        epoch context attribute placeholders(for code hints)
        """
        # global step information
        # total steps
        self.total_steps: int = NOTHING
        # the current step
        self.current_step: int = NOTHING
        # epoch information
        # total epochs
        self.total_epochs: int = NOTHING
        # the current epoch
        self.current_epoch: int = NOTHING
        # average information in one period (e.g. epoch or a specified number of steps)
        # average train metrics
        self.train_metrics: Dict = NOTHING
        # average eval metrics
        self.eval_metrics: Dict = NOTHING
        # average train loss value(s)
        self.train_loss_value: Union[float, Dict[Any, float], Nothing] = NOTHING
        # average eval loss value(s)
        self.eval_loss_value: Union[float, Dict[Any, float], Nothing] = NOTHING


class RunContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        # handler containers that define the process of training, evaluating and predicting.
        from torchslime.core.handlers import HandlerContainer
        self.train: Union[HandlerContainer, Nothing] = NOTHING
        self.eval: Union[HandlerContainer, Nothing] = NOTHING
        self.predict: Union[HandlerContainer, Nothing] = NOTHING
        
        # optimizer
        self.optimizer: Optimizer = NOTHING
        # loss_func
        self.loss_func: Module = NOTHING
        # gradient accumulation
        self.grad_acc: int = 1
        # learning rate
        self.lr: NUMBER = NOTHING
        # learning rate decay
        self.lr_decay: Any = NOTHING
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
        self.metrics: MetricContainer = NOTHING
        # loss wrapper
        from torchslime.components.metric import LossWrapper
        self.loss_wrapper: Type[LossWrapper] = LossWrapper
        # loss reduction func
        from torchslime.components.metric import LossReductionFactory
        self.loss_reduction: Callable[[BaseContext], Tensor] = LossReductionFactory.get('mean')


class HandlerContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        from torchslime.core import handlers
        # handler class
        self.Handler = handlers.Handler
        self.Container = handlers.HandlerContainer
        self.Wrapper = handlers.HandlerWrapper
        self.EpochIteration = handlers.EpochIterationHandler
        self.Iteration = handlers.IterationHandler
        self.StepIteration = handlers.StepIterationHandler
        self.Forward = handlers.ForwardHandler
        self.Loss = handlers.LossHandler
        self.Backward = handlers.BackwardHandler
        self.Optimizer = handlers.OptimizerHandler
        self.Metrics = handlers.MetricsHandler
        self.AverageInit = handlers.AverageInitHandler
        self.Average = handlers.AverageHandler
        self.GatherAverage = handlers.GatherAverageHandler
        self.Display = handlers.DisplayHandler
        self.State = handlers.StateHandler
        self.LRDecay = handlers.LRDecayHandler
        self.Lambda = handlers.LambdaHandler


class CustomContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        self.__dict__.clear()
        logger.debug('Custom context has been initialized.')


class InnerContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        self.__dict__.clear()
        logger.debug('Inner context has been initialized.')


class HookContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        self.valid_mode = 'epoch'
        self.train_mode = 'epoch'
        self.lr_decay_mode = 'epoch'
        
        # hooks
        from ..hooks.plugin import PluginContainer
        self.plugins: PluginContainer = PluginContainer()
        from ..hooks.launch import LaunchHook
        self.launch: LaunchHook = NOTHING
        from ..hooks.build import BuildHook
        self.build: BuildHook = NOTHING
        from ..hooks.state import StateHook
        self.state: StateHook = NOTHING


class DistributedContext(TempContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        self.torch_comm: TorchComm = TorchComm()

    def is_ready(self):
        """
        Check whether the torch distributed settings are ready.
        """
        self.ctx.hook_ctx.launch.is_distributed_ready()

    def get_rank(self, group=None):
        self.ctx.hook_ctx.launch.get_rank(group=group)
    
    def get_world_size(self, group=None):
        self.ctx.hook_ctx.launch.get_world_size(group=group)

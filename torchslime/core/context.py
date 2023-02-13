from torchslime.util import Base, NOTHING, BaseList, Nothing, TorchComm
from torch.nn import Module
from torch import device, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchslime.util.type import NUMBER
from typing import Any, Sequence, Union, Dict, Tuple, Callable, Type
from torchslime.log import logger
from abc import abstractmethod


class BaseContext(Base):
    """
    Base Context in the whole life time.
    """

    def __init__(self):
        super().__init__()
        
        """
        context attribute placeholders(for code hints)
        """
        # device for pytorch
        self.device: Union[str, device] = NOTHING
        # model
        self.model: Module = NOTHING
        # proxy status(train, eval, etc.)
        from torchslime.core.status import Status
        self.status: Status = NOTHING
        # the current dataset for running
        self.dataset: DataLoader = NOTHING
        # run context
        self.run: RunContext = RunContext()
        # information in one epoch
        self.epoch: EpochContext = EpochContext()
        # information in one step
        self.step: StepContext = StepContext()
        # handler context
        self.handler: Union[HandlerContext, DistributedHandlerContext] = \
            DistributedHandlerContext() if self.is_distributed_context() is True else HandlerContext()
        # custom context
        self.custom: CustomContext = CustomContext()
        # inner context
        self.inner: InnerContext = InnerContext()
        # build context
        self.build: BuildContext = BuildContext()

    def ctx_check(self, items: Union[str, Sequence[str]], silent: bool = True):
        # check single item
        def _check(_item):
            _result = super(BaseContext, self).check(_item)
            if _result is False:
                msg = 'Context check failed: got NOTHING with key \'%s\'.' % _item
                if silent is True:
                    logger.debug(msg)
                else:
                    logger.warn(msg)
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
    
    def is_distributed_context(self) -> bool:
        """
        Whether distributed features are used in TorchSlime.
        """
        return False


class TempContext(Base):
    """Temp context that defines a initialize method to quickly reset the context.

    Args:
        Base (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        # initialize
        self.initialize()
    
    @abstractmethod
    def initialize(self):
        pass


class StepContext(TempContext):

    def __init__(self):
        super().__init__()
    
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


class EpochContext(TempContext):

    def __init__(self):
        super().__init__()

    def initialize(self):
        """
        epoch context attribute placeholders(for code hints)
        """
        # total epochs for training
        self.total: int = NOTHING
        # the current epoch
        self.current: int = NOTHING
        # average train metrics in one epoch
        self.train_metrics: Dict = NOTHING
        # average eval metrics in one epoch
        self.eval_metrics: Dict = NOTHING
        # average train loss value(s) in one epoch
        self.train_loss_value: Union[float, Dict[Any, float], Nothing] = NOTHING
        # average eval loss value(s) in one epoch
        self.eval_loss_value: Union[float, Dict[Any, float], Nothing] = NOTHING


class RunContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        # handler containers that define the process of training, evaluating and predicting.
        from torchslime.core.handler import HandlerContainer
        self.train: HandlerContainer = NOTHING
        self.eval: HandlerContainer = NOTHING
        self.predict: HandlerContainer = NOTHING
        
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
        from torchslime.data import DataProvider
        self.train_provider: DataProvider = NOTHING
        self.eval_provider: DataProvider = NOTHING
        # data parser
        from torchslime.data import DataParser, IndexParser
        # the data parser should be set to IndexParser as default
        self.data_parser: DataParser = IndexParser()
        # run callback executor
        from torchslime.callback import CallbackContainer
        self.callbacks: CallbackContainer = NOTHING
        # metric container
        from torchslime.metric import MetricContainer
        self.metrics: MetricContainer = NOTHING
        # loss wrapper
        from torchslime.metric import LossWrapper
        self.loss_wrapper: Type[LossWrapper] = LossWrapper
        # loss reduction func
        from torchslime.metric import LossReductionFactory
        self.loss_reduction: Callable[[BaseContext], Tensor] = LossReductionFactory.get('mean')


class GlobalContext(TempContext):
    # TODO: rename

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        pass


class HandlerContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        from torchslime.core import handler
        # handler class
        self.Container = handler.HandlerContainer
        self.EpochIteration = handler.EpochIterationHandler
        self.Iteration = handler.IterationHandler
        self.Handler = handler.Handler
        self.Forward = handler.ForwardHandler
        self.Loss = handler.LossHandler
        self.Backward = handler.BackwardHandler
        self.Optimizer = handler.OptimizerHandler
        self.Metrics = handler.MetricsHandler
        self.Average = handler.AverageHandler
        self.Display = handler.DisplayHandler
        self.Dataset = handler.DatasetHandler
        self.Status = handler.StatusHandler
        self.LRDecay = handler.LRDecayHandler
        self.Callback = handler.CallbackHandler


class DistributedHandlerContext(HandlerContext):

    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

        from torchslime.core import handler
        self.GatherAverage = handler.GatherAverageHandler
        self.DistributedDisplay = handler.DistributedDisplayHandler
        self.DistributedEpochIteration = handler.DistributedEpochIterationHandler
        self.DistributedIteration = handler.DistributedIterationHandler
        self.DistributedContainer = handler.DistributedHandlerContainer


class CustomContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        self.__dict__.clear()
        logger.debug('Custom context has been initialized.')


class InnerContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        self.__dict__.clear()
        logger.debug('Inner context has been initialized.')


class BuildContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        self.valid_mode = 'epoch'
        self.train_mode = 'epoch'
        self.lr_decay_mode = 'epoch'
        
        # plugins
        from torchslime.core.plugin import PluginContainer
        self.plugins: PluginContainer = PluginContainer()


class DistributedConfigContext(TempContext):

    def __init__(self):
        super().__init__()
    
    def initialize(self):
        self.exec_ranks: BaseList = BaseList(0)
        self.torch_comm: TorchComm = TorchComm()

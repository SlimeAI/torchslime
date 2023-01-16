from typing import Any, Dict, Optional, Union, TypeVar
from torchslime.data import ConstantProvider, DataParser, DataProvider, IndexParser
from torchslime.metric import M_SEQ, MetricContainer
from torchslime.callback import C_SEQ, CallbackContainer, DistributedCallbackContainer
from torchslime.util import NOTHING, get_device, type_cast, MethodChaining, InvocationDebug, logger, \
    is_nothing, count_params, BaseList
from torchslime.util.type import NUMBER, INT_SEQ_N
from torchslime.core.context import Context
from torch.utils.data import DataLoader
from torch.optim import Optimizer


T = TypeVar('T', bound='Proxy')
DATASET = Union[DataLoader, DataProvider]


class Proxy(Context):

    def __init__(self, model, device=None):
        # init context
        super().__init__()
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)

    @InvocationDebug('Proxy.Train')
    def train(
        self,
        train_dataset: DATASET,
        total_epochs: int = 1,
        eval_dataset: DATASET = NOTHING,
        callbacks: C_SEQ = NOTHING,
        grad_acc: int = 1,
        log_option = None  # TODO: log system design
    ):
        self.compile_total_epochs(total_epochs)
        self.compile_callbacks(callbacks)
        self.compile_dataset(train_dataset, 'train')
        self.compile_dataset(eval_dataset, 'eval')
        self.compile_grad_acc(grad_acc)
        logger.info('Using device {0} to train.'.format(str(self.device)))
        self.run.train(self)

    @InvocationDebug('Proxy.Predict')
    def predict(
        self,
        dataset: DATASET,
        callbacks: C_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self.compile_callbacks(callbacks)
        self.compile_dataset(dataset, 'eval')
        logger.info('Using device {0} to predict.'.format(str(self.device)))
        self.run.predict(self)

    @InvocationDebug('Proxy.Eval')
    def eval(
        self,
        dataset: DATASET,
        callbacks: C_SEQ = NOTHING,
        log_option = None  # TODO: log system design
    ):
        self.compile_callbacks(callbacks)
        self.compile_dataset(dataset, 'eval')
        logger.info('Using device {0} to eval.'.format(str(self.device)))
        self.run.eval(self)

    @InvocationDebug('Proxy.Summary')
    def summary(self):
        pass

    @InvocationDebug('Proxy.CountParams')
    def count_params(self, format: str = None, decimal: int = 2, log: bool = True):
        result = count_params(self.model, format, decimal)
        if log is True:
            logger.info('Model parameters: {0}'.format(result))
        return result

    @InvocationDebug('Proxy.Compile')
    @MethodChaining
    def compile(
        self,
        loss = None,
        metrics: M_SEQ = None,
        optimizer: Union[str, Optimizer] = None,
        lr: NUMBER = None,
        lr_decay: Any = None,
        optimizer_options: Optional[Dict] = None,
        lr_decay_options: Optional[Dict] = None,
        data_parser: Optional[DataParser] = None
    ) -> T:
        self.compile_loss(loss)
        self.compile_metrics(metrics)
        self.compile_data_parser(data_parser)
        self.compile_optimizer(optimizer, lr, optimizer_options)
        self.compile_lr_decay(lr_decay, lr_decay_options)

    @InvocationDebug('Proxy.ProcessBuilder')
    @MethodChaining
    def build_all(self) -> T:
        self.build_train().build_eval().build_predict()

    @InvocationDebug('Proxy.TrainBuilder')
    @MethodChaining
    def build_train(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build training process using handlers
        self.run.train = handler.Container([
            # begin callback
            handler.Callback('begin'),
            # epoch iter
            handler.EpochIteration([
                # epoch begin callback
                handler.Callback('epoch_begin'),
                # set status to 'train'
                handler.Status('train'),
                # get dataset
                handler.Dataset(),
                # clear average metrics
                handler.Average('clear'),
                # dataset iter
                handler.Iteration([
                    # step begin callback
                    handler.Callback('step_begin'),
                    # forward
                    handler.Forward(),
                    # compute loss
                    handler.Loss(),
                    # backward and optimizer step
                    handler.Optimizer([
                        handler.Backward()
                    ]),
                    # compute metrics
                    handler.Metrics(),
                    # compute average metrics
                    handler.Average('avg'),
                    # display in console or in log files
                    handler.Display(),
                    # step end callback
                    handler.Callback('step_end')
                ]),
                # apply learning rate decay
                handler.LRDecay(),
                # set status to 'val'
                handler.Status('val'),
                # get dataset
                handler.Dataset(),
                # clear average metrics
                handler.Average('clear'),
                # dataset iter
                handler.Iteration([
                    # forward
                    handler.Forward(),
                    # compute loss
                    handler.Loss(),
                    # metrics
                    handler.Metrics(),
                    # compute average metrics
                    handler.Average('avg'),
                    # display in console or in log files
                    handler.Display()
                ]),
                # epoch end callback
                handler.Callback('epoch_end')
            ]),
            # end callback
            handler.Callback('end')
        ])

    @InvocationDebug('Proxy.PredictBuilder')
    @MethodChaining
    def build_predict(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build predicting process using handlers
        self.run.predict = handler.Container([
            # begin callback
            handler.Callback('begin'),
            # set status to 'predict'
            handler.Status('predict'),
            # get dataset
            handler.Dataset(),
            # dataset iteration
            handler.Iteration([
                # step begin callback
                handler.Callback('step_begin'),
                # forward
                handler.Forward(),
                # display
                handler.Display(),
                # step end callback
                handler.Callback('step_end')
            ]),
            # end callback
            handler.Callback('end')
        ])

    @InvocationDebug('Proxy.EvalBuilder')
    @MethodChaining
    def build_eval(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build evaluating process using handlers
        self.run.eval = handler.Container([
            # begin callback
            handler.Callback('begin'),
            # set status to 'eval'
            handler.Status('eval'),
            # get dataset
            handler.Dataset(),
            # clear average metrics
            handler.Average('clear'),
            # dataset iteration
            handler.Iteration([
                # step begin callback
                handler.Callback('step_begin'),
                # forward
                handler.Forward(),
                # compute loss
                handler.Loss(),
                # compute metrics
                handler.Metrics(),
                # compute average metrics
                handler.Average('avg'),
                # display
                handler.Display(),
                # step end callback
                handler.Callback('step_end')
            ]),
            # end callback
            handler.Callback('end')
        ])

    @InvocationDebug('Proxy.compile_loss')
    def compile_loss(self, loss):
        if loss is not None:
            self.run.loss = loss

    @InvocationDebug('Proxy.compile_metrics')
    def compile_metrics(self, metrics):
        if metrics is not None:
            self.run.metrics = MetricContainer(metrics) if is_nothing(metrics) is False else NOTHING

    @InvocationDebug('Proxy.compile_data_parser')
    def compile_data_parser(self, data_parser):
        if data_parser is not None:
            self.run.data_parser = data_parser if is_nothing(data_parser) is False else IndexParser()

    @InvocationDebug('Proxy.compile_callbacks')
    def compile_callbacks(self, callbacks):
        if callbacks is not None:
            self.run.callbacks = CallbackContainer(callbacks) if is_nothing(callbacks) is False else NOTHING

    @InvocationDebug('Proxy.compile_optimizer')
    def compile_optimizer(self, optimizer, lr, optimizer_options):
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.run.optimizer = optimizer

    @InvocationDebug('Proxy.compile_lr_decay')
    def compile_lr_decay(self, lr_decay, lr_decay_options):
        if lr_decay is not None:
            if isinstance(lr_decay, str) is False:
                self.run.lr_decay = lr_decay

    @InvocationDebug('Proxy.compile_total_epochs')
    def compile_total_epochs(self, total_epochs):
        self.epoch.total = total_epochs if isinstance(total_epochs, int) else NOTHING

    @InvocationDebug('Proxy.compile_dataset')
    def compile_dataset(self, dataset, mode: str):
        if dataset is not None:
            if is_nothing(dataset):
                dataset = NOTHING
            else:
                dataset = dataset if isinstance(dataset, DataProvider) else ConstantProvider(dataset)

            if mode == 'train':
                self.run.train_provider = dataset
            elif mode == 'eval':
                self.run.eval_provider = dataset
            else:
                logger.warn('compile_dataset mode not supported.')

    @InvocationDebug('Proxy.compile_grad_acc')
    def compile_grad_acc(self, grad_acc: int):
        if grad_acc is not None:
            self.run.grad_acc = grad_acc


DIST_T = TypeVar('DIST_T', bound='DistributedProxy')


class DistributedProxy(Proxy):

    def __init__(self, model, device=None, exec_ranks: INT_SEQ_N = 0):
        super().__init__(model, device)
        self.set_distributed_context()
        self.distributed.exec_ranks = BaseList.create_nothing(exec_ranks)

    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        # TODO: duck typing check?
        pass

    def set_distributed_context(self):
        from torchslime.core.context import DistributedContext, DistributedHandlerContext
        self.distributed: DistributedContext = DistributedContext()
        self.handler: DistributedHandlerContext = DistributedHandlerContext()

    def check_distributed_ready(self):
        """
        Check whether the torch distributed settings are ready.
        """
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()

    def get_rank(self, group=None):
        import torch.distributed as dist
        return dist.get_rank(group=group)

    def get_world_size(self, group=None):
        import torch.distributed as dist
        return dist.get_world_size(group=group)

    @InvocationDebug('DistributedProxy.TrainBuilder')
    @MethodChaining
    def build_train(self) -> DIST_T:
        pass

    @InvocationDebug('DistributedProxy.PredictBuilder')
    @MethodChaining
    def build_predict(self) -> DIST_T:
        pass

    @InvocationDebug('DistributedProxy.EvalBuilder')
    @MethodChaining
    def build_eval(self) -> DIST_T:
        pass

    @InvocationDebug('DistributedProxy.compile_callbacks')
    def compile_callbacks(self, callbacks):
        if callbacks is not None:
            self.run.callbacks = DistributedCallbackContainer(callbacks) if is_nothing(callbacks) is False else NOTHING
            self.run.callbacks.set_exec_ranks(self.distributed.exec_ranks)

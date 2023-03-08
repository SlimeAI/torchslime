from typing import Any, Dict, Optional, Union, TypeVar, Callable
from torchslime.data import ConstantProvider, DataParser, DataProvider, IndexParser
from torchslime.metric import M_SEQ, MetricContainer, LossReductionFactory
from torchslime.callback import C_SEQ, CallbackContainer, DistributedCallbackContainer
from torchslime.util import NOTHING, get_device, type_cast, MethodChaining, InvocationDebug, logger, \
    is_nothing, count_params, BaseList
from torchslime.util.type import NUMBER, INT_SEQ_N
from torchslime.core.context import BaseContext, DistributedConfigContext
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor


T = TypeVar('T', bound='Context')
DATASET = Union[DataLoader, DataProvider]


class Context(BaseContext):

    def __init__(self, model, device=None):
        # init context
        super().__init__()
        # set device
        self.device = device if device is not None else get_device(model)
        # set model and apply type cast
        self.model = type_cast(model, self.device)

    @InvocationDebug('Context.Train')
    def train(
        self,
        train_dataset: DATASET,
        total_epochs: int = 1,
        val_dataset: DATASET = NOTHING,
        callbacks: C_SEQ = NOTHING,
        grad_acc: int = 1,
        log_option = None  # TODO: log system design
    ):
        self.compile_total_epochs(total_epochs)
        self.compile_callbacks(callbacks)
        self.compile_dataset(train_dataset, 'train')
        self.compile_dataset(val_dataset, 'eval')
        self.compile_grad_acc(grad_acc)
        logger.info('Using device {0} to train.'.format(str(self.device)))
        self.run.train(self)

    @InvocationDebug('Context.Predict')
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

    @InvocationDebug('Context.Eval')
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

    @InvocationDebug('Context.Summary')
    def summary(self):
        pass

    @InvocationDebug('Context.CountParams')
    def count_params(self, format: str = None, decimal: int = 2, log: bool = True):
        result = count_params(self.model, format, decimal)
        if log is True:
            logger.info('Model parameters: {0}'.format(result))
        return result

    @InvocationDebug('Context.Compile')
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
    ) -> T:
        self.compile_loss_func(loss_func)
        self.compile_loss_reduction(loss_reduction)
        self.compile_metrics(metrics)
        self.compile_data_parser(data_parser)
        self.compile_optimizer(optimizer, lr, optimizer_options)
        self.compile_lr_decay(lr_decay, lr_decay_options)

    @InvocationDebug('Context.ProcessBuilder')
    @MethodChaining
    def build_all(self) -> T:
        self.build_train().build_eval().build_predict()

    @InvocationDebug('Context.TrainBuilder')
    @MethodChaining
    def build_train(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build training process using handlers
        self.run.train = handler.Container([
            # begin callback
            handler.Callback('begin', _id='train_begin_callback'),
            # epoch iter
            handler.EpochIteration([
                # epoch begin callback
                handler.Callback('epoch_begin', _id='train_epoch_begin_callback'),
                # set status to 'train'
                handler.Status('train', _id='train_status_train'),
                # get dataset
                handler.Dataset(_id='train_dataset_train'),
                # init average setting
                handler.AverageInit(_id='train_average_init_train'),
                # dataset iter
                handler.Iteration([
                    # step begin callback
                    handler.Callback('step_begin', _id='train_step_begin_callback'),
                    # forward
                    handler.Forward(_id='train_forward_train'),
                    # compute loss
                    handler.Loss(_id='train_loss_train'),
                    # backward and optimizer step
                    handler.Optimizer([
                        handler.Backward(_id='train_backward')
                    ], _id='train_optimizer'),
                    # compute metrics
                    handler.Metrics(_id='train_metrics_train'),
                    # compute average loss value and metrics
                    handler.Average(_id='train_average_train'),
                    # display in console or in log files
                    handler.Display(_id='train_display_train'),
                    # step end callback
                    handler.Callback('step_end', _id='train_step_end_callback')
                ], _id='train_iteration_train'),
                # apply learning rate decay
                handler.LRDecay(_id='train_lr_decay'),
                # set status to 'val'
                handler.Status('val', _id='train_status_val'),
                # get dataset
                handler.Dataset(_id='train_dataset_val'),
                # init average setting
                handler.AverageInit(_id='train_average_init_val'),
                # dataset iter
                handler.Iteration([
                    # forward
                    handler.Forward(_id='train_forward_val'),
                    # compute loss
                    handler.Loss(_id='train_loss_val'),
                    # metrics
                    handler.Metrics(_id='train_metrics_val'),
                    # compute average loss value and metrics
                    handler.Average(_id='train_average_val'),
                    # display in console or in log files
                    handler.Display(_id='train_display_val')
                ], _id='train_iteration_val'),
                # epoch end callback
                handler.Callback('epoch_end', _id='train_epoch_end_callback')
            ], _id='train_epoch_iteration'),
            # end callback
            handler.Callback('end', _id='train_end_callback')
        ], _id='train_container')

    @InvocationDebug('Context.PredictBuilder')
    @MethodChaining
    def build_predict(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build predicting process using handlers
        self.run.predict = handler.Container([
            # begin callback
            handler.Callback('begin', _id='predict_begin_callback'),
            # set status to 'predict'
            handler.Status('predict', _id='predict_status'),
            # get dataset
            handler.Dataset(_id='predict_dataset'),
            # dataset iteration
            handler.Iteration([
                # step begin callback
                handler.Callback('step_begin', _id='predict_step_begin_callback'),
                # forward
                handler.Forward(_id='predict_forward'),
                # display
                handler.Display(_id='predict_display'),
                # step end callback
                handler.Callback('step_end', _id='predict_step_end_callback')
            ], _id='predict_iteration'),
            # end callback
            handler.Callback('end', _id='predict_end_callback')
        ], _id='predict_container')

    @InvocationDebug('Context.EvalBuilder')
    @MethodChaining
    def build_eval(self) -> T:
        # get handler classes from context
        handler = self.handler
        # build evaluating process using handlers
        self.run.eval = handler.Container([
            # begin callback
            handler.Callback('begin', _id='eval_begin_callback'),
            # set status to 'eval'
            handler.Status('eval', _id='eval_status'),
            # get dataset
            handler.Dataset(_id='eval_dataset'),
            # clear average metrics
            handler.AverageInit(_id='eval_average_init'),
            # dataset iteration
            handler.Iteration([
                # step begin callback
                handler.Callback('step_begin', _id='eval_step_begin_callback'),
                # forward
                handler.Forward(_id='eval_forward'),
                # compute loss
                handler.Loss(_id='eval_loss'),
                # compute metrics
                handler.Metrics(_id='eval_metrics'),
                # compute average metrics
                handler.Average(_id='eval_average'),
                # display
                handler.Display(_id='eval_display'),
                # step end callback
                handler.Callback('step_end', _id='eval_step_end_callback')
            ], _id='eval_iteration'),
            # end callback
            handler.Callback('end', _id='eval_end_callback')
        ], _id='eval_container')

    @InvocationDebug('Context.compile_loss_func')
    def compile_loss_func(self, loss_func):
        if loss_func is not None:
            self.run.loss_func = loss_func

    @InvocationDebug('Context.compile_loss_reduction')
    def compile_loss_reduction(self, loss_reduction):
        if loss_reduction is not None:
            self.run.loss_reduction = LossReductionFactory.get(loss_reduction)

    @InvocationDebug('Context.compile_metrics')
    def compile_metrics(self, metrics):
        if metrics is not None:
            self.run.metrics = MetricContainer(metrics) if is_nothing(metrics) is False else NOTHING

    @InvocationDebug('Context.compile_data_parser')
    def compile_data_parser(self, data_parser):
        if data_parser is not None:
            self.run.data_parser = data_parser if is_nothing(data_parser) is False else IndexParser()

    @InvocationDebug('Context.compile_callbacks')
    def compile_callbacks(self, callbacks):
        if callbacks is not None:
            self.run.callbacks = CallbackContainer(callbacks) if is_nothing(callbacks) is False else NOTHING

    @InvocationDebug('Context.compile_optimizer')
    def compile_optimizer(self, optimizer, lr, optimizer_options):
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.run.optimizer = optimizer

    @InvocationDebug('Context.compile_lr_decay')
    def compile_lr_decay(self, lr_decay, lr_decay_options):
        if lr_decay is not None:
            if isinstance(lr_decay, str) is False:
                self.run.lr_decay = lr_decay

    @InvocationDebug('Context.compile_total_epochs')
    def compile_total_epochs(self, total_epochs):
        self.epoch.total = total_epochs if isinstance(total_epochs, int) else NOTHING

    @InvocationDebug('Context.compile_dataset')
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

    @InvocationDebug('Context.compile_grad_acc')
    def compile_grad_acc(self, grad_acc: int):
        if grad_acc is not None:
            self.run.grad_acc = grad_acc


DIST_T = TypeVar('DIST_T', bound='DistributedContext')


class DistributedContext(Context):

    def __init__(self, model, device=None, exec_ranks: INT_SEQ_N = 0):
        super().__init__(model, device)
        # distributed config context.
        self.distributed: DistributedConfigContext = DistributedConfigContext()
        self.distributed.exec_ranks = BaseList.create(exec_ranks)

    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        self.distributed.exec_ranks = BaseList.create(exec_ranks)
        self.run.callbacks.set_exec_ranks(exec_ranks)
        self.run.train.set_exec_ranks(exec_ranks)
        self.run.eval.set_exec_ranks(exec_ranks)
        self.run.predict.set_exec_ranks(exec_ranks)

    def is_distributed(self) -> bool:
        """
        Distributed Context belongs to distributed context.
        """
        return True

    def is_distributed_ready(self):
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

    @InvocationDebug('DistributedContext.TrainBuilder')
    @MethodChaining
    def build_train(self) -> DIST_T:
        # get handler classes from context
        handler = self.handler
        # build distributed training process using handlers
        self.run.train = handler.DistributedContainer([
            # begin callback
            handler.Callback('begin', _id='train_begin_callback'),
            # epoch iter
            handler.DistributedEpochIteration([
                # epoch begin callback
                handler.Callback('epoch_begin', _id='train_epoch_begin_callback'),
                # set status to 'train'
                handler.Status('train', _id='train_status_train'),
                # get dataset
                handler.Dataset(_id='train_dataset_train'),
                # init average setting
                handler.AverageInit(_id='train_average_init_train'),
                # dataset iter
                handler.DistributedIteration([
                    # step begin callback
                    handler.Callback('step_begin', _id='train_step_begin_callback'),
                    # forward
                    handler.Forward(_id='train_forward_train'),
                    # compute loss
                    handler.Loss(_id='train_loss_train'),
                    # backward and optimizer step
                    handler.Optimizer([
                        handler.Backward(_id='train_backward')
                    ], _id='train_optimizer'),
                    # compute metrics
                    handler.Metrics(_id='train_metrics_train'),
                    # gather loss and metrics
                    handler.GatherAverage(_id='train_gather_average_train'),
                    # compute average loss value and metrics
                    handler.Average(_id='train_average_train'),
                    # display in console or in log files
                    handler.DistributedDisplay(_id='train_display_train'),
                    # step end callback
                    handler.Callback('step_end', _id='train_step_end_callback')
                ], _id='train_iteration_train'),
                # apply learning rate decay
                handler.LRDecay(_id='train_lr_decay'),
                # set status to 'val'
                handler.Status('val', _id='train_status_val'),
                # get dataset
                handler.Dataset(_id='train_dataset_val'),
                # init average setting
                handler.AverageInit(_id='train_average_init_val'),
                # dataset iter
                handler.Iteration([
                    # forward
                    handler.Forward(_id='train_forward_val'),
                    # compute loss
                    handler.Loss(_id='train_loss_val'),
                    # metrics
                    handler.Metrics(_id='train_metrics_val'),
                    # gather loss and metrics
                    handler.GatherAverage(_id='train_gather_average_val'),
                    # compute average loss value and metrics
                    handler.Average(_id='train_average_val'),
                    # display in console or in log files
                    handler.DistributedDisplay(_id='train_display_val')
                ], _id='train_iteration_val'),
                # epoch end callback
                handler.Callback('epoch_end', _id='train_epoch_end_callback')
            ], _id='train_epoch_iteration'),
            # end callback
            handler.Callback('end', _id='train_end_callback')
        ], _id='train_container')
        # set exec ranks
        self.run.train.set_exec_ranks(self.distributed.exec_ranks)

    @InvocationDebug('DistributedContext.PredictBuilder')
    @MethodChaining
    def build_predict(self) -> DIST_T:
        # get handler classes from context
        handler = self.handler
        # build predicting process using handlers
        self.run.predict = handler.DistributedContainer([
            # begin callback
            handler.Callback('begin', _id='predict_begin_callback'),
            # set status to 'predict'
            handler.Status('predict', _id='predict_status'),
            # get dataset
            handler.Dataset(_id='predict_dataset'),
            # dataset iteration
            handler.DistributedIteration([
                # step begin callback
                handler.Callback('step_begin', _id='predict_step_begin_callback'),
                # forward
                handler.Forward(_id='predict_forward'),
                # display
                handler.DistributedDisplay(_id='predict_display'),
                # step end callback
                handler.Callback('step_end', _id='predict_step_end_callback')
            ], _id='predict_iteration'),
            # end callback
            handler.Callback('end', _id='predict_end_callback')
        ], _id='predict_container')
        self.run.predict.set_exec_ranks(self.distributed.exec_ranks)

    @InvocationDebug('DistributedContext.EvalBuilder')
    @MethodChaining
    def build_eval(self) -> DIST_T:
        # get handler classes from context
        handler = self.handler
        # build evaluating process using handlers
        self.run.eval = handler.DistributedContainer([
            # begin callback
            handler.Callback('begin', _id='eval_begin_callback'),
            # set status to 'eval'
            handler.Status('eval', _id='eval_status'),
            # get dataset
            handler.Dataset(_id='eval_dataset'),
            # clear average metrics
            handler.AverageInit(_id='eval_average_init'),
            # dataset iteration
            handler.DistributedIteration([
                # step begin callback
                handler.Callback('step_begin', _id='eval_step_begin_callback'),
                # forward
                handler.Forward(_id='eval_forward'),
                # compute loss
                handler.Loss(_id='eval_loss'),
                # compute metrics
                handler.Metrics(_id='eval_metrics'),
                # gather loss and metrics
                handler.GatherAverage(_id='eval_gather_average'),
                # compute average metrics
                handler.Average(_id='eval_average'),
                # display
                handler.DistributedDisplay(_id='eval_display'),
                # step end callback
                handler.Callback('step_end', _id='eval_step_end_callback')
            ], _id='eval_iteration'),
            # end callback
            handler.Callback('end', _id='eval_end_callback')
        ], _id='eval_container')
        self.run.eval.set_exec_ranks(self.distributed.exec_ranks)

    @InvocationDebug('DistributedContext.compile_callbacks')
    def compile_callbacks(self, callbacks):
        if callbacks is not None:
            self.run.callbacks = DistributedCallbackContainer(callbacks) if is_nothing(callbacks) is False else NOTHING
            self.run.callbacks.set_exec_ranks(self.distributed.exec_ranks)

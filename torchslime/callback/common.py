import os

from torchslime.util import is_nothing
from torchslime.callback import Callback, DistributedCallbackWrapper
from torchslime.core.context import Context
from torchslime.log.directory import get_checkpoint_path, join_path, get_metric_path, safe_makedirs
from torchslime.log import logger
from torchslime.util.type import INT_SEQ_N
import torch
from typing import Sequence, Union, Callable
import json

EPOCH_SEQ = Union[int, Sequence[int]]


class SaveCheckpoint(Callback):

    def __init__(
        self,
        save_per: EPOCH_SEQ,
        checkpoint_name: Union[str, Callable[[Context], str]]=None,
        save_model: bool = True,
        save_optimizer: bool = False,
        save_epoch: bool = False
    ):
        super().__init__()
        self.checkpoint_path = get_checkpoint_path()
        safe_makedirs(self.checkpoint_path)
        self.save_per = save_per

        if isinstance(checkpoint_name, str):
            logger.warn('The checkpoint name is set to a constant string, and the previous checkpoint will be overwritten when a new checkpoint is saved.')
        if (checkpoint_name is None or isinstance(checkpoint_name, str) or callable(checkpoint_name)) is False:
            checkpoint_name = None
            logger.error('You have set an unsupported checkpoint name. The checkpoint name should be a string, a function or None. Now the checkpoint name is set to default(None).')

        self.checkpoint_name = checkpoint_name
        self.save_options = {
            'model': save_model,
            'optimizer': save_optimizer,
            'epoch': save_epoch
        }.items()
        self.save_options = list(map(lambda item: item[0], filter(lambda item: item[1] is True, self.save_options)))
        assert len(self.save_options) > 0, 'You should choose at least one item to be saved when using the "SaveCheckpoint" Callback.'
    
    def epoch_end(self, ctx: Context):
        if (isinstance(self.save_per, (list, tuple)) and (ctx.epoch.current + 1) in self.save_per)\
            or (ctx.epoch.current + 1) % self.save_per == 0:
            if len(self.save_options) > 1:
                item = self.save_dict(ctx, self.save_options)
            else:
                item = self.save_single(ctx, self.save_options[0])
            
            if isinstance(self.checkpoint_name, str):
                checkpoint_name = self.checkpoint_name
            elif callable(self.checkpoint_name):
                checkpoint_name = self.checkpoint_name(ctx)
            else:
                checkpoint_name = 'checkpoint_{0}.pth'.format(ctx.epoch.current + 1)
            torch.save(item, join_path(self.checkpoint_path, checkpoint_name))

    def save_dict(self, ctx: Context, save_options):
        item = {}
        for key in save_options:
            if key == 'model':
                item['model'] = self.save_single(ctx, key)
            elif key == 'optimizer':
                item['optimizer'] = self.save_single(ctx, key)
            elif key == 'epoch':
                item['epoch'] = self.save_single(ctx, key)
        return item
    
    def save_single(self, ctx: Context, key):
        if key == 'model':
            return ctx.model.state_dict()
        elif key == 'optimizer':
            return ctx.run.optimizer.state_dict()
        elif key == 'epoch':
            return ctx.epoch.current + 1


class DistributedSaveCheckpoint(DistributedCallbackWrapper):

    def __init__(
        self,
        save_per: EPOCH_SEQ,
        checkpoint_name: Union[str, Callable[[Context], str]]=None,
        save_model: bool = True,
        save_optimizer: bool = False,
        save_epoch: bool = False,
        exec_ranks: INT_SEQ_N = None
    ):
        wrapped_callback = SaveCheckpoint(
            save_per,
            checkpoint_name,
            save_model,
            save_optimizer,
            save_epoch
        )
        super().__init__(wrapped_callback, exec_ranks)


class SaveMetrics(Callback):

    def __init__(
        self,
        save_train: bool = True,
        save_eval: bool = True,
        save_per: EPOCH_SEQ = 1
    ):
        super().__init__()
        self.metric_path = get_metric_path()
        self.save_per = save_per
        self.save_options = {
            'train': save_train,
            'eval': save_eval
        }.items()
        self.save_options = list(map(lambda item: item[0], filter(lambda item: item[1] is True, self.save_options)))
        assert len(self.save_options) > 0, 'You should choose at least one item to be saved when using the "SaveMetrics" Callback.'

    def epoch_end(self, ctx: Context):
        if (isinstance(self.save_per, (list, tuple)) and (ctx.epoch.current + 1) in self.save_per)\
            or (ctx.epoch.current + 1) % self.save_per == 0:
            list_len = self.append_list(self.parse(ctx, self.save_options))
            if list_len > ctx.epoch.current + 1:
                logger.warn('The length of metric list is greater than number of epochs that have been executed, possibly there are some other items included in the list.')

    def parse(self, ctx: Context, save_options):
        item = {}
        for key in save_options:
            if key == 'train':
                item.update(**ctx.epoch.train_metrics)
                if is_nothing(ctx.epoch.train_loss) is False:
                    item.update(loss=ctx.epoch.train_loss)
            elif key == 'eval':
                item.update(**ctx.epoch.eval_metrics)
                if is_nothing(ctx.epoch.eval_loss) is False:
                    item.update(val_loss=ctx.epoch.eval_loss)
        return item

    def append_list(self, item):
        if os.path.exists(self.metric_path):
            try:
                with open(self.metric_path, 'r') as f:
                    history = json.load(f)
            except Exception:
                history = []
        else:
            history = []
        history.append(item)
        with open(self.metric_path, 'w') as f:
            json.dump(history, f, indent=4)
        return len(history)


class DistributedSaveMetrics(DistributedCallbackWrapper):

    def __init__(
        self,
        save_train: bool = True,
        save_eval: bool = True,
        save_per: EPOCH_SEQ = 1,
        exec_ranks: INT_SEQ_N = None
    ):
        wrapped_callback = SaveMetrics(
            save_train,
            save_eval,
            save_per
        )
        super().__init__(wrapped_callback, exec_ranks)

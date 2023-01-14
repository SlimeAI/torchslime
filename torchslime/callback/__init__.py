from torchslime.core.context import Context
from torchslime.util import BaseList, is_nothing, NOTHING
from torchslime.util.type import INT_SEQ_N
from typing import Union, Sequence
import torch.distributed as dist
from log import logger


class Callback:
    """
    Callback for running operations(training, evaluation, prediction, etc.).
    """
    def __init__(self):
        super().__init__()

    def begin(self, ctx: Context):
        pass

    def end(self, ctx: Context):
        pass

    def step_begin(self, ctx: Context):
        pass
    
    def step_end(self, ctx: Context):
        pass

    def epoch_begin(self, ctx: Context):
        pass

    def epoch_end(self, ctx: Context):
        pass


# callback or sequence of callbacks
C_SEQ = Union[Callback, Sequence[Callback]]


class CallbackContainer(Callback, BaseList):
    """
    Maintaining a list that contains callbacks, combination mode.
    """

    def __init__(self, callbacks: C_SEQ = None):
        Callback.__init__(self)
        BaseList.__init__(self, callbacks)

    def begin(self, ctx: Context):
        for callback in self:
            callback.begin(ctx)
    
    def end(self, ctx: Context):
        for callback in self:
            callback.end(ctx)
    
    def step_begin(self, ctx: Context):
        for callback in self:
            callback.step_begin(ctx)
    
    def step_end(self, ctx: Context):
        for callback in self:
            callback.step_end(ctx)
    
    def epoch_begin(self, ctx: Context):
        for callback in self:
            callback.epoch_begin(ctx)
    
    def epoch_end(self, ctx: Context):
        for callback in self:
            callback.epoch_end(ctx)


class DistributedCallback(Callback):
    """
    Distributed callback.
    """

    def __init__(self, exec_ranks: INT_SEQ_N = None):
        super().__init__()
        self.exec_ranks = BaseList.create(exec_ranks)
    
    def set_exec_ranks(self, exec_ranks):
        if self.exec_ranks is None or exec_ranks is None:
            # the exec_ranks are changeable
            self.exec_ranks = BaseList.create(exec_ranks)


class DistributedCallbackWrapper(DistributedCallback):
    """
    Distributed Callback Wrapper that makes a normal callback distributed-enabled.
    """

    def __init__(self, wrapped_callback: Callback, exec_ranks: INT_SEQ_N = None):
        super().__init__(exec_ranks)
        self._wrapped_callback = wrapped_callback
    
    def begin(self, ctx: Context):
        self._wrapped_callback.begin(ctx)
    
    def end(self, ctx: Context):
        self._wrapped_callback.end(ctx)
    
    def step_begin(self, ctx: Context):
        self._wrapped_callback.step_begin(ctx)
    
    def step_end(self, ctx: Context):
        self._wrapped_callback.step_end(ctx)
    
    def epoch_begin(self, ctx: Context):
        self._wrapped_callback.epoch_begin(ctx)
    
    def epoch_end(self, ctx: Context):
        self._wrapped_callback.epoch_end(ctx)


class DistributedCallbackContainer(CallbackContainer):
    """
    Distributed callback container.
    """

    def __init__(self, callbacks: C_SEQ = None, default_exec_ranks: INT_SEQ_N = None):
        super().__init__(callbacks)
        # the distributed callback container is always executed.
        self.exec_ranks = NOTHING
        # exec ranks that are set to its sub-callbacks
        self.default_exec_ranks = BaseList.create(default_exec_ranks)

    def set_exec_ranks(self, exec_ranks):
        if self.default_exec_ranks is None or exec_ranks is None:
            # the default_exec_ranks are changeable
            self.default_exec_ranks = BaseList.create(exec_ranks)
        for callback in filter(lambda item: isinstance(item, DISTRIBUTED_CLASSES), self):
            callback.set_exec_ranks(self.default_exec_ranks)

    def begin(self, ctx: Context):
        for callback in filter(self._check_exec, self):
            callback.begin(ctx)
    
    def end(self, ctx: Context):
        for callback in filter(self._check_exec, self):
            callback.end(ctx)
    
    def step_begin(self, ctx: Context):
        for callback in filter(self._check_exec, self):
            callback.step_begin(ctx)
    
    def step_end(self, ctx: Context):
        for callback in filter(self._check_exec, self):
            callback.step_end(ctx)
    
    def epoch_begin(self, ctx: Context):
        for callback in filter(self._check_exec, self):
            callback.epoch_begin(ctx)
    
    def epoch_end(self, ctx: Context):
        for callback in filter(self._check_exec, self):
            callback.epoch_end(ctx)

    def _check_exec(self, callback: Callback):
        rank = dist.get_rank()
        # check exec condition
        distributed_execution = isinstance(callback, DISTRIBUTED_CLASSES) and \
            callback.exec_ranks is not None and \
            (is_nothing(callback.exec_ranks) or rank in callback.exec_ranks)

        non_distributed_execution = isinstance(callback, Callback) and \
            isinstance(callback, DISTRIBUTED_CLASSES) is False

        return distributed_execution or non_distributed_execution


DISTRIBUTED_CLASSES = (DistributedCallback, DistributedCallbackContainer)

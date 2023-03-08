from torchslime.core.context import BaseContext
from torchslime.util import BaseList, is_nothing, NOTHING, is_none_or_nothing
from torchslime.util.type import INT_SEQ_N
from typing import Union, Sequence
import torch.distributed as dist


class Callback:
    """
    Callback for running operations(training, evaluation, prediction, etc.).
    """
    def __init__(self):
        super().__init__()

    def begin(self, ctx: BaseContext):
        pass

    def end(self, ctx: BaseContext):
        pass

    def step_begin(self, ctx: BaseContext):
        pass
    
    def step_end(self, ctx: BaseContext):
        pass

    def epoch_begin(self, ctx: BaseContext):
        pass

    def epoch_end(self, ctx: BaseContext):
        pass

    def _exec_hook(self, _hook: str, ctx: BaseContext):
        """
        Used to dispatch callback hooks.
        """
        getattr(self, _hook)(ctx)
    
    def is_distributed(self) -> bool:
        return False


# callback or sequence of callbacks
C_SEQ = Union[Callback, Sequence[Callback]]


class CallbackContainer(Callback, BaseList):
    """
    Maintaining a list that contains callbacks, combination mode.
    """

    def __init__(self, callbacks: C_SEQ = None):
        Callback.__init__(self)
        BaseList.__init__(self, callbacks)

    def begin(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('begin', ctx)
    
    def end(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('end', ctx)
    
    def step_begin(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('step_begin', ctx)
    
    def step_end(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('step_end', ctx)
    
    def epoch_begin(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('epoch_begin', ctx)
    
    def epoch_end(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('epoch_end', ctx)


class DistributedCallback(Callback):
    """
    Distributed callback.
    """

    def __init__(self, exec_ranks: INT_SEQ_N = NOTHING):
        super().__init__()
        self.exec_ranks = BaseList.create(exec_ranks)
        self.exec_locked = (is_nothing(exec_ranks) is False)
    
    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        if self.exec_locked is False:
            # the exec_ranks are changeable
            self.exec_ranks = BaseList.create(exec_ranks)

    def _exec_hook(self, _hook: str, ctx: BaseContext):
        rank = ctx.get_rank()
        if is_none_or_nothing(self.exec_ranks) is False and \
            (self.exec_ranks is ... or rank in self.exec_ranks):
            super()._exec_hook(_hook, ctx)
    
    def is_distributed(self) -> bool:
        return True


class DistributedCallbackWrapper(DistributedCallback):
    """
    Distributed Callback Wrapper that makes a normal callback distributed-enabled.
    """

    def __init__(self, wrapped_callback: Callback, exec_ranks: INT_SEQ_N = NOTHING):
        super().__init__(exec_ranks)
        self._wrapped_callback = wrapped_callback
    
    def begin(self, ctx: BaseContext):
        self._wrapped_callback._exec_hook('begin', ctx)
    
    def end(self, ctx: BaseContext):
        self._wrapped_callback._exec_hook('end', ctx)
    
    def step_begin(self, ctx: BaseContext):
        self._wrapped_callback._exec_hook('step_begin', ctx)
    
    def step_end(self, ctx: BaseContext):
        self._wrapped_callback._exec_hook('step_end', ctx)
    
    def epoch_begin(self, ctx: BaseContext):
        self._wrapped_callback._exec_hook('epoch_begin', ctx)
    
    def epoch_end(self, ctx: BaseContext):
        self._wrapped_callback._exec_hook('epoch_end', ctx)


class DistributedCallbackContainer(CallbackContainer):
    """
    Distributed callback container.
    """

    def __init__(self, callbacks: C_SEQ = None):
        super().__init__(callbacks)
        # the distributed callback container is always executed.
        self.exec_ranks = ...
        # the exec_locked is set to True by default.
        self.exec_locked = True

    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        if self.exec_locked is False:
            # the exec_ranks are changeable
            self.exec_ranks = BaseList.create(exec_ranks)
        # set sub-callback exec_ranks
        for callback in filter(lambda item: item.is_distributed(), self):
            callback.set_exec_ranks(exec_ranks)

    def begin(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('begin', ctx)
    
    def end(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('end', ctx)
    
    def step_begin(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('step_begin', ctx)
    
    def step_end(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('step_end', ctx)
    
    def epoch_begin(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('epoch_begin', ctx)
    
    def epoch_end(self, ctx: BaseContext):
        for callback in self:
            callback._exec_hook('epoch_end', ctx)

    def _exec_hook(self, _hook: str, ctx: BaseContext):
        rank = ctx.get_rank()
        if is_none_or_nothing(self.exec_ranks) is False and \
            (self.exec_ranks is ... or rank in self.exec_ranks):
            super()._exec_hook(_hook, ctx)
    
    def is_distributed(self) -> bool:
        return True


class DistributedCallbackContainerWrapper(DistributedCallbackContainer):

    def __init__(self, wrapped_callback_container: CallbackContainer):
        super().__init__(None)
        self._wrapped_callback_container = wrapped_callback_container
        self.set_list(wrapped_callback_container.get_list())

    def begin(self, ctx: BaseContext):
        self._wrapped_callback_container._exec_hook('begin', ctx)
    
    def end(self, ctx: BaseContext):
        self._wrapped_callback_container._exec_hook('end', ctx)
    
    def step_begin(self, ctx: BaseContext):
        self._wrapped_callback_container._exec_hook('step_begin', ctx)
    
    def step_end(self, ctx: BaseContext):
        self._wrapped_callback_container._exec_hook('step_end', ctx)
    
    def epoch_begin(self, ctx: BaseContext):
        self._wrapped_callback_container._exec_hook('epoch_begin', ctx)
    
    def epoch_end(self, ctx: BaseContext):
        self._wrapped_callback_container._exec_hook('epoch_end', ctx)

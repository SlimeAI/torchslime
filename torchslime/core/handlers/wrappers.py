from torchslime.core.context.base import BaseContext
from torchslime.core.handlers.common import BaseContext
from torchslime.core.handlers.wrappers import BaseContext
from . import Handler
from torchslime.core.context import BaseContext
from torchslime.core.hooks.state import StateHook, state_registry
from torchslime.utils.bases import BaseList, Nothing, NOTHING, is_none_or_nothing
from torchslime.utils.typing import (
    Union,
    Iterable,
    SupportsIndex,
    List,
    Callable
)
from torchslime.utils.decorators import CallDebug
from torchslime.utils import window_iter
from torchslime.components.exception import APIMisused
from torchslime.log import logger


class HandlerWrapper(Handler):
    
    __handler: Handler = NOTHING
    
    def set_handler(self, handler: Handler) -> None:
        self.__handler = handler
        
    def get_handler(self) -> Handler:
        return self.__handler if hasattr(self, '_HandlerWrapper__handler') else NOTHING
    
    def handle(self, ctx: BaseContext):
        # NOTE: use ``handle`` rather than ``__call__`` here
        # NOTE: use launch hook to process ``exec_ranks``
        ctx.hook_ctx.launch.handler_handle(self.__handler, ctx)


class HandlerWrapperContainer(Handler, BaseList[HandlerWrapper]):
    
    def __init__(self, wrappers: List[HandlerWrapper]):
        Handler.__init__(self)
        BaseList.__init__(self, wrappers)
    
    def handle(self, ctx: BaseContext):
        if len(self) < 1:
            raise APIMisused('Call an empty ``HandlerWrapperContainer`` is not allowed.')
        if is_none_or_nothing(self[-1].get_handler()):
            raise APIMisused('You should apply ``bind(handler)`` first before calling ``HandlerWrapperContainer``.')
        # NOTE: use ``handle`` rather than ``__call__`` here
        # NOTE: use launch hook to process ``exec_ranks``
        # call the first handler here
        ctx.hook_ctx.launch.handler_handle(self[0], ctx)
    
    def bind(self, handler: Handler):
        if len(self) < 1:
            raise APIMisused('Could not bind an empty ``HandlerWrapperContainer``.')
        self[-1].set_handler(handler)
    
    def __setitem__(
        self,
        __key: Union[SupportsIndex, slice],
        __value: Union[HandlerWrapper, Iterable[HandlerWrapper]]
    ) -> None:
        super().__setitem__(__key, __value)
        # TODO: performance optimization
        self.link_list()
    
    def __delitem__(
        self,
        __key: Union[SupportsIndex, slice]
    ):
        super().__delitem__(__key)
        # TODO: performance optimization
        self.link_list()
    
    def insert(self, __index: SupportsIndex, __handler: HandlerWrapper) -> None:
        super().insert(__index, __handler)
        # get index int
        __index = __index.__index__()
        
        # link previous
        if __index >= 1:
            self[__index - 1].set_handler(__handler)
        # link next
        if __index <= len(self) - 2:
            __handler.set_handler(self[__index + 1])
    
    def link_list(self):
        for _prev, _next in window_iter(self, 2):
            _prev.set_handler(_next)

#
# StateHandler
#

class StateHandler(HandlerWrapper):
    
    def __init__(
        self,
        state: str = 'train',
        restore: bool = True
    ):
        super().__init__()
        # get state supported
        from torchslime.core.hooks.state import state_registry
        mode_supported = list(state_registry.keys())
        if state not in mode_supported:
            logger.warn('An unsupported state is set, this may cause some problems.')
        self.state = state
        self.restore = restore
    
    @CallDebug(module_name='StateHandler')
    def handle(self, ctx: BaseContext):
        # cache the state before state set
        self.restore_state: Union[StateHook, Nothing] = ctx.hook_ctx.state
        ctx.hook_ctx.state: StateHook = state_registry.get(self.state)()
        ctx.hook_ctx.state.set_model_mode(ctx)
        # call wrapped handler
        super().handle(ctx)
        # restore
        if self.restore:
            from torchslime.core.hooks.state import StateHook
            ctx.hook_ctx.state: StateHook = self.restore_state
            ctx.hook_ctx.state.set_model_mode(ctx)
        if hasattr(self, 'restore_state'):
            # destroy cached state
            del self.restore_state
    
    def _get_attr_dict(self) -> dict:
        custom_attrs = {
            'state': self.state,
            'restore': self.restore
        }
        return {
            **super()._get_attr_dict(),
            **custom_attrs
        }

#
# Condition Handler
#

class ConditionHandler(HandlerWrapper):
    
    def __init__(self, condition: Callable[[BaseContext], bool]):
        super().__init__()
        self.condition = condition
    
    @CallDebug(module_name='ConditionHandler')
    def handle(self, ctx: BaseContext):
        if self.condition(ctx):
            super().handle(ctx)

#
# Condition Operators
#

class _ConditionOperator:
    
    def __init__(self, *conditions: Callable[[BaseContext], bool]) -> None:
        self.conditions = conditions

    def __call__(self, ctx: BaseContext) -> bool: pass

class And(_ConditionOperator):
    
    def __call__(self, ctx: BaseContext) -> bool:
        for condition in self.conditions:
            if not condition(ctx):
                return False
        return True

class Or(_ConditionOperator):
    
    def __call__(self, ctx: BaseContext) -> bool:
        for condition in self.conditions:
            if condition(ctx):
                return True
        return False

class Not(_ConditionOperator):
    
    def __init__(self, *conditions: Callable[[BaseContext], bool]) -> None:
        super().__init__(*conditions)
        if len(conditions) != 1:
            raise ValueError('``Not`` operation only accept 1 argument, but {} found'.format(len(conditions)))
    
    def __call__(self, ctx: BaseContext) -> bool:
        return not self.conditions[0](ctx)

#
# Condition Functions
#

def validation_check(ctx: BaseContext) -> bool:
    valid_freq = ctx.run_ctx.valid_freq
    if isinstance(valid_freq, Callable):
        return valid_freq(ctx)
    elif isinstance(valid_freq, list):
        return ctx.iteration_ctx.current in valid_freq
    else:
        # NOTE: current step is added by 1
        current = ctx.iteration_ctx.current + 1
        total = ctx.iteration_ctx.total
        
        valid_per = total // valid_freq
        is_validation = current % valid_per == 0
        is_final_validation = (total - current) < valid_per
        is_final = current >= total
        # the last validation is set at the last step, rather than set by ``valid_per``
        return (is_validation and not is_final_validation) or is_final

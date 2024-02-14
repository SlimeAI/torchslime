from torchslime.core.context.base import BaseContext
from .riching import HandlerWrapperContainerProfiler
from . import Handler, HandlerContainer
from torchslime.utils.bases import (
    BaseGenerator,
    GeneratorQueue
)
from torchslime.utils.typing import (
    NOTHING,
    NoneOrNothing,
    Nothing,
    Union,
    List,
    Callable,
    Generator,
    TypeVar,
    NoReturn,
    TYPE_CHECKING
)
from torchslime.components.exception import (
    APIMisused,
    HandlerBaseException,
    HandlerWrapperException
)
from torchslime.logging.logger import logger
from torchslime.logging.rich import RenderInterface, RenderableType
if TYPE_CHECKING:
    from torchslime.core.hooks.state import StateHook

__all__ = [
    'HandlerWrapper',
    'HandlerWrapperContainer',
    'StateWrapper',
    'ConditionWrapper'
]
_T = TypeVar('_T')
_YieldT_co = TypeVar('_YieldT_co', covariant=True)
_SendT_contra = TypeVar('_SendT_contra', contravariant=True)
_ReturnT_co = TypeVar('_ReturnT_co', covariant=True)


class HandlerWrapperGenerator(BaseGenerator[_YieldT_co, _SendT_contra, _ReturnT_co]):
    
    def __init__(
        self,
        __handler: 'HandlerWrapper',
        __ctx: BaseContext,
        *,
        exit_allowed: bool = True
    ) -> None:
        self.handler = __handler
        __gen = __handler.handle_yield(__ctx)
        super().__init__(__gen, exit_allowed=exit_allowed)
    
    def call__(self, __caller: Callable[[], _T]) -> _T:
        try:
            return super().call__(__caller)
        # directly raise Handler Base Exception
        except HandlerBaseException as hbe:
            raise hbe
        # wrap other Exception with Handler Wrapper Exception
        except Exception as e:
            raise HandlerWrapperException(exception_handler=self.handler, exception=e)


class HandlerWrapper(Handler):
    
    def __init__(self, *, id: Union[str, NoneOrNothing] = NOTHING):
        super().__init__(id=id)
    
    def handle(self, ctx: BaseContext) -> NoReturn:
        raise APIMisused(
            '``HandlerWrapper`` does not support ``handle``. Please use ``handle_yield`` instead.'
        )
    
    def get_display_attr_dict(self) -> dict:
        return {
            'id': self.get_id()
        }
    
    # set and get method won't work
    def set_exec_ranks(self, *args, **kwargs) -> None: pass
    def get_exec_ranks(self) -> Nothing: return NOTHING
    def set_wrappers(self, *args, **kwargs) -> None: pass
    def get_wrappers(self) -> Nothing: return NOTHING
    def set_lifecycle(self, *args, **kwargs): pass
    def get_lifecycle(self) -> Nothing: return NOTHING
    # yield method
    def handle_yield(self, ctx: BaseContext) -> Generator: yield True
    def gen__(self, ctx: BaseContext) -> HandlerWrapperGenerator: return HandlerWrapperGenerator(self, ctx)


_T_HandlerWrapper = TypeVar('_T_HandlerWrapper', bound=HandlerWrapper)

class HandlerWrapperContainer(HandlerWrapper, HandlerContainer[_T_HandlerWrapper], RenderInterface):
    
    def __init__(self, wrappers: List[_T_HandlerWrapper], *, id: Union[str, NoneOrNothing] = NOTHING):
        HandlerContainer.__init__(self, wrappers, id=id)
        self.profiler = HandlerWrapperContainerProfiler()
    
    def handle(self, ctx: BaseContext, wrapped: Handler) -> None:
        # the original generator list
        gen_list: List[HandlerWrapperGenerator] = [wrapper.gen__(ctx) for wrapper in self]
        # generator stack
        stack = GeneratorQueue[HandlerWrapperGenerator](reverse=True)
        
        with stack:
            # yield state controlling wrapper exec
            state = True
            # before handle
            for gen in gen_list:
                state = gen()
                stack.append(gen)
                if not state:
                    break
            # handle
            if state:
                wrapped.handle(ctx)
        
        # after handle
        for gen in stack.pop_gen__():
            with stack:
                gen.send(wrapped)
    
    def render__(self) -> RenderableType:
        return self.profiler.profile(self)

#
# StateHandler
#

class StateWrapper(HandlerWrapper):
    
    def __init__(
        self,
        state: Union[str, "StateHook"] = 'train',
        restore: bool = True,
        *,
        id: Union[str, NoneOrNothing] = NOTHING
    ):
        super().__init__(id=id)
        # get state supported
        from torchslime.core.hooks.state import StateHook, state_registry
        registered_states = list(state_registry.keys())
        if not isinstance(state, StateHook) and state not in registered_states:
            logger.warning(
                'An unregistered state is set, this may cause some problems. '
                f'Registered states: {registered_states} - Specified state: {state}.'
            )
        self.state = state
        self.restore = restore
    
    def handle_yield(self, ctx: BaseContext):
        # cache the state before state set
        self.restore_state: Union["StateHook", Nothing] = ctx.hook_ctx.state
        ctx.compile.state_hook_compile__(self.state)
        # call wrapped handler
        yield True
        # restore
        if self.restore:
            ctx.compile.state_hook_compile__(self.restore_state)
        if hasattr(self, 'restore_state'):
            # destroy cached state
            del self.restore_state
    
    def get_display_attr_dict(self) -> dict:
        custom_attrs = {
            'state': self.state,
            'restore': self.restore
        }
        return {
            **super().get_display_attr_dict(),
            **custom_attrs
        }

#
# Condition Handler
#

class ConditionWrapper(HandlerWrapper):
    
    def __init__(
        self,
        condition: Callable[[BaseContext], bool],
        *,
        id: Union[str, NoneOrNothing] = NOTHING
    ):
        super().__init__(id=id)
        self.condition = condition
    
    def handle_yield(self, ctx: BaseContext):
        yield self.condition(ctx)

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
    
    def __init__(self, condition: Callable[[BaseContext], bool]) -> None:
        super().__init__(condition)
        self.condition = self.conditions[0]
    
    def __call__(self, ctx: BaseContext) -> bool:
        return not self.condition(ctx)

#
# Condition Functions
#

def validation_check(ctx: BaseContext) -> bool:
    valid_freq = ctx.run_ctx.valid_freq
    if callable(valid_freq):
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

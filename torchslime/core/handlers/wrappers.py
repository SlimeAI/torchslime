from torchslime.core.context.base import BaseContext
from .riching import HandlerWrapperContainerProfiler
from . import Handler, HandlerContainer, HandlerMeta
from torchslime.core.hooks.state import state_registry
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
    overload,
    Type,
    NoReturn
)
from torchslime.components.exception import (
    APIMisused,
    HandlerBaseException,
    HandlerWrapperException
)
from torchslime.utils.decorators import RemoveOverload
from torchslime.logging.logger import logger
from torchslime.logging.rich import RenderInterface, RenderableType
from functools import partial

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
    
    def send(self, __value: _SendT_contra) -> _YieldT_co:
        return self.gen_call__(partial(super().send, __value))
    
    def throw(self, __typ, __val=None, __tb=None) -> _YieldT_co:
        return self.gen_call__(partial(super().throw, __typ, __val, __tb))
    
    def gen_call__(self, __caller: Callable[[], _T]) -> _T:
        try:
            return __caller()
        # directly raise Handler Base Exception
        except HandlerBaseException as hbe:
            raise hbe
        # wrap other Exception with Handler Wrapper Exception
        except Exception as e:
            raise HandlerWrapperException(exception_handler=self.handler, exception=e)


@RemoveOverload(checklist=['m__'])
class HandlerWrapperMeta(HandlerMeta):
    
    # just for type hint
    @overload
    @classmethod
    def m__(
        cls: Type[_T],
        id: Union[str, NoneOrNothing] = NOTHING
    ) -> Type[_T]: pass
    
    def m_init__(self, id=NOTHING):
        self.set_id(id)
    
    def get_meta_dict(self) -> dict:
        return {
            'id': self.get_id()
        }
    
    def set_exec_ranks(self, *args, **kwargs) -> None: pass
    def get_exec_ranks(self) -> Nothing: return NOTHING
    def set_wrappers(self, *args, **kwargs) -> None: pass
    def get_wrappers(self) -> Nothing: return NOTHING
    def set_lifecycle(self, *args, **kwargs): pass
    def get_lifecycle(self) -> Nothing: return NOTHING


class HandlerWrapper(HandlerWrapperMeta, Handler):
    
    def handle(self, ctx: BaseContext) -> NoReturn:
        raise APIMisused(
            '``HandlerWrapper`` does not support ``handle``. Please use ``handle_yield`` instead.'
        )
    
    def handle_yield(self, ctx: BaseContext) -> Generator: yield True
    def gen__(self, ctx: BaseContext) -> HandlerWrapperGenerator: return HandlerWrapperGenerator(self, ctx)


_T_HandlerWrapper = TypeVar('_T_HandlerWrapper', bound=HandlerWrapper)

class HandlerWrapperContainer(HandlerWrapperMeta, HandlerContainer[_T_HandlerWrapper], RenderInterface):
    
    def __init__(self, wrappers: List[_T_HandlerWrapper]):
        super().__init__(wrappers)
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
        state: str = 'train',
        restore: bool = True
    ):
        super().__init__()
        # get state supported
        from torchslime.core.hooks.state import state_registry
        mode_supported = list(state_registry.keys())
        if state not in mode_supported:
            logger.warning('An unsupported state is set, this may cause some problems.')
        self.state = state
        self.restore = restore
    
    def handle_yield(self, ctx: BaseContext):
        # cache the state before state set
        self.restore_state: Union[StateHook, Nothing] = ctx.hook_ctx.state
        ctx.hook_ctx.state: StateHook = state_registry.get(self.state)()
        ctx.hook_ctx.state.set_model_mode(ctx)
        # call wrapped handler
        yield True
        # restore
        if self.restore:
            from torchslime.core.hooks.state import StateHook
            ctx.hook_ctx.state: StateHook = self.restore_state
            ctx.hook_ctx.state.set_model_mode(ctx)
        if hasattr(self, 'restore_state'):
            # destroy cached state
            del self.restore_state
    
    def get_attr_dict(self) -> dict:
        custom_attrs = {
            'state': self.state,
            'restore': self.restore
        }
        return {
            **super().get_attr_dict(),
            **custom_attrs
        }

#
# Condition Handler
#

class ConditionWrapper(HandlerWrapper):
    
    def __init__(self, condition: Callable[[BaseContext], bool]):
        super().__init__()
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

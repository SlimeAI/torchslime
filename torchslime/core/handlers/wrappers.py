from torchslime.core.context.base import BaseContext
from . import Handler, HandlerContainer
from torchslime.core.hooks.state import state_registry
from torchslime.utils.bases import Nothing
from torchslime.utils.typing import (
    Union,
    List,
    Callable,
    Generator,
    Tuple,
    TypeVar
)
from torchslime.utils import GeneratorCaller, FuncCaller
from torchslime.components.exception import (
    APIMisused,
    HandlerBaseException,
    HandlerWrapperException
)
from torchslime.log import logger

__all__ = [
    'HandlerWrapper',
    'HandlerWrapperContainer',
    'StateHandler',
    'ConditionHandler'
]
_T = TypeVar('_T')


class HandlerWrapper(Handler):
    
    def handle(self, ctx: BaseContext) -> Generator: pass
    
    def gen(self, ctx: BaseContext) -> GeneratorCaller:
        return GeneratorCaller(FuncCaller(self.handle, ctx))


class HandlerWrapperContainer(HandlerContainer[Union[HandlerWrapper, _T]]):
    
    def __init__(self, wrappers: List[HandlerWrapper]):
        super().__init__(wrappers)
    
    def handle(self, ctx: BaseContext, wrapped: Handler):
        gen_list: List[Tuple[GeneratorCaller, HandlerWrapper]] = []
        flag = True
        # before handle
        for gen, wrapper in zip(
            map(lambda _wrapper: _wrapper.gen(ctx), self),
            self
        ):
            flag = _gen_call(gen, wrapper)
            gen_list.append((gen, wrapper))
            if not flag:
                break
        # handle
        if flag:
            wrapped.handle(ctx)
        # after handle
        for gen, wrapper in reversed(gen_list):
            _gen_call(gen, wrapper)

def _gen_call(gen: GeneratorCaller, wrapper: HandlerWrapper):
    try:
        return gen()
    # directly raise Handler Base Exception
    except HandlerBaseException as hbe:
        raise hbe
    # wrap other Exception with Handler Wrapper Exception
    except Exception as e:
        raise HandlerWrapperException(exception_handler=wrapper, exception=e)

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
    
    def handle(self, ctx: BaseContext):
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
    
    def handle(self, ctx: BaseContext):
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

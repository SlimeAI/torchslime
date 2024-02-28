from .riching import HandlerWrapperContainerProfiler
from . import Handler, HandlerContainer
from torchslime.utils.base import (
    ContextGenerator,
    ContextGeneratorStack
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
    TYPE_CHECKING,
    STOP,
    Iterable,
    Pass,
    PASS
)
from torchslime.utils.exception import (
    HandlerBaseException,
    HandlerWrapperException
)
from torchslime.logging.logger import logger
from torchslime.logging.rich import RenderInterface, RenderableType
if TYPE_CHECKING:
    from torchslime.pipelines.state import ModelState
    from torchslime.context import Context

__all__ = [
    'HandlerWrapper',
    'HandlerWrapperContainer',
    'StateWrapper',
    'ConditionWrapper'
]
_T = TypeVar("_T")
_YieldT_co = TypeVar("_YieldT_co", covariant=True)
_SendT_contra = TypeVar("_SendT_contra", contravariant=True)
_ReturnT_co = TypeVar("_ReturnT_co", covariant=True)


class HandlerWrapperGenerator(ContextGenerator[_YieldT_co, _SendT_contra, _ReturnT_co]):
    """
    ``HandlerWrapperGenerator`` defines custom exception handling in the ``call__`` method 
    compared to ``ContextGenerator``.
    """
    
    def __init__(
        self,
        __gen: Generator[_YieldT_co, _SendT_contra, _ReturnT_co],
        __wrapper: "HandlerWrapper",
        *,
        exit_allowed: bool = True
    ) -> None:
        ContextGenerator.__init__(self, __gen, exit_allowed=exit_allowed)
        self.wrapper = __wrapper
    
    def call__(self, __caller: Callable[[], _T]) -> _T:
        try:
            return super().call__(__caller)
        # directly raise Handler Base Exception
        except HandlerBaseException:
            raise
        # wrap other Exception with Handler Wrapper Exception
        except Exception as e:
            raise HandlerWrapperException(exception_handler=self.wrapper, exception=e)


_HandlerT = TypeVar("_HandlerT", bound=Handler)

class HandlerWrapper(HandlerContainer[_HandlerT]):
    
    def handle(self, ctx: "Context") -> None:
        # Use ``ContextGenerator`` here rather than ``HandlerWrapperGenerator``, 
        # because when calling ``handle`` method, ``HandlerWrapper`` works as a 
        # normal ``HandlerContainer``.
        # The ``wrapped`` param is set to ``self``.
        with ContextGenerator(self.handle_yield(ctx, wrapped=self)) as val:
            if val is not STOP:
                super().handle(ctx)
    
    # yield method API
    def handle_yield(self, ctx: "Context", wrapped: Handler) -> Generator: yield
    
    def gen__(self, ctx: "Context", wrapped: Handler) -> HandlerWrapperGenerator:
        """
        Creates a new ``HandlerWrapperGenerator`` that calls ``handle_yield`` method.
        """
        return HandlerWrapperGenerator(
            self.handle_yield(ctx, wrapped),
            self
        )


_HandlerWrapperT = TypeVar("_HandlerWrapperT", bound=HandlerWrapper)

class HandlerWrapperContainer(HandlerContainer[_HandlerWrapperT], RenderInterface):
    
    def __init__(
        self,
        wrappers: List[_HandlerWrapperT],
        *,
        id: Union[str, NoneOrNothing] = NOTHING
    ) -> None:
        HandlerContainer.__init__(self, wrappers, id=id)
        # Display ``HandlerWrapper`` using ``rich``.
        self.profiler = HandlerWrapperContainerProfiler()
    
    def handle(self, ctx: "Context", wrapped: Handler) -> None:
        with ContextGeneratorStack((
            wrapper.gen__(ctx, wrapped) for wrapper in self
        )) as vals:
            # Check whether the yielded value is ``STOP``
            val = vals[-1] if len(vals) > 0 else True
            if val is not STOP:
                wrapped.handle(ctx)
    
    def render__(self) -> RenderableType:
        return self.profiler.profile(self)

#
# StateHandler
#

class StateWrapper(HandlerWrapper[_HandlerT]):
    
    def __init__(
        self,
        state: Union[str, "ModelState"] = 'train',
        restore: bool = True,
        handlers: Union[Iterable[_HandlerT], NoneOrNothing] = NOTHING,
        *,
        id: Union[str, NoneOrNothing] = NOTHING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass] = PASS,
        wrappers: Union[Iterable['HandlerWrapper'], NoneOrNothing] = NOTHING,
        lifecycle=NOTHING
    ):
        HandlerWrapper.__init__(
            self,
            handlers,
            id=id,
            exec_ranks=exec_ranks,
            wrappers=wrappers,
            lifecycle=lifecycle
        )
        # get state supported
        from torchslime.pipelines.state import ModelState, state_registry
        registered_states = list(state_registry.keys())
        if not isinstance(state, ModelState) and state not in registered_states:
            logger.warning(
                'An unregistered state is set, this may cause some problems. '
                f'Registered states: {registered_states} - Specified state: {state}.'
            )
        self.state = state
        self.restore = restore
    
    def handle_yield(self, ctx: "Context", wrapped: Handler) -> Generator:
        # cache the state before state set
        self.restore_state: Union["ModelState", Nothing] = ctx.pipeline_ctx.model_state
        ctx.compile.model_state_compile__(self.state)
        # call wrapped handler
        yield
        # restore
        if self.restore:
            ctx.compile.model_state_compile__(self.restore_state)
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

class ConditionWrapper(HandlerWrapper[_HandlerT]):
    
    def __init__(
        self,
        condition: Callable[["Context"], bool],
        handlers: Union[Iterable[_HandlerT], NoneOrNothing] = NOTHING,
        *,
        id: Union[str, NoneOrNothing] = NOTHING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass] = PASS,
        wrappers: Union[Iterable['HandlerWrapper'], NoneOrNothing] = NOTHING,
        lifecycle=NOTHING
    ):
        HandlerWrapper.__init__(
            self,
            handlers,
            id=id,
            exec_ranks=exec_ranks,
            wrappers=wrappers,
            lifecycle=lifecycle
        )
        self.condition = condition
    
    def handle_yield(self, ctx: "Context", wrapped: Handler) -> Generator:
        yield True if self.condition(ctx) else STOP

#
# Condition Operators
#

class _ConditionOperator:
    
    def __init__(self, *conditions: Callable[["Context"], bool]) -> None:
        self.conditions = conditions

    def __call__(self, ctx: "Context") -> bool: pass


class And(_ConditionOperator):
    
    def __call__(self, ctx: "Context") -> bool:
        for condition in self.conditions:
            if not condition(ctx):
                return False
        return True


class Or(_ConditionOperator):
    
    def __call__(self, ctx: "Context") -> bool:
        for condition in self.conditions:
            if condition(ctx):
                return True
        return False


class Not(_ConditionOperator):
    
    def __init__(self, condition: Callable[["Context"], bool]) -> None:
        super().__init__(condition)
        self.condition = self.conditions[0]
    
    def __call__(self, ctx: "Context") -> bool:
        return not self.condition(ctx)

#
# Condition Functions
#

def validation_check(ctx: "Context") -> bool:
    valid_freq = ctx.pipeline_ctx.valid_freq
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

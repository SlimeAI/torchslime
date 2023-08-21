from . import HandlerContainer, Handler
from torchslime.utils.typing import (
    Callable,
    Sequence,
    Union,
    Iterable
)
from torchslime.core.context.base import BaseContext
from torchslime.utils.bases import Nothing


class HandlerCondition(HandlerContainer):
    
    def __init__(
        self,
        handlers: Union[Iterable[Handler], None, Nothing] = None,
        *args,
        condition: Callable[[BaseContext], bool],
        **kwargs
    ):
        super().__init__(handlers, *args, **kwargs)
        self.condition__ = condition
    
    def handle(self, ctx: BaseContext):
        if self.condition__(ctx):
            super().handle(ctx)

#
# Condition Operators
#

class _ConditionOperator:
    
    def __init__(self, *conditions: Callable[[BaseContext], bool]) -> None:
        self.conditions__ = conditions

    def __call__(self, ctx: BaseContext) -> bool: pass

class And(_ConditionOperator):
    
    def __call__(self, ctx: BaseContext) -> bool:
        for condition in self.conditions__:
            if not condition(ctx):
                return False
        return True

class Or(_ConditionOperator):
    
    def __call__(self, ctx: BaseContext) -> bool:
        for condition in self.conditions__:
            if condition(ctx):
                return True
        return False

class Not(_ConditionOperator):
    
    def __init__(self, *conditions: Callable[[BaseContext], bool]) -> None:
        super().__init__(*conditions)
        if len(conditions) != 1:
            raise ValueError('``Not`` operation only accept 1 argument, but {} found'.format(len(conditions)))
    
    def __call__(self, ctx: BaseContext) -> bool:
        return not self.conditions__[0](ctx)

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

from . import HandlerContainer, Handler
from typing import (
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

def validation_step_check(ctx: BaseContext) -> bool:
    pass


def validation_epoch_check(ctx: BaseContext) -> bool:
    pass

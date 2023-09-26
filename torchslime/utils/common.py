from .typing import (
    Type,
    TypeVar,
    overload
)
from .meta import Meta
from .decorators import RemoveOverload

_T = TypeVar('_T')

@RemoveOverload(checklist=['m__'])
class ScopedAttrRestore(Meta):
    
    def m_init__(self, *args, **kwargs):
        return super().m_init__(*args, **kwargs)
    
    @overload
    @classmethod
    def m__(
        cls: Type[_T],
    ) -> Type[_T]: pass


class ScopedAttrAssign(ScopedAttrRestore):
    
    pass



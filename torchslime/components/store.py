from torchslime.utils.typing import (
    NOTHING,
    Any,
    Type,
    TypeVar,
    is_none_or_nothing,
    overload,
    is_slime_naming,
    List,
    Union,
    Nothing,
    MISSING,
    TYPE_CHECKING,
    Missing,
    TextIO,
    NoneOrNothing
)
from torchslime.utils.bases import (
    Base,
    AttrObservable
)
from torchslime.utils.decorators import Singleton, ItemAttrBinding, RemoveOverload
from io import TextIOWrapper
import threading
import os
# type hint only
if TYPE_CHECKING:
    from torchslime.logging.rich import SlimeConsole, SlimeAltConsole
    from torchslime.utils.launch import LaunchUtil
    from torchslime.utils.bases import (
        AttrObserver,
        ScopedAttrAssign,
        ScopedAttrRestore
    )

_T = TypeVar('_T')

#
# Scoped Store
#

class ScopedStore(Base, AttrObservable):
    
    def __init__(self) -> None:
        super().__init__()
    
    def init__(self, __name: str, __value: Any):
        """
        Init attribute only when it is not set or is ``MISSING``
        """
        if not self.hasattr__(__name) or \
                getattr(self, __name, MISSING) is MISSING:
            setattr(self, __name, __value)

@Singleton
class BuiltinScopedStore(ScopedStore):
    
    def __init__(self) -> None:
        """
        set ``builtin__`` store config
        """
        super().__init__()
        # whether to use call debug
        self.call_debug = False
        # indent str for CLI display
        self.indent_str = ' ' * 4  # default is 4 spaces
        # log template
        self.log_template: Union[str, Missing] = MISSING
        self.log_rich_template: Union[str, Missing] = MISSING
        self.log_dateformat: Union[str, Missing] = MISSING
        # launch
        self.launch: Union[str, "LaunchUtil"] = 'vanilla'
        # console
        self.console: Union["SlimeConsole", Nothing, Missing] = MISSING
        self.alt_console: Union["SlimeAltConsole", Nothing, Missing] = MISSING
        self.alt_console_files: List[Union[TextIO, TextIOWrapper]] = []

BUILTIN_SCOPED_STORE_KEY = 'builtin__'
_builtin_scoped_store = BuiltinScopedStore()
_scoped_store_dict = {}

#
# Store
#

@ItemAttrBinding
@Singleton
@RemoveOverload(checklist=[
    'subscribe__',
    'subscribe_attr__',
    'unsubscribe__',
    'unsubscribe_attr__',
    'assign__',
    'restore__'
])
class Store:
    
    def scope__(self, __key) -> Union[ScopedStore, BuiltinScopedStore]:
        if __key == BUILTIN_SCOPED_STORE_KEY:
            return _builtin_scoped_store
        elif __key in _scoped_store_dict:
            return _scoped_store_dict[__key]
        else:
            return _scoped_store_dict.setdefault(__key, ScopedStore())

    def current__(self) -> ScopedStore:
        return self.scope__(self.get_current_key__())

    def builtin__(self) -> BuiltinScopedStore:
        return self.scope__(BUILTIN_SCOPED_STORE_KEY)

    def destroy__(self, __key=NOTHING):
        if is_none_or_nothing(__key):
            __key = self.get_current_key__()
        
        if __key in _scoped_store_dict:
            del _scoped_store_dict[__key]

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.current__(), __name)

    def __getattribute__(self, __name: str) -> Any:
        # slime naming
        if is_slime_naming(__name) is True:
            return super().__getattribute__(__name)
        # else get from ScopedStore object
        return getattr(self.current__(), __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        setattr(self.current__(), __name, __value)
    
    def __delattr__(self, __name: str) -> None:
        delattr(self.current__(), __name)
    
    @staticmethod
    def get_current_key__() -> str:
        pid = os.getpid(),
        tid = threading.get_ident()
        return f'p{pid}-t{tid}'
    
    @overload
    def subscribe__(self, __observer: "AttrObserver", *, init: bool = True) -> None: pass
    @overload
    def subscribe_attr__(self, __observer: "AttrObserver", __name: str, *, init: bool = True): pass
    @overload
    def unsubscribe__(self, __observer: "AttrObserver") -> None: pass
    @overload
    def unsubscribe_attr__(self, __observer: "AttrObserver", __name: str) -> None: pass
    @overload
    def assign__(self, **kwargs) -> "ScopedAttrAssign": pass
    @overload
    def restore__(self, *attrs: str) -> "ScopedAttrRestore": pass

store = Store()

#
# Store Assign
#

from torchslime.utils.bases import ScopedAttrAssign

@RemoveOverload(checklist=['m__'])
class StoreAssign(ScopedAttrAssign[Union[ScopedStore, _T]], directly_new_allowed=True):
    
    def m_init__(
        self,
        key: Union[str, NoneOrNothing] = NOTHING,
        restore: bool = True
    ) -> None:
        self.key = store.get_current_key__() if is_none_or_nothing(key) else key
        super().m_init__(store.scope__(self.key), restore)
    
    @overload
    @classmethod
    def m__(
        cls: Type[_T],
        key: Union[str, NoneOrNothing] = NOTHING,
        restore: bool = True
    ) -> Type[_T]: pass


@RemoveOverload(checklist=['m__'])
class BuiltinStoreAssign(StoreAssign[Union[BuiltinScopedStore, _T]]):
    
    def m_init__(
        self,
        restore: bool = True
    ) -> None:
        super().m_init__(BUILTIN_SCOPED_STORE_KEY, restore)
    
    @overload
    @classmethod
    def m__(
        cls: Type[_T],
        restore: bool = True
    ) -> Type[_T]: pass

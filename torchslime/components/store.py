from torchslime.utils.typing import (
    NOTHING,
    Any,
    Dict,
    TypeVar,
    is_none_or_nothing,
    overload,
    is_slime_naming,
    List,
    Union,
    Nothing,
    MISSING,
    TYPE_CHECKING,
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
    from torchslime.utils.launch import LaunchUtil
    from torchslime.utils.bases import (
        AttrObserver,
        ScopedAttrAssign,
        ScopedAttrRestore
    )

#
# Scoped Store
#

class ScopedStore(Base, AttrObservable):
    
    def __init__(self) -> None:
        Base.__init__(self)
        AttrObservable.__init__(self)
    
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
        # call debug config
        self.call_debug = False
        self.call_debug_full_exec_name = False
        # indent str for CLI display
        self.indent_str = ' ' * 4  # default is 4 spaces
        # log template
        self.log_template: str = '{prefix__} - {asctime} - "{filename}:{lineno}" - {message}'
        self.log_rich_template: str = '{message}'
        self.log_dateformat: str = '%Y/%m/%d %H:%M:%S'
        # launch
        self.launch: Union[str, "LaunchUtil"] = 'vanilla'
    
    def delay_init__(self) -> None:
        """
        Delay initialization.
        Initialization of some items should be delayed due to circular import.
        This method should be called after creation of ``torchslime.components.store.store``.
        """
        # console
        from torchslime.logging.rich import (
            SlimeConsoleLauncher,
            SlimeAltConsoleLauncher,
            rich
        )
        self.console_launcher: Union[SlimeConsoleLauncher, Nothing] = SlimeConsoleLauncher()
        self.alt_console_launcher: Union[SlimeAltConsoleLauncher, Nothing] = SlimeAltConsoleLauncher(
            color_system=None,
            force_terminal=False,
            force_jupyter=False,
            force_interactive=False
        )
        self.alt_console_files: List[Union[TextIO, TextIOWrapper]] = []
        # set rich default console
        rich._console = self.console_launcher

BUILTIN_SCOPED_STORE_KEY = 'builtin__'
_builtin_scoped_store = BuiltinScopedStore()
_scoped_store_dict = {}

#
# Store
#

@ItemAttrBinding
@Singleton
@RemoveOverload(checklist=[
    'attach__',
    'attach_attr__',
    'detach__',
    'detach_attr__',
    'assign__',
    'restore__'
])
class Store:
    
    def scope__(self, __key) -> Union[ScopedStore, BuiltinScopedStore]:
        if __key == BUILTIN_SCOPED_STORE_KEY:
            return _builtin_scoped_store
        
        if __key not in _scoped_store_dict:
            _scoped_store_dict[__key] = ScopedStore()
        
        return _scoped_store_dict[__key]

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
        pid = os.getpid()
        tid = threading.get_ident()
        return f'p{pid}-t{tid}'
    
    @overload
    def attach__(self, __observer: "AttrObserver", *, init: bool = True) -> None: pass
    @overload
    def attach_attr__(self, __observer: "AttrObserver", __name: str, *, init: bool = True): pass
    @overload
    def detach__(self, __observer: "AttrObserver") -> None: pass
    @overload
    def detach_attr__(self, __observer: "AttrObserver", __name: str) -> None: pass
    @overload
    def assign__(self, **kwargs) -> "ScopedAttrAssign": pass
    @overload
    def restore__(self, *attrs: str) -> "ScopedAttrRestore": pass

store = Store()
# Builtin scoped store delay initialization.
store.builtin__().delay_init__()

#
# Store Assign
#

from torchslime.utils.bases import ScopedAttrAssign
_T_ScopedStore = TypeVar('_T_ScopedStore', bound=ScopedStore)

class StoreAssign(ScopedAttrAssign[_T_ScopedStore]):
    
    def __init__(
        self,
        attr_assign: Dict[str, Any],
        key: Union[str, NoneOrNothing] = NOTHING
    ) -> None:
        self.key = store.get_current_key__() if is_none_or_nothing(key) else key
        super().__init__(store.scope__(self.key), attr_assign)


_T_BuiltinScopedStore = TypeVar('_T_BuiltinScopedStore', bound=BuiltinScopedStore)

class BuiltinStoreAssign(StoreAssign[_T_BuiltinScopedStore]):
    
    def __init__(
        self,
        attr_assign: dict[str, Any]
    ) -> None:
        super().__init__(attr_assign, BUILTIN_SCOPED_STORE_KEY)

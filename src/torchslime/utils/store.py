from .typing import (
    Any,
    Dict,
    TypeVar,
    List,
    Union,
    Nothing,
    Missing,
    MISSING,
    TYPE_CHECKING,
    TextIO
)
from .base import (
    Singleton
)
from io import TextIOWrapper
from slime_core.utils.store import (
    ScopedStore,
    CoreStore
)
# type hint only
if TYPE_CHECKING:
    from .launch import LaunchUtil
    from .base import (
        ScopedAttrAssign
    )


class BuiltinScopedStore(ScopedStore, Singleton):
    
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
        This method should be called after creation of ``torchslime.utils.store.store``.
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

#
# Store
#

class Store(CoreStore):
    
    # NOTE: ``_builtin_scoped_store`` is not contained in the 
    # ``scoped_store_dict__``.
    scoped_store_dict__: Dict[str, ScopedStore] = {}
    
    def scope__(self, __key: str) -> Union[ScopedStore, BuiltinScopedStore]:
        if __key == BUILTIN_SCOPED_STORE_KEY:
            return _builtin_scoped_store
        
        return super().scope__(__key)

    def builtin__(self) -> BuiltinScopedStore:
        return self.scope__(BUILTIN_SCOPED_STORE_KEY)


store = Store()
# Builtin scoped store delay initialization.
store.builtin__().delay_init__()

#
# Store Assign
#

from .base import ScopedAttrAssign
_ScopedStoreT = TypeVar("_ScopedStoreT", bound=ScopedStore)
_BuiltinScopedStoreT = TypeVar("_BuiltinScopedStoreT", bound=BuiltinScopedStore)


class StoreAssign(ScopedAttrAssign[_ScopedStoreT]):
    
    def __init__(
        self,
        attr_assign: Dict[str, Any],
        key: Union[str, Missing] = MISSING
    ) -> None:
        self.key = store.get_current_key__() if key is MISSING else key
        super().__init__(store.scope__(self.key), attr_assign)


class BuiltinStoreAssign(StoreAssign[_BuiltinScopedStoreT]):
    
    def __init__(
        self,
        attr_assign: Dict[str, Any]
    ) -> None:
        super().__init__(attr_assign, BUILTIN_SCOPED_STORE_KEY)

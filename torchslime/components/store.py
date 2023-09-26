from torchslime.utils.typing import (
    NOTHING,
    Any,
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
    TextIO
)
from torchslime.utils.bases import Base, BaseAttrObservable, BaseAttrObserver
from torchslime.utils.decorators import Singleton, ItemAttrBinding, ContextDecoratorBinding, RemoveOverload
from io import TextIOWrapper
import threading
import os
# type hint only
if TYPE_CHECKING:
    from torchslime.logging.rich import SlimeConsole, SlimeAltConsole
    from torchslime.utils.launch import LaunchUtil

_T = TypeVar('_T')


class ScopedStore(Base, BaseAttrObservable):
    
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

_builtin_scoped_store = BuiltinScopedStore()

_scoped_store_dict = {}

@ItemAttrBinding
@Singleton
@RemoveOverload(checklist=[
    'subscribe__',
    'subscribe_attr__',
    'unsubscribe__',
    'unsubscribe_attr__'
])
class Store:
    
    def scope__(self, __key) -> Union[ScopedStore, BuiltinScopedStore]:
        if __key == 'builtin__':
            return _builtin_scoped_store
        elif __key in _scoped_store_dict:
            return _scoped_store_dict[__key]
        else:
            return _scoped_store_dict.setdefault(__key, ScopedStore())

    def current__(self) -> ScopedStore:
        return self.scope__(self.get_current_key__())

    def builtin__(self) -> BuiltinScopedStore:
        return self.scope__('builtin__')

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
    def subscribe__(self, __observer: BaseAttrObserver, *, init: bool = True) -> None: pass
    @overload
    def subscribe_attr__(self, __observer: BaseAttrObserver, __name: str, *, init: bool = True): pass
    @overload
    def unsubscribe__(self, __observer: BaseAttrObserver) -> None: pass
    @overload
    def unsubscribe_attr__(self, __observer: BaseAttrObserver, __name: str) -> None: pass


@ContextDecoratorBinding
@RemoveOverload(checklist=['__call__'])
class StoreSet:

    def __init__(self, __name: str, __value: Any, *, restore: bool = True, key=NOTHING) -> None:
        self.name = __name
        self.value = __value
        self.restore = restore
        self.key = store.get_current_key__() if is_none_or_nothing(key) is True else key
        self._store = store.scope__(self.key)
    
    # just for type hint
    @overload
    def __call__(self, func: _T) -> _T: pass

    def __enter__(self) -> 'StoreSet':
        self._set_value()
        return self

    def __exit__(self, *args, **kwargs):
        self._restore_value()
    
    def _set_value(self):
        # cache the store value before ``StoreSet``
        self.prev_value = self._store[self.name]
        # set value
        self._store[self.name] = self.value

    def _restore_value(self):
        if self.restore is True:
            self._store[self.name] = self.prev_value
        else:
            del self._store[self.name]
        del self.prev_value


class BuiltinStoreSet(StoreSet):
    
    def __init__(self, __name: str, __value: Any, *, restore: bool = True) -> None:
        super().__init__(__name, __value, restore=restore, key='builtin__')


store = Store()

"""
Rich Native Utils
"""
from torchslime.utils.store import store
from torchslime.utils.typing import (
    MISSING,
    Iterable,
    Missing,
    NoneOrNothing,
    Pass,
    Union,
    Nothing,
    NOTHING,
    TypeVar,
    Type
)
from torchslime.utils.launch import LaunchUtil, Launcher
from torchslime.utils.base import (
    AttrObserver,
    AttrObserve,
    AttrObservable,
    ScopedAttrRestore,
    BiList,
    MutableBiListItem
)
from torchslime.utils.metaclass import (
    Metaclasses,
    InitOnceMetaclass
)
from abc import (
    ABCMeta
)
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.console import Console, RenderableType, Group
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.logging import RichHandler
from rich.table import Table
from rich.protocol import is_renderable
import threading
import multiprocessing

# NOTE: import rich api for compatibility
import rich
from rich.markup import escape

_T = TypeVar("_T")

_LauncherT = TypeVar("_LauncherT", bound=Launcher)

class RichLauncher(Launcher):
    
    def get__(self: _LauncherT) -> Union[_LauncherT, Nothing]:
        return self if self.is_exec__() else NOTHING

#
# Rich Console Adapter
#

class SlimeConsoleLauncher(Console, RichLauncher):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        *args,
        **kwargs
    ):
        Console.__init__(self, *args, **kwargs)
        RichLauncher.__init__(self, launch, exec_ranks)


class SlimeAltConsoleLauncher(SlimeConsoleLauncher):
    
    __t_lock = threading.Lock()
    __p_lock = multiprocessing.Lock()
    
    def get__(self):
        console = super().get__()
        # check whether the console is exec
        if console is NOTHING:
            # use ``yield from`` rather than ``yield`` to create an empty generator
            yield from NOTHING
        else:
            with self.__t_lock, self.__p_lock, ScopedAttrRestore(console, ['file']):
                # set files to the alt console
                for file in store.builtin__().alt_console_files:
                    console.file = file
                    yield console


def yield_console(
    enable_console: bool = True,
    enable_alt_console: bool = False
):
    if enable_console:
        console_launcher = store.builtin__().console_launcher.get__()
        yield from console_launcher if console_launcher is NOTHING else [console_launcher]
    
    if enable_alt_console:
        yield from store.builtin__().alt_console_launcher.get__()

#
# Console Observer
#

class SlimeConsoleObserver(AttrObserver):
    
    def __init__(self) -> None:
        AttrObserver.__init__(self)
        # auto attach observer
        store.builtin__().attach__(self, namespaces=['builtin_store_console__'])
    
    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None: pass
    
    @AttrObserve(namespace='builtin_store_console__')
    def console_launcher_observe__(self, new_value: SlimeConsoleLauncher, old_value, observable: AttrObservable):
        self.set_console__(new_value)

#
# Start-Stop Observer Manager
#

class _StartStopObserverManager:
    """
    Because some ``rich`` objects work only when they start, we want to detach some 
    observed attributes when they stop in order to improve performance, and re-attach 
    them when they start again.
    """
    def __init__(self) -> None: pass
    
    def start_attach__(self):
        """
        Attach attributes before start.
        """
        pass
    
    def stop_detach__(self):
        """
        Detach attributes after stop.
        """
        pass

#
# Rich Live Adapter
#

class SlimeLiveLauncher(
    Live,
    _StartStopObserverManager,
    RichLauncher,
    SlimeConsoleObserver,
    metaclass=Metaclasses(ABCMeta, InitOnceMetaclass)
):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        *args,
        **kwargs
    ) -> None:
        Live.__init__(self, *args, **kwargs)
        RichLauncher.__init__(self, launch, exec_ranks)
        SlimeConsoleObserver.__init__(self)
        # stop detach first.
        self.stop_detach__()
    
    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None:
        self.console = __console
    
    def start(self, *args, **kwargs) -> None:
        self.start_attach__()
        return super().start(*args, **kwargs)
    
    def stop(self, *args, **kwargs) -> None:
        result = super().stop(*args, **kwargs)
        self.stop_detach__()
        return result
    
    def start_attach__(self):
        # attach console
        store.builtin__().attach__(self, namespaces=['builtin_store_console__'])
        # attach launch (if and only if ``bind_launch_to_builtin_store__`` is True)
        if self.bind_launch_to_builtin_store__:
            store.builtin__().attach__(self, namespaces=['builtin_store_launch__'])
    
    def stop_detach__(self):
        # detach builtin store.
        store.builtin__().detach__(self, namespaces=['builtin_store_console__', 'builtin_store_launch__'])

#
# Rich Logging Adapter
#

class SlimeRichHandler(RichHandler, SlimeConsoleObserver):
    
    def __init__(self, *args, **kwargs):
        RichHandler.__init__(self, *args, **kwargs)
        SlimeConsoleObserver.__init__(self)

#
# Renderable Adapter
#

class SlimeProgressLauncher(
    Progress,
    _StartStopObserverManager,
    MutableBiListItem,
    RichLauncher,
    SlimeConsoleObserver,
    metaclass=Metaclasses(ABCMeta, InitOnceMetaclass)
):

    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        *args,
        **kwargs
    ) -> None:
        Progress.__init__(self, *args, **kwargs)
        MutableBiListItem.__init__(self)
        RichLauncher.__init__(self, launch, exec_ranks)
        SlimeConsoleObserver.__init__(self)
        # stop detach first.
        self.stop_detach__()

    @classmethod
    def create__(cls: Type[_T]) -> _T:
        return cls(
            # ``launch`` and ``exec_ranks`` args
            MISSING,
            MISSING,
            # rich.progress.Progress args
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )

    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None:
        self.live.console = __console
    
    def start(self, *args, **kwargs) -> None:
        self.start_attach__()
        return super().start(*args, **kwargs)
    
    def stop(self, *args, **kwargs) -> None:
        result = super().stop(*args, **kwargs)
        self.stop_detach__()
        return result
    
    def start_attach__(self):
        # attach console
        store.builtin__().attach__(self, namespaces=['builtin_store_console__'])
        # attach launch (if and only if ``bind_launch_to_builtin_store__`` is True)
        if self.bind_launch_to_builtin_store__:
            store.builtin__().attach__(self, namespaces=['builtin_store_launch__'])
    
    def stop_detach__(self):
        # detach builtin store.
        store.builtin__().detach__(self, namespaces=['builtin_store_console__', 'builtin_store_launch__'])


class SlimeText(Text, MutableBiListItem):
    
    def __init__(self, *args, **kwargs):
        Text.__init__(self, *args, **kwargs)
        MutableBiListItem.__init__(self)


class SlimeTree(Tree, MutableBiListItem):
    
    def __init__(self, *args, **kwargs):
        Tree.__init__(self, *args, **kwargs)
        MutableBiListItem.__init__(self)


class SlimePanel(Panel, MutableBiListItem):
    
    def __init__(self, *args, **kwargs):
        Panel.__init__(self, *args, **kwargs)
        MutableBiListItem.__init__(self)


class SlimeTable(Table, MutableBiListItem):
    
    def __init__(self, *args, **kwargs):
        Table.__init__(self, *args, **kwargs)
        MutableBiListItem.__init__(self)


_RichRenderableT = TypeVar("_RichRenderableT", bound=Union[RenderableType, MutableBiListItem])

class SlimeGroup(Group, MutableBiListItem, BiList[_RichRenderableT]):
    
    def __init__(self, *renderables: RenderableType, fit: bool = True) -> None:
        Group.__init__(self, *renderables, fit=fit)
        MutableBiListItem.__init__(self)
        BiList.__init__(self)
        self.set_list__(self.renderables)

#
# Custom Rich Components
#

class ProfileProgress(SlimeGroup[_RichRenderableT]):
    
    def __init__(
        self,
        __progress: Union[SlimeProgressLauncher, Missing] = MISSING,
        __text: Union[str, SlimeText, Missing] = MISSING,
        *,
        fit: bool = True
    ) -> None:
        self.progress = self._parse_progress(__progress)
        self.text = self._parse_text(__text)
        
        super().__init__(
            self.progress,
            self.text,
            fit=fit
        )
    
    def _parse_progress(
        self,
        __progress: Union[SlimeProgressLauncher, Missing] = MISSING
    ) -> SlimeProgressLauncher:
        if __progress is MISSING:
            __progress = SlimeProgressLauncher.create__()
        return __progress
    
    def set_progress__(self, __progress: Union[SlimeProgressLauncher, Missing] = MISSING):
        __progress = self._parse_progress(__progress)
        self.progress.replace_self__(__progress)
        self.progress = __progress
    
    def _parse_text(
        self,
        __text: Union[str, SlimeText, Missing] = MISSING
    ) -> SlimeText:
        if __text is MISSING:
            __text = SlimeText("")
        elif isinstance(__text, str):
            __text = SlimeText(__text)
        return __text
    
    def set_text__(self, __text: Union[str, SlimeText, Missing] = MISSING):
        __text = self._parse_text(__text)
        self.text.replace_self__(__text)
        self.text = __text


class RenderInterface:
    def render__(self) -> RenderableType: pass

def is_render_interface(item) -> bool:
    return isinstance(item, RenderInterface) or hasattr(item, 'render__')


def parse_renderable(item) -> RenderableType:
    # NOTE: ``NOTHING`` will be inspected as ``renderable``, because 
    # ``hasattr(NOTHING, '__rich__')`` and ``hasattr(NOTHING, '__rich_console__')`` 
    # will both return True. However, it should not actually be seen as ``renderable``.
    if item is NOTHING:
        return str(item)
    elif is_render_interface(item):
        item: RenderInterface
        return item.render__()
    elif is_renderable(item):
        return item
    else:
        return str(item)

"""
Rich Native Utils
"""
from torchslime.components.store import store
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
    overload,
    Type,
    TYPE_CHECKING
)
from torchslime.utils.launch import LaunchUtil, Launcher
from torchslime.utils.decorators import RemoveOverload
from torchslime.utils.bases import (
    AttrObserver,
    AttrObserve,
    ScopedAttrRestore,
    BiList,
    MutableBiListItem
)
from torchslime.utils.meta import Meta
import rich
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
import threading
import multiprocessing

_T = TypeVar('_T')

_T_Launcher = TypeVar('_T_Launcher', bound=Launcher)

class RichLauncher(Launcher):
    
    def get__(self: _T_Launcher) -> Union[_T_Launcher, Nothing]:
        return self if self.is_exec__() else NOTHING

#
# Rich Console Adapter
#

class SlimeConsoleLauncher(Console, RichLauncher): pass
store.builtin__().init__('console_launcher', SlimeConsoleLauncher())
# set rich default console
rich._console = store.builtin__().console_launcher


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
            with self.__t_lock, self.__p_lock, ScopedAttrRestore.m__(console)('file'):
                # set files to the alt console
                for file in store.builtin__().alt_console_files:
                    console.file = file
                    yield console

store.builtin__().init__('alt_console_launcher', SlimeAltConsoleLauncher())

#
# Console Observer
#

@RemoveOverload(checklist=['m__'])
class SlimeConsoleObserver(AttrObserver, Meta):
    
    def m_init__(self) -> None:
        # auto attach observer
        store.builtin__().attach__(self)
    
    @overload
    @classmethod
    def m__(cls: Type[_T]) -> Type[_T]: pass
    
    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None: pass
    
    @AttrObserve
    def console_launcher_observe__(self, new_value: SlimeConsoleLauncher, old_value):
        self.set_console__(new_value)

#
# Rich Live Adapter
#

@RemoveOverload(checklist=['m__'])
class SlimeLiveLauncher(Live, RichLauncher, SlimeConsoleObserver):
    
    def m_init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING
    ) -> None:
        RichLauncher.m_init__(self, launch, exec_ranks)
        SlimeConsoleObserver.m_init__(self)
    
    @overload
    @classmethod
    def m__(
        cls: Type[_T],
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING
    ) -> Type[_T]: pass
    
    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None:
        self.console = __console
    
    def start(self, *args, **kwargs) -> None:
        store.builtin__().attach__(self)
        return super().start(*args, **kwargs)
    
    def stop(self, *args, **kwargs) -> None:
        result = super().stop(*args, **kwargs)
        store.builtin__().detach__(self)
        return result

#
# Rich Logging Adapter
#

class SlimeRichHandler(RichHandler, SlimeConsoleObserver): pass

#
# Renderable Adapter
#

class SlimeProgressLauncher(Progress, MutableBiListItem, RichLauncher, SlimeConsoleObserver):

    @classmethod
    def create__(cls: Type[_T]) -> _T:
        return cls(
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
        store.builtin__().attach__(self)
        return super().start(*args, **kwargs)
    
    def stop(self, *args, **kwargs) -> None:
        result = super().stop(*args, **kwargs)
        store.builtin__().detach__(self)
        return result


class SlimeText(Text, MutableBiListItem): pass

class SlimeTree(Tree, MutableBiListItem): pass

class SlimePanel(Panel, MutableBiListItem): pass


_T_RichRenderable = TypeVar('_T_RichRenderable', bound=Union[RenderableType, MutableBiListItem])

class SlimeGroup(Group, MutableBiListItem, BiList[_T_RichRenderable]):
    
    def __init__(self, *renderables: RenderableType, fit: bool = True) -> None:
        Group.__init__(self, *renderables, fit=fit)
        BiList[_T_RichRenderable].__init__(self)
        self.set_list__(self.renderables)

#
# Custom Rich Components
#

class HandlerProgress(SlimeGroup[_T_RichRenderable]):
    
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


if TYPE_CHECKING:
    from torchslime.core.handlers import Handler

def HandlerTree(handler: "Handler") -> Tree:
    pass

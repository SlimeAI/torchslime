from torchslime.utils.typing import (
    MISSING,
    Iterable,
    Missing,
    NoneOrNothing,
    Pass,
    Union,
    Nothing,
    NOTHING,
    is_none_or_nothing
)
from torchslime.utils.launch import LaunchUtil, Launcher
from torchslime.utils.decorators import ReadonlyAttr
from torchslime.components.store import store
from torchslime.utils.bases import AttrObserver, AttrObserve
from rich import get_console
from rich.progress import Progress
from rich.console import Console
from rich.tree import Tree
import threading
import multiprocessing


@ReadonlyAttr(['console'])
class SlimeConsole(Launcher):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        console: Union[Console, NoneOrNothing, Missing] = MISSING
    ) -> None:
        if exec_ranks is MISSING:
            exec_ranks = [0]
        
        super().__init__(launch, exec_ranks)
        self.console = console if console is not MISSING else get_console()
    
    def get__(self) -> Union[Console, Nothing]:
        console = self.console
        return console if (
            self.is_exec__() and \
            not is_none_or_nothing(console)
        ) else NOTHING

store.builtin__().init__('console', SlimeConsole())


class SlimeAltConsole(SlimeConsole):
    
    __t_lock = threading.Lock()
    __p_lock = multiprocessing.Lock()
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        console: Union[Console, NoneOrNothing, Missing] = MISSING
    ) -> None:
        console = console if console is not MISSING else Console()
        super().__init__(launch, exec_ranks)
    
    def get__(self):
        console = super().get__()
        # check whether the console is exec
        if console is NOTHING:
            yield NOTHING
        else:
            with self.__t_lock, self.__p_lock:
                prev_file = self.console.file
                # set files to the alt console
                for file in store.builtin__().alt_console_files:
                    self.console.file = file
                    yield self.console
                self.console.file = prev_file

store.builtin__().init__('alt_console', SlimeAltConsole())


class SlimeProgress(Launcher, AttrObserver):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        progress: Union[Progress, NoneOrNothing, Missing] = MISSING
    ) -> None:
        if exec_ranks is MISSING:
            exec_ranks = [0]
        
        super().__init__(launch, exec_ranks)
        self.progress = progress if progress is not MISSING else Progress(
            
        )
        
        store.builtin__().attach__(self)
    
    def set__(self, progress: Union[Progress, NoneOrNothing]) -> None:
        self.progress = progress
        self.set_console__(store.builtin__().console.console)
    
    def get__(self) -> Union[Progress, Nothing]:
        progress = self.progress
        return progress if (
            self.is_exec__() and \
            not is_none_or_nothing(progress)
        ) else NOTHING

    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None:
        self.progress.live.console = __console

    @AttrObserve
    def console_observe__(self, new_value: SlimeConsole, old_value):
        # update progress console
        self.set_console__(new_value.console)


class HandlerStructure:
    pass

from .typing import (
    MISSING,
    Iterable,
    Missing,
    NoneOrNothing,
    Pass,
    Union
)
from .launch import LaunchUtil, Launcher
from rich.progress import Progress
from rich.tree import Tree


class HandlerProgress(Launcher):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        progress: Union[Progress, NoneOrNothing, Missing] = MISSING
    ) -> None:
        if exec_ranks is MISSING:
            exec_ranks = [0]
        
        super().__init__(launch, exec_ranks)
        self.__progress = progress
    
    def set_progress():
        pass
    
    def get_progress():
        pass


class HandlerTree(Launcher):
    pass

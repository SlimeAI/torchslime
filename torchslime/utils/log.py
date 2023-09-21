import logging
from logging import Formatter, Filter, Handler, LogRecord, StreamHandler, Logger
from torchslime.components.store import StoreListener, store
from torchslime.components.registry import Registry
import torchslime.utils.cli as Cursor
from .launch import LaunchUtil, launch_util_registry
from .typing import (
    NOTHING,
    is_none_or_nothing,
    Iterable,
    Union,
    NoneOrNothing,
    Pass,
    Callable,
    MISSING,
    Missing,
    Any
)
from .bases import BaseList
from .decorators import Singleton
from bisect import bisect_right
import sys


# initialize log template (if not specified)
if is_none_or_nothing(store.builtin__().log_template):
    store.builtin__().log_template = '{prefix__} - {asctime} - "{filename}:{lineno}" - {message}'
if is_none_or_nothing(store.builtin__().log_template_with_color):
    store.builtin__().log_template_with_color = '{color__}{prefix__} - {asctime} - "{filename}:{lineno}" - {message}'f'{Cursor.single_color("$")}'
if is_none_or_nothing(store.builtin__().log_dateformat):
    store.builtin__().log_dateformat = '%Y/%m/%d %H:%M:%S'


def _get_color(levelno: int) -> str:
    # ``len(color_list)`` should be equal to ``len(level_list) + 1``
    color_list = ['w', 'w', 'g', 'b', 'y', 'r', 'p']
    level_list = [
        logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL
    ]
    return Cursor.single_color(color_list[bisect_right(level_list, levelno)])


class SlimeLogger(Logger):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil] = 'vanilla',
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.set_launch__(launch)
        
        if exec_ranks is MISSING:
            exec_ranks = [0]
        self.set_exec_ranks__(exec_ranks)
    
    def set_launch__(self, launch: Union[str, LaunchUtil]):
        if isinstance(launch, str):
            launch = launch_util_registry.get(launch)()
        self.launch__ = launch
    
    def set_exec_ranks__(self, exec_ranks: Union[Iterable[int], NoneOrNothing, Pass]):
        self.exec_ranks__ = BaseList.create__(exec_ranks)
    
    def is_exec__(self):
        return self.launch__.is_exec(self.exec_ranks__)
    
    def addHandler(self, handler: Handler) -> None:
        super().addHandler(handler)
        if isinstance(handler, SlimeCLIHandler):
            store.builtin__().add_listener__('stderr', handler, init=True)
            if not handler.formatter:
                handler.setFormatter(SlimeCLIFormatter())
        else:
            if not handler.formatter:
                handler.setFormatter(SlimeFormatter())
    
    def removeHandler(self, handler: Handler) -> None:
        super().removeHandler(handler)
        if isinstance(handler, SlimeCLIHandler):
            store.builtin__().remove_listener__('stderr', handler)

#
# Slime Filter
#

class SlimeFilter(Filter):
    
    def filter(self, record: LogRecord) -> bool:
        record.prefix__ = f'[TorchSlime {record.levelname.upper()}]'
        record.color__ = f'{_get_color(record.levelno)}'
        record.rank__ = f'{logger.launch__.get_rank()}'
        return logger.is_exec__()

#
# Slime Formatters
#

class SlimeFormatter(Formatter):

    def __init__(self) -> None:
        super().__init__(
            store.builtin__().log_template,
            store.builtin__().log_dateformat,
            style='{'
        )

class SlimeCLIFormatter(Formatter):
    
    def __init__(self) -> None:
        super().__init__(
            store.builtin__().log_template_with_color,
            store.builtin__().log_dateformat,
            style='{'
        )

@Singleton
class SlimeFormatterListener(StoreListener):
    
    listener_response_registry = Registry[Callable[['SlimeFormatterListener'], None]]('listener_response_registry')

    @listener_response_registry(name='log_template')
    def _update_slime_formatter(self) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeFormatter):
                handler.setFormatter(SlimeFormatter())
    
    @listener_response_registry(name='log_template_with_color')
    def _update_slime_cli_formatter(self) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeCLIFormatter):
                handler.setFormatter((SlimeCLIFormatter()))
    
    @listener_response_registry(name='log_dateformat')
    def _update_all_slime_formatter(self) -> None:
        self._update_slime_formatter()
        self._update_slime_cli_formatter()
    
    def value_change__(self, new_value: Any, old_value: Any, key: str) -> None:
        self.listener_response_registry.get(key, NOTHING)(self)

slime_formatter_listener = SlimeFormatterListener()
for key in ['log_template', 'log_template_with_color', 'log_dateformat']:
    store.builtin__().add_listener__(key, slime_formatter_listener)

#
# Slime CLI Handler
#

class SlimeCLIHandler(StreamHandler, StoreListener):
    
    def __init__(self):
        StreamHandler.__init__(self, sys.stderr)
        StoreListener.__init__(self)
    
    def value_change__(self, new_value: Any, old_value: Any, key: str) -> None:
        self.setStream(new_value)


# initialize logger
logger: SlimeLogger = SlimeLogger(name='builtin__', level=logging.INFO)
logger.propagate = False
logger.addFilter(SlimeFilter())
logger.addHandler(SlimeCLIHandler())


# update stdout and stderr
Cursor.update_store_stdout()
Cursor.update_store_stderr()

import logging
from logging import Formatter, Filter, Handler, LogRecord, Logger
from rich.logging import RichHandler
from torchslime.components.store import store
from torchslime.utils.launch import LaunchUtil, Launcher
from torchslime.utils.typing import (
    NOTHING,
    is_none_or_nothing,
    Iterable,
    Union,
    NoneOrNothing,
    Pass,
    MISSING,
    Missing,
    Any
)
from torchslime.utils.bases import BaseDict, BaseAttrObserver, BaseAttrObserve
from torchslime.utils.decorators import Singleton
import sys


# initialize log template (if not specified)
if is_none_or_nothing(store.builtin__().log_template):
    store.builtin__().log_template = '{prefix__} - {asctime} - "{filename}:{lineno}" - {message}'

if is_none_or_nothing(store.builtin__().log_rich_template):
    store.builtin__().log_rich_template = '{message}'

if is_none_or_nothing(store.builtin__().log_dateformat):
    store.builtin__().log_dateformat = '%Y/%m/%d %H:%M:%S'

#
# Slime Logger
#

class SlimeLogger(Logger, Launcher, BaseAttrObserver):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        *args,
        **kwargs
    ) -> None:
        if exec_ranks is MISSING:
            exec_ranks = [0]
        
        Logger.__init__(self, *args, **kwargs)
        Launcher.__init__(self, launch, exec_ranks)
        BaseAttrObserver.__init__(self)

    def addHandler(self, handler: Handler) -> None:
        if not handler.formatter:
            if isinstance(handler, RichHandler):
                handler.setFormatter(SlimeRichFormatter())
            else:
                handler.setFormatter(SlimeFormatter())
        super().addHandler(handler)

#
# Logger Func Arg Adapter
#

class LoggerKwargs(BaseDict[str, Any]):
    
    def __init__(self, **kwargs):
        # ``stacklevel`` argument was added after py3.8
        if sys.version_info < (3, 8):
            kwargs.pop('stacklevel', NOTHING)
        super().__init__(**kwargs)

#
# Slime Filter
#

class SlimeFilter(Filter):
    
    def filter(self, record: LogRecord) -> bool:
        record.prefix__ = f'[TorchSlime {record.levelname.upper()}]'
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

class SlimeRichFormatter(Formatter):
    
    def __init__(self) -> None:
        super().__init__(
            store.builtin__().log_rich_template,
            store.builtin__().log_dateformat,
            style='{'
        )

@Singleton
class SlimeFormatterObserver(BaseAttrObserver):
    
    @BaseAttrObserve
    def log_template_observe__(self, new_value, old_value) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeFormatter):
                handler.setFormatter(SlimeFormatter())
    
    @BaseAttrObserve
    def log_rich_template_observe__(self, new_value, old_value) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeRichFormatter):
                handler.setFormatter((SlimeRichFormatter()))
    
    @BaseAttrObserve
    def log_dateformat_observe__(self, new_value, old_value) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeFormatter):
                handler.setFormatter(SlimeFormatter())
            if isinstance(handler.formatter, SlimeRichFormatter):
                handler.setFormatter((SlimeRichFormatter()))

slime_formatter_observer = SlimeFormatterObserver()
# set ``init`` to False. ``logger`` instance has not been created here
store.builtin__().subscribe__(slime_formatter_observer, init=False)

#
# initialize logger
#

logger: SlimeLogger = SlimeLogger(name='builtin__', level=logging.INFO)
logger.propagate = False
logger.addFilter(SlimeFilter())
logger.addHandler(RichHandler())

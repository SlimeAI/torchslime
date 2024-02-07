import logging
from logging import Formatter, Filter, Handler, LogRecord, Logger
from torchslime.components.store import store
from torchslime.utils.launch import (
    Launcher,
    LaunchUtil
)
from torchslime.utils.typing import (
    NOTHING,
    Any,
    Union,
    Missing,
    MISSING,
    Iterable,
    NoneOrNothing,
    Pass
)
from torchslime.utils.bases import BaseDict, AttrObserver, AttrObserve
from torchslime.utils.decorators import Singleton
from .rich import RichHandler, SlimeRichHandler
import sys


# initialize log template
store.builtin__().init__('log_template', '{prefix__} - {asctime} - "{filename}:{lineno}" - {message}')
store.builtin__().init__('log_rich_template', '{message}')
store.builtin__().init__('log_dateformat', '%Y/%m/%d %H:%M:%S')

#
# Slime Logger
#

class SlimeLogger(Logger, Launcher, AttrObserver):
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING,
        *args,
        **kwargs
    ) -> None:
        Logger.__init__(self, *args, **kwargs)
        Launcher.__init__(self, launch, exec_ranks)
        AttrObserver.__init__(self)

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
class SlimeFormatterObserver(AttrObserver):
    
    @AttrObserve
    def log_template_observe__(self, new_value, old_value) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeFormatter):
                handler.setFormatter(SlimeFormatter())
    
    @AttrObserve
    def log_rich_template_observe__(self, new_value, old_value) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeRichFormatter):
                handler.setFormatter((SlimeRichFormatter()))
    
    @AttrObserve
    def log_dateformat_observe__(self, new_value, old_value) -> None:
        for handler in logger.handlers:
            if isinstance(handler.formatter, SlimeFormatter):
                handler.setFormatter(SlimeFormatter())
            if isinstance(handler.formatter, SlimeRichFormatter):
                handler.setFormatter((SlimeRichFormatter()))

slime_formatter_observer = SlimeFormatterObserver()
# set ``init`` to False. ``logger`` instance has not been created here
store.builtin__().attach__(slime_formatter_observer, init=False)

#
# initialize logger
#

logger: SlimeLogger = SlimeLogger(name='builtin__', level=logging.INFO)
logger.propagate = False
logger.addFilter(SlimeFilter())
logger.addHandler(SlimeRichHandler(
    console=store.builtin__().console_launcher,
    rich_tracebacks=True
))

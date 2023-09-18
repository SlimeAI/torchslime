from collections.abc import Mapping
from typing import Any
from torchslime.utils import bound_clip
from torchslime.log.common import TerminalLoggerItem, LoggerItem
from torchslime.utils.bases import BaseList
from torchslime.utils.decorators import Singleton
from torchslime.utils.typing import (
    INT_SEQ_N, NOTHING,
    is_none_or_nothing,
    Iterable,
    Union,
    NoneOrNothing,
    Pass,
    PASS
)
from datetime import datetime
from torchslime.utils.typing import Type, Any, Union
import inspect
from inspect import FrameInfo
import os


TIME_FORMAT = '%Y/%m/%d %H:%M:%S'
LOG_FORMAT = '{ts_prefix} - {ts_time} - {ts_exec} - {ts_msg}'


def set_time_format(format: str):
    global TIME_FORMAT
    TIME_FORMAT = format


def set_log_format(format: str):
    global LOG_FORMAT
    LOG_FORMAT = format


from .launch import launch_util_registry


from logging import Formatter, Filter, LogRecord, StreamHandler, Logger
import logging
from .launch import LaunchUtil


class SlimeFilter(Filter):
    
    def filter(self, record: LogRecord) -> bool:
        record.prefix__ = f'[TorchSlime {record.levelname}]'
        record.color__ = f''
        return logger_config.is_exec()


logger = logging.getLogger('builtin__')


@Singleton
class LoggerConfig:
    
    def __init__(
        self,
        launch: Union[str, LaunchUtil] = 'vanilla',
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass] = PASS
    ) -> None:
        self.set_launch(launch)
        self.set_exec_ranks(exec_ranks)
    
    def set_launch(self, launch: Union[str, LaunchUtil]):
        if isinstance(launch, str):
            launch = launch_util_registry.get(launch)()
        self.launch = launch
    
    def set_exec_ranks(self, exec_ranks: Union[Iterable[int], NoneOrNothing, Pass]):
        self.exec_ranks = BaseList.create__(exec_ranks)
    
    def is_exec(self):
        return self.launch.is_exec(self.exec_ranks)


logger_config = LoggerConfig()
stream_handler = StreamHandler()


def setup_logger():
    logger.handlers = []
    logger.filters = []
    logger.propagate = False
    
    logger.addFilter(SlimeFilter())
    logger.addHandler(stream_handler)

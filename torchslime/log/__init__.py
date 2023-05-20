from torchslime.utils import Singleton, BaseList, NOTHING, is_none_or_nothing, bound_clip
from torchslime.log.common import TerminalLoggerItem, LoggerItem
from torchslime.utils.tstype import INT_SEQ_N
from datetime import datetime
from typing import Type, Any, Union
import inspect
from inspect import FrameInfo
import os


INFO_PREFIX = '[TorchSlime INFO]'
WARN_PREFIX = '[TorchSlime WARN]'
ERROR_PREFIX = '[TorchSlime ERROR]'
DEBUG_PREFIX = '[TorchSlime DEBUG]'

TIME_FORMAT = '%Y/%m/%d %H:%M:%S'
LOG_FORMAT = '{ts_prefix} - {ts_time} - {ts_exec} - {ts_msg}'


def set_time_format(format: str):
    global TIME_FORMAT
    TIME_FORMAT = format


def set_log_format(format: str):
    global LOG_FORMAT
    LOG_FORMAT = format


@Singleton
class Logger(BaseList):
    
    def __init__(self):
        super().__init__([TerminalLoggerItem()])
        self.config = {
            'debug': False,
            'info': True,
            'warn': True,
            'error': True,
            'log': True
        }

    def add_logger_item(self, logger_item: LoggerItem):
        if logger_item in self:
            self.warn('Logger item is already in logger.')
            return
        self.append(logger_item)

    def remove_logger_item(self, logger_item: LoggerItem):
        if logger_item not in self:
            self.warn('Logger item is NOT in logger.')
            return
        self.remove(logger_item)

    def debug(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self.config['debug'] is True:
            item = self.format(msg, _exec_info, _frame_offset, DEBUG_PREFIX)
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.debug(item)

    def info(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self.config['info'] is True:
            item = self.format(msg, _exec_info, _frame_offset, INFO_PREFIX)
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.info(item)

    def warn(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self.config['warn'] is True:
            item = self.format(msg, _exec_info, _frame_offset, WARN_PREFIX)
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.warn(item)

    def error(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self.config['error'] is True:
            item = self.format(msg, _exec_info, _frame_offset, ERROR_PREFIX)
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.error(item)

    def log(self, msg):
        if self.config['log'] is True:
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.log(msg)
    
    def format(self, msg, _exec_info, _frame_offset, ts_prefix):
        CALL_OFFSET = 1
        
        _time = self._get_time()
        _exec_info_dict = self._get_exec_info(_frame_offset=_frame_offset + CALL_OFFSET) if is_none_or_nothing(_exec_info) is True else \
            self._format_exec_info(_exec_info['exec_name'], _exec_info['full_exec_name'], _exec_info['lineno'])
        return LOG_FORMAT.format(
            ts_prefix=ts_prefix,
            ts_time=_time,
            ts_exec=_exec_info_dict['ts_exec'],
            ts_exec_full=_exec_info_dict['ts_exec_full'],
            ts_msg=msg
        )
    
    def _get_time(self):
        now = datetime.now()
        return now.strftime(TIME_FORMAT)
    
    def _get_exec_info(self, _frame_offset: int = 0):
        # offset due to function call
        CALL_OFFSET = 2
        
        stack = inspect.stack()
        # true frame offset
        _true_offset = int(bound_clip(CALL_OFFSET + _frame_offset, 0, len(stack) - 1))
        
        frame_info = stack[_true_offset]
        
        # exec_name
        exec_name = self._get_short_exec_name(frame_info)
        # full_exec_name
        full_exec_name = os.path.abspath(frame_info.filename)
        # get lineno
        lineno = frame_info.lineno
        # return short exec info and full exec info
        return self._format_exec_info(exec_name, full_exec_name, lineno)

    def _format_exec_info(self, exec_name, full_exec_name, lineno):
        return {
            'ts_exec': '"{}:{}"'.format(exec_name, lineno),
            'ts_exec_full': '"{}:{}"'.format(full_exec_name, lineno)
        }

    def _get_short_exec_name(self, frame_info: FrameInfo):
        # get cwd and filename path
        cwd = os.path.normpath(os.path.realpath(os.getcwd()))
        filename = os.path.normpath(os.path.realpath(frame_info.filename))
        # get common path to check whether filename is a sub-path of cwd
        try:
            common_path = os.path.commonpath([cwd, filename])
        except Exception:
            common_path = NOTHING
        # workspace module
        if common_path == cwd:
            try:
                exec_name = '{}'.format(os.path.relpath(filename, cwd))
            except Exception:
                exec_name = '{}'.format(os.path.basename(filename))
        # external module
        else:
            module = inspect.getmodule(frame_info.frame)
            exec_name = module.__name__
        return exec_name
    
    def is_distributed(self) -> bool:
        return False


# enable type hint
Logger._wrapped: Type[Logger] = Logger._wrapped


@Singleton
class DistributedLogger(Logger._wrapped):
    
    def __init__(self) -> None:
        super().__init__()
        # default exec_ranks are set to [0]
        self.set_exec_ranks([0])
    
    def set_exec_ranks(self, exec_ranks: INT_SEQ_N):
        self.exec_ranks = BaseList.create(exec_ranks)
    
    def debug(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self._check_exec() is True:
            return super().debug(msg, _exec_info, _frame_offset)
    
    def info(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self._check_exec() is True:
            return super().info(msg, _exec_info, _frame_offset)
    
    def warn(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self._check_exec() is True:
            return super().warn(msg, _exec_info, _frame_offset)
    
    def error(self, msg, _exec_info: dict = NOTHING, _frame_offset: int = 0):
        if self._check_exec() is True:
            return super().error(msg, _exec_info, _frame_offset)
    
    def log(self, msg):
        if self._check_exec() is True:
            return super().log(msg)
    
    def _check_exec(self):
        import torch.distributed as dist
        rank = dist.get_rank()
        return is_none_or_nothing(self.exec_ranks) is False and \
            (self.exec_ranks is ... or rank in self.exec_ranks)
    
    def _get_rank_info(self):
        import torch.distributed as dist
        rank = dist.get_rank()
        return 'RANK {}'.format(rank)
    
    def _get_exec_info(self, _frame_offset: int = 0):
        # set total frame offset
        SUB_CLASS_OFFSET = 1
        OVERRIDE_METHOD_OFFSET = 1
        _total_frame_offset = _frame_offset + SUB_CLASS_OFFSET + OVERRIDE_METHOD_OFFSET
        
        rank_info = self._get_rank_info()
        exec_info_dict = {}
        for key, value in super()._get_exec_info(_total_frame_offset).items():
            exec_info_dict[key] = '{} - {}'.format(value, rank_info)
        return exec_info_dict
    
    def is_distributed(self) -> bool:
        return True


@Singleton
class LoggerProxy:

    def __init__(self) -> None:
        self._logger: Union[Logger, DistributedLogger] = Logger()
    
    def __getattr__(self, __name: str) -> Any:
        if __name == '_logger':
            return super().__getattr__(__name)
        return self._logger.__getattr__(__name)
    
    def __getattribute__(self, __name: str) -> Any:
        if __name == '_logger':
            return super().__getattribute__(__name)
        return self._logger.__getattribute__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == '_logger':
            return super().__setattr__(__name, __value)
        return self._logger.__setattr__(__name, __value)
    
    def __getitem__(self, __i) -> Any:
        return self._logger.__getitem__(self, __i)
    
    def __setitem__(self, __i, __v) -> None:
        return self._logger.__setitem__(self, __i, __v)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._logger.__call__(self, *args, **kwargs)


# use ``Union[Logger, DistributedLogger, LoggerProxy]`` to enable type hint
logger: Union[Logger, DistributedLogger, LoggerProxy] = LoggerProxy()


def set_logger(_logger: Logger._wrapped):
    logger._logger = _logger

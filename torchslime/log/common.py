from torchslime.util.terminal import single_color
from torchslime.experiment.directory import get_log_path
from torchslime.util import Nothing, NOTHING, is_none_or_nothing
from typing import Union
import sys

INFO_PREFIX = '[TorchSlime INFO]'
WARN_PREFIX = '[TorchSlime WARN]'
ERROR_PREFIX = '[TorchSlime ERROR]'
DEBUG_PREFIX = '[TorchSlime DEBUG]'

TIME_FORMAT = '%Y/%m/%d %H:%M:%S'


def set_time_format(format: str):
    global TIME_FORMAT
    TIME_FORMAT = format


class LoggerItem:
    
    def debug(self, msg: str, time: str, _exec: str):
        pass
    
    def info(self, msg: str, time: str, _exec: str):
        pass
    
    def warn(self, msg: str, time: str, _exec: str):
        pass
    
    def error(self, msg: str, time: str, _exec: str):
        pass

    def log(self, msg: str):
        pass
    
    def remove_self(self):
        from torchslime.log import logger
        logger.remove_logger_item(self)
    
    def format(self, msg: str, time: str, _exec: str):
        return ' - '.join([time, _exec, msg])


class TerminalLogger(LoggerItem):
    
    def debug(self, msg: str, time: str, _exec: str):
        print('{}{}{}'.format(
            single_color('g'), ' - '.join([DEBUG_PREFIX, self.format(msg, time, _exec)]), single_color('w')
        ))
    
    def info(self, msg: str, time: str, _exec: str):
        print('{}{}{}'.format(
            single_color('b'), ' - '.join([INFO_PREFIX, self.format(msg, time, _exec)]), single_color('w')
        ))
    
    def warn(self, msg: str, time: str, _exec: str):
        print('{}{}{}'.format(
            single_color('y'), ' - '.join([WARN_PREFIX, self.format(msg, time, _exec)]), single_color('w')
        ))
    
    def error(self, msg: str, time: str, _exec: str):
        print('{}{}{}'.format(
            single_color('r'), ' - '.join([ERROR_PREFIX, self.format(msg, time, _exec)]), single_color('w')
        ))

    def log(self, msg: str):
        file = sys.stdout
        file.write(msg)
        file.flush()


class FileLogger(LoggerItem):
    
    def __init__(self, log_path: Union[str, Nothing] = NOTHING) -> None:
        super().__init__()
        self.log_path = get_log_path() if is_none_or_nothing(log_path) is True else log_path
    
    def debug(self, msg: str, time: str, _exec: str):
        with open(self.log_path, 'a') as f:
            f.write(' - '.join([DEBUG_PREFIX, self.format(msg, time, _exec)]))
            f.write('\n')
    
    def info(self, msg: str, time: str, _exec: str):
        with open(self.log_path, 'a') as f:
            f.write(' - '.join([INFO_PREFIX, self.format(msg, time, _exec)]))
            f.write('\n')
    
    def warn(self, msg: str, time: str, _exec: str):
        with open(self.log_path, 'a') as f:
            f.write(' - '.join([WARN_PREFIX, self.format(msg, time, _exec)]))
            f.write('\n')
    
    def error(self, msg: str, time: str, _exec: str):
        with open(self.log_path, 'a') as f:
            f.write(' - '.join([ERROR_PREFIX, self.format(msg, time, _exec)]))
            f.write('\n')

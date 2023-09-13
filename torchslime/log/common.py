from torchslime.utils.typing import Nothing
from torchslime.utils.cli import single_color
from torchslime.experiment.directory import get_log_path
from torchslime.utils.typing import NOTHING, is_none_or_nothing
from torchslime.utils.typing import Union
import sys


class LoggerItem:
    
    def debug(self, item):
        pass
    
    def info(self, item):
        pass
    
    def warn(self, item):
        pass
    
    def error(self, item):
        pass

    def log(self, item):
        pass
    
    def remove_self(self):
        from torchslime.log import logger
        logger.remove_logger_item(self)


class TerminalLoggerItem(LoggerItem):
    
    def debug(self, item):
        print(f'{single_color("g")}{item}{single_color("w")}')
    
    def info(self, item):
        print(f'{single_color("b")}{item}{single_color("w")}')
    
    def warn(self, item):
        print(f'{single_color("y")}{item}{single_color("w")}')
    
    def error(self, item):
        print(f'{single_color("r")}{item}{single_color("w")}')

    def log(self, item):
        file = sys.stdout
        file.write(item)
        file.flush()


class FileLoggerItem(LoggerItem):
    
    def __init__(self, log_path: Union[str, Nothing] = NOTHING) -> None:
        super().__init__()
        self.log_path = get_log_path() if is_none_or_nothing(log_path) is True else log_path
    
    def debug(self, item):
        self.append_log_file(item)
    
    def info(self, item):
        self.append_log_file(item)
    
    def warn(self, item):
        self.append_log_file(item)
    
    def error(self, item):
        self.append_log_file(item)
    
    def append_log_file(self, item):
        with open(self.log_path, 'a') as f:
            f.write(item)
            f.write('\n')

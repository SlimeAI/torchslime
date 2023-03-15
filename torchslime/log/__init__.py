from torchslime.util import Singleton, BaseList, NOTHING, is_none_or_nothing
from torchslime.log.common import set_time_format, TIME_FORMAT, TerminalLogger, LoggerItem
from datetime import datetime
import inspect
import os


@Singleton
class Logger(BaseList):
    
    def __init__(self):
        super().__init__([TerminalLogger()])
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

    def info(self, msg, _exec: dict = NOTHING, _frame_offset: int = 0):
        if self.config['info'] is True:
            time = self._get_time()
            _exec = self._get_exec(_exec=_exec, _frame_offset=_frame_offset)
            
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.info(msg, time, _exec)

    def warn(self, msg, _exec: dict = NOTHING, _frame_offset: int = 0):
        if self.config['warn'] is True:
            time = self._get_time()
            _exec = self._get_exec(_exec=_exec, _frame_offset=_frame_offset)
            
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.warn(msg, time, _exec)

    def error(self, msg, _exec: dict = NOTHING, _frame_offset: int = 0):
        if self.config['error'] is True:
            time = self._get_time()
            _exec = self._get_exec(_exec=_exec, _frame_offset=_frame_offset)
            
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.error(msg, time, _exec)

    def debug(self, msg, _exec: dict = NOTHING, _frame_offset: int = 0):
        if self.config['debug'] is True:
            time = self._get_time()
            _exec = self._get_exec(_exec=_exec, _frame_offset=_frame_offset)
            
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.debug(msg, time, _exec)

    def log(self, msg):
        if self.config['log'] is True:
            for logger_item in self:
                logger_item: LoggerItem = logger_item
                logger_item.log(msg)
    
    def _get_time(self):
        now = datetime.now()
        return now.strftime(TIME_FORMAT)
    
    def _get_exec(self, _exec: dict = NOTHING, _frame_offset: int = 0):
        if is_none_or_nothing(_exec) is False:
            if 'exec_name' in _exec and 'lineno' in _exec:
                return '{}, line {}'.format(_exec['exec_name'], _exec['lineno'])
        
        stack = inspect.stack()
        frame = stack[min(2 + _frame_offset, len(stack) - 1)]
        # get cwd and filename path
        cwd = os.path.normpath(os.path.realpath(os.getcwd()))
        filename = os.path.normpath(os.path.realpath(frame.filename))
        # get common path to check whether filename is a sub-path of cwd
        try:
            common_path = os.path.commonpath([cwd, filename])
        except Exception:
            common_path = NOTHING
        # workspace module
        if common_path == cwd:
            # use double quotes here to enable quick jump in some code editors(e.g. VSCode -> Ctrl + click)
            try:
                exec_name = '"{}"'.format(os.path.relpath(filename, cwd))
            except Exception:
                exec_name = '"{}"'.format(os.path.basename(filename))
        # external module
        else:
            # DO NOT use quotes here to illustrate that the caller is an external module
            module = inspect.getmodule(frame.frame)
            exec_name = module.__name__
        # get lineno
        lineno = frame.lineno
        
        return '{}, line {}'.format(exec_name, lineno)


logger = Logger()

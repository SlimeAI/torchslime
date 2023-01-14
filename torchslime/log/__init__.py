from ..util import Singleton
import sys


color_dict = {
    'r': 31,  # red
    'g': 32,  # green
    'y': 33,  # yellow
    'b': 34,  # blue
    'p': 35,  # purple
    'c': 36,  # cyan
    'w': 37  # white
}

info_prefix = '[torchslime INFO]'
warn_prefix = '[torchslime WARN]'
error_prefix = '[torchslime ERROR]'
debug_prefix = '[torchslime DEBUG]'


def color_format(*args, color: str, sep: str = ' '):
    color_prefix = '\033[%dm' % color_dict.get(color, 38)
    color_suffix = '\033[0m'
    return '%s%s%s' % (color_prefix, sep.join(str(arg) for arg in args), color_suffix)


@Singleton
class Logger:

    # TODO: 如果同时想要文件输出怎么设计
    def __init__(self):
        self._control = {
            'info': True,
            'warn': True,
            'error': True,
            'debug': False
        }

    def info(self, *args):
        self.output(info_prefix, *args, type='info', color='b')

    def warn(self, *args):
        self.output(warn_prefix, *args, type='warn', color='y')

    def error(self, *args):
        self.output(error_prefix, *args, type='error', color='r')

    def debug(self, *args):
        self.output(debug_prefix, *args, type='debug', color='g')

    def log(self, *args, **kwargs):
        print(*args, **kwargs)

    def output(self, *args, type: str, color: str = 'w'):
        if self._control.get(type, False) is True:
            print(color_format(*args, color=color))


logger = Logger()

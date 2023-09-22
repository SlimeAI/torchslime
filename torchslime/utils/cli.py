import sys
from .decorators import ContextDecoratorBinding
from .typing import (
    Union,
    TextIO,
    AnyStr,
    List,
    IO,
    NoneOrNothing,
    MISSING
)
from .bases import BaseProxy, BaseGenerator
from torchslime.components.store import store, BuiltinStoreSet
from io import TextIOWrapper

#
# ``sys.stdout and sys.stderr`` operations
#

def update_store_stdout():
    store.builtin__().stdout = sys.stdout

def set_stdout(stdout: Union[TextIO, NoneOrNothing]):
    sys.stdout = stdout
    update_store_stdout()

def update_store_stderr():
    store.builtin__().stderr = sys.stderr

def set_stderr(stderr: Union[TextIO, NoneOrNothing]):
    sys.stderr = stderr
    update_store_stderr()


@ContextDecoratorBinding
class set_cli_interceptor:
    
    def __init__(
        self,
        __out: Union['CLIInterceptor', NoneOrNothing] = None,
        __err: Union['CLIInterceptor', NoneOrNothing] = None,
        *,
        restore_out: bool = True,
        restore_err: bool = True
    ) -> None:
        # stdout
        self.prev_out = sys.stdout
        self.restore_out = restore_out
        if __out is not None:
            set_stdout(__out)
        # stderr
        self.prev_err = sys.stderr
        self.restore_err = restore_err
        if __err is not None:
            set_stderr(__err)
    
    def __enter__(self) -> None: pass
    
    def __exit__(self, *args, **kwargs) -> None:
        if self.restore_out:
            set_stdout(self.prev_out)
        if self.restore_err:
            set_stderr(self.prev_err)

class CLIInterceptor(TextIO, BaseProxy[TextIOWrapper]):
    
    __slots__ = ('text_io_wrapper', 'adapter_list')
    
    def __init__(
        self,
        text_io_wrapper: TextIOWrapper,
    ) -> None:
        TextIO.__init__(self)
        BaseProxy.__init__(self, text_io_wrapper, [
            'buffer', 'encoding', 'errors', 'line_buffering', 'newlines',
            'mode', 'name', 'closed', 'close', 'fileno', 'flush', 'isatty',
            'read', 'readable', 'readline', 'readlines', 'seek', 'seekable',
            'tell', 'truncate', 'writable'
        ])
    
    def write(self, s: AnyStr) -> int:
        gen = BaseGenerator(self.output_control__())
        # before output
        state = gen()
        # output
        if state:
            result = self.obj__.write(s)
        # after output
        gen()
        return result

    def writelines(self, lines: List[AnyStr]) -> None:
        gen = BaseGenerator(self.output_control__())
        # before output
        state = gen()
        # output
        if state:
            result = self.obj__.writelines(lines)
        # after output
        gen()
        return result
    
    def output_control__(self):
        if store.builtin__().prev_refresh and not store.builtin__().refresh_state:
            self.obj__.write('\n')
        yield True
        store.builtin__().prev_refresh = store.builtin__().refresh_state
    
    # adapter
    def __enter__(self) -> IO[AnyStr]: return self.obj__.__enter__()
    def __exit__(self, type, value, traceback) -> None: return self.obj__.__exit__(type, value, traceback)


ESC = '\x1b'  # the ANSI escape code.
CSI = ESC + '['  # Control Sequence Introducer.
CURSOR_UP = CSI + '{}A'
CURSOR_DOWN = CSI + '{}B'
CURSOR_LEFT = CSI + '{}C'
CURSOR_RIGHT = CSI + '{}D'
CURSOR_START = '\r'  # move the cursor to the start of the row.
CURSOR_INVISIBLE = CSI + '?25l'
CURSOR_VISIBLE = CSI + '?25h'
CLEAR_SCREEN = CSI + '{}J'
CLEAR_LINE = CSI + '{}K'

CURSOR_VISIBILITY_ENABLED: bool = True


def set_cursor_visibility_enabled(enabled: bool):
    global CURSOR_VISIBILITY_ENABLED
    CURSOR_VISIBILITY_ENABLED = enabled


def up(row: int = 1):
    return CURSOR_UP.format(row)


def down(row: int = 1):
    return CURSOR_DOWN.format(row)


def left(column: int = 1):
    return CURSOR_LEFT.format(column)


def right(column: int = 1):
    return CURSOR_RIGHT.format(column)


def start():
    return CURSOR_START


def clear_line(mode: str = 'all'):
    clear_mode = {
        'after': 0,
        'before': 1,
        'all': 2
    }
    return CLEAR_LINE.format(clear_mode.get(mode, 2))


def single_color(color: str):
    color_dict = {
        'r': 31,  # red
        'g': 32,  # green
        'y': 33,  # yellow
        'b': 34,  # blue
        'p': 35,  # purple
        'c': 36,  # cyan
        'w': 37,  # white
        '$': 0  # clear
    }
    return CSI + str(color_dict.get(color, 37)) + 'm'


def reset_style():
    return CSI + str(0) + 'm'


def execute(*commands, file=sys.stdout):
    """
    Execute cursor commands.
    """
    file.write(''.join(commands))
    file.flush()


def refresh_print(
    *contents,
    sep: str = ' ',
    file=MISSING,
    end='',
    cursor_location: str = ''
):
    """
    Execute refresh command and output contents.
    """
    file = sys.stderr if file is MISSING else file
    if store.builtin__().prev_refresh:
        execute(cursor_location, start(), clear_line())
    with BuiltinStoreSet('refresh_state', True):
        file.write(sep.join(contents) + end)
        file.flush()
    # set ``prev_refresh`` to ``True``, still work for original stdout and stderr
    store.builtin__().prev_refresh = True


def multi_lines(lines):
    class MultiLineCursor:
        def __enter__(self):
            pass

        def __exit__(self):
            pass
    return MultiLineCursor()


def cursor_invisible(file=sys.stdout):
    """
    Make cursor invisible to avoid cursor flickering.
    """
    class InvisibleCursor:
        def __init__(self, file=sys.stdout) -> None:
            self.file = file
        
        def __enter__(self):
            if CURSOR_VISIBILITY_ENABLED:
                execute(CURSOR_INVISIBLE, file=self.file)

        def __exit__(self, *_):
            if CURSOR_VISIBILITY_ENABLED:
                execute(CURSOR_VISIBLE, file=self.file)
    return InvisibleCursor(file=file)

import sys


ESC = '\x1b'  # the ANSI escape code.
CSI = ESC + '['  # control Sequence Introducer.
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
        'w': 37  # white
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
    file=sys.stdout,
    end='',
    cursor_location: str = ''
):
    """
    Execute refresh command and output contents.
    """
    execute(cursor_location, start(), clear_line())
    file.write(sep.join(contents) + end)
    file.flush()


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

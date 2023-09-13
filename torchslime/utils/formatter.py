"""defines some format functions for log output.
"""
from .typing import NOTHING, Tuple, Union, Mapping, Sequence, is_none_or_nothing
import time
from . import cli as Cursor
from .typing import Nothing
from torchslime.components.store import store
from textwrap import indent


class ProgressStyle:

    def __init__(
        self,
        left_sep,
        finished,
        next,
        unfinished,
        right_sep,
        finished_color: str = 'w',
        all_finished_color: str = 'w',
        next_color: str = 'w',
        unfinished_color: str = 'w'
    ) -> None:
        self.left_sep = left_sep
        self.finished = finished
        self.next = next
        self.unfinished = unfinished
        self.right_sep = right_sep

        convert_color = lambda item: Cursor.single_color(item) if isinstance(item, str) and len(item) == 1 else item
        
        self.finished_color = convert_color(finished_color)
        self.all_finished_color = convert_color(all_finished_color)
        self.next_color = convert_color(next_color)
        self.unfinished_color = convert_color(unfinished_color)


progress_style = {
    'cube': ProgressStyle(chr(0x007c), chr(0x2588), chr(0x0020), chr(0x0020), chr(0x007c), 'b', 'g', 'w', 'w'),  # style: |███   |
    'line': ProgressStyle(chr(0x007c), chr(0x2501), chr(0x2578), chr(0x2501), chr(0x007c), 'p', 'g', 'p', 'w'),  # style: |━━━╸━━|
    'arrow': ProgressStyle(chr(0x005b), chr(0x003d), chr(0x003e), chr(0x002d), chr(0x005d), 'b', 'g', 'b', 'w')  # style: |===>--|
}


def progress_format(
    progress: Tuple[int, int],
    percentage: bool = True,
    proportion: bool = True,
    length: int = 25,
    style: Union[str, ProgressStyle] = 'cube',
    newline: bool = True
):
    """
    Format a progress bar output.
    """
    current, total = progress[0] + 1, progress[1]
    p_style = progress_style[style] if isinstance(style, str) else style
    output = ''
    if percentage is True:
        output += f'{int(current * 100 / total):>3}%'
    
    finished_length = int(current * length / total)
    reset = Cursor.reset_style()
    finished_color = p_style.finished_color if finished_length < length else p_style.all_finished_color
    next_color = p_style.next_color
    unfinished_color = p_style.unfinished_color
    
    output += p_style.left_sep
    output += finished_color + finished_length * p_style.finished + reset
    if finished_length < length:
        output += next_color + p_style.next + reset
    if finished_length < length - 1:
        output += unfinished_color + (length - finished_length - 1) * p_style.unfinished + reset
    output += p_style.right_sep

    if proportion is True:
        output += f' {int(current)}/{int(total)}'
    
    if current >= total and newline:
        output += '\n'
    return output


def period_time_format(_time: float) -> str:
    if _time < 0:
        return '--'
    # parse to int
    _time = int(_time)

    m, s = divmod(_time, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f'{h}:{m:0>2}:{s:0>2}'
    elif m > 0:
        return f'{m:0>2}:{s:0>2}'
    else:
        return f'{s}s'


def eta_format(from_time, remain_steps, to_time: Union[str, float] = 'now'):
    """
    Format an Estimated-Time-of-Arrival(eta) output.
    """
    if to_time == 'now':
        to_time = time.time()
    return period_time_format((to_time - from_time) * remain_steps)

#
# dict and list formatter
#

def dict_to_key_value_str(
    __dict: Mapping,
    key_value_sep: str = '=',
    str_sep: str = ', '
) -> str:
    return str_sep.join(dict_to_key_value_str_list(__dict, key_value_sep=key_value_sep))


def dict_to_key_value_str_list(
    __dict: Mapping,
    key_value_sep: str = '='
) -> list:
    return [f'{key}{key_value_sep}{value}' for key, value in __dict.items()]


def concat_format(
    __left: str,
    __content: Sequence[str],
    __right: str,
    *,
    item_sep: str = ',',
    indent_prefix: Union[str, None, Nothing] = NOTHING,
    break_line: bool = True
) -> str:
    if len(__content) < 1:
        # empty content: simply concat ``__left`` and ``__right``
        return __left + __right
    
    break_line_sep = '\n'
    if not break_line:
        indent_prefix = ''
    elif is_none_or_nothing(indent_prefix):
        indent_prefix: str = store.builtin__().indent_str
    # format content
    content_sep = item_sep + (break_line_sep if break_line else '')
    __content = indent(content_sep.join(__content), prefix=indent_prefix)
    # format concat
    concat_sep = break_line_sep if break_line else ''
    return concat_sep.join([__left, __content, __right])

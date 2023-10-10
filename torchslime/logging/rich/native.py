"""
Rich Native Utils
"""
from torchslime.components.store import store
from torchslime.utils.typing import (
    MISSING,
    Iterable,
    Missing,
    NoneOrNothing,
    is_none_or_nothing,
    Pass,
    Union,
    Nothing,
    NOTHING,
    TypeVar,
    overload,
    Type,
    TYPE_CHECKING,
    Callable
)
from torchslime.utils.launch import LaunchUtil, Launcher
from torchslime.utils.decorators import RemoveOverload
from torchslime.utils.bases import (
    AttrObserver,
    AttrObserve,
    ScopedAttrRestore,
    BiList,
    MutableBiListItem,
    CompositeBFT,
    BaseList
)
from torchslime.utils.meta import Meta
import rich
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.console import Console, RenderableType, Group
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.logging import RichHandler
from rich.table import Table
from rich.protocol import is_renderable
import threading
import multiprocessing

_T = TypeVar('_T')

_T_Launcher = TypeVar('_T_Launcher', bound=Launcher)

class RichLauncher(Launcher):
    
    def get__(self: _T_Launcher) -> Union[_T_Launcher, Nothing]:
        return self if self.is_exec__() else NOTHING

#
# Rich Console Adapter
#

class SlimeConsoleLauncher(Console, RichLauncher): pass
store.builtin__().init__('console_launcher', SlimeConsoleLauncher())
# set rich default console
rich._console = store.builtin__().console_launcher


class SlimeAltConsoleLauncher(SlimeConsoleLauncher):
    
    __t_lock = threading.Lock()
    __p_lock = multiprocessing.Lock()
    
    def get__(self):
        console = super().get__()
        # check whether the console is exec
        if console is NOTHING:
            # use ``yield from`` rather than ``yield`` to create an empty generator
            yield from NOTHING
        else:
            with self.__t_lock, self.__p_lock, ScopedAttrRestore.m__(console)('file'):
                # set files to the alt console
                for file in store.builtin__().alt_console_files:
                    console.file = file
                    yield console

store.builtin__().init__('alt_console_launcher', SlimeAltConsoleLauncher(
    color_system=None,
    force_terminal=False,
    force_jupyter=False,
    force_interactive=False
))


def yield_console(
    enable_console: bool = True,
    enable_alt_console: bool = False
):
    if enable_console:
        console_launcher = store.builtin__().console_launcher.get__()
        yield from console_launcher if console_launcher is NOTHING else [console_launcher]
    
    if enable_alt_console:
        yield from store.builtin__().alt_console_launcher.get__()

#
# Console Observer
#

@RemoveOverload(checklist=['m__'])
class SlimeConsoleObserver(AttrObserver, Meta):
    
    def m_init__(self) -> None:
        # auto attach observer
        store.builtin__().attach__(self)
    
    @overload
    @classmethod
    def m__(cls: Type[_T]) -> Type[_T]: pass
    
    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None: pass
    
    @AttrObserve
    def console_launcher_observe__(self, new_value: SlimeConsoleLauncher, old_value):
        self.set_console__(new_value)

#
# Rich Live Adapter
#

@RemoveOverload(checklist=['m__'])
class SlimeLiveLauncher(Live, RichLauncher, SlimeConsoleObserver):
    
    def m_init__(
        self,
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING
    ) -> None:
        RichLauncher.m_init__(self, launch, exec_ranks)
        SlimeConsoleObserver.m_init__(self)
    
    @overload
    @classmethod
    def m__(
        cls: Type[_T],
        launch: Union[str, LaunchUtil, Missing] = MISSING,
        exec_ranks: Union[Iterable[int], NoneOrNothing, Pass, Missing] = MISSING
    ) -> Type[_T]: pass
    
    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None:
        self.console = __console
    
    def start(self, *args, **kwargs) -> None:
        store.builtin__().attach__(self)
        return super().start(*args, **kwargs)
    
    def stop(self, *args, **kwargs) -> None:
        result = super().stop(*args, **kwargs)
        store.builtin__().detach__(self)
        return result

#
# Rich Logging Adapter
#

class SlimeRichHandler(RichHandler, SlimeConsoleObserver): pass

#
# Renderable Adapter
#

class SlimeProgressLauncher(Progress, MutableBiListItem, RichLauncher, SlimeConsoleObserver):

    @classmethod
    def create__(cls: Type[_T]) -> _T:
        return cls(
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )

    def set_console__(self, __console: Union[Console, NoneOrNothing]) -> None:
        self.live.console = __console
    
    def start(self, *args, **kwargs) -> None:
        store.builtin__().attach__(self)
        return super().start(*args, **kwargs)
    
    def stop(self, *args, **kwargs) -> None:
        result = super().stop(*args, **kwargs)
        store.builtin__().detach__(self)
        return result


class SlimeText(Text, MutableBiListItem): pass

class SlimeTree(Tree, MutableBiListItem): pass

class SlimePanel(Panel, MutableBiListItem): pass

class SlimeTable(Table, MutableBiListItem): pass


_T_RichRenderable = TypeVar('_T_RichRenderable', bound=Union[RenderableType, MutableBiListItem])

class SlimeGroup(Group, MutableBiListItem, BiList[_T_RichRenderable]):
    
    def __init__(self, *renderables: RenderableType, fit: bool = True) -> None:
        Group.__init__(self, *renderables, fit=fit)
        BiList[_T_RichRenderable].__init__(self)
        self.set_list__(self.renderables)

#
# Custom Rich Components
#

class ProfileProgress(SlimeGroup[_T_RichRenderable]):
    
    def __init__(
        self,
        __progress: Union[SlimeProgressLauncher, Missing] = MISSING,
        __text: Union[str, SlimeText, Missing] = MISSING,
        *,
        fit: bool = True
    ) -> None:
        self.progress = self._parse_progress(__progress)
        self.text = self._parse_text(__text)
        
        super().__init__(
            self.progress,
            self.text,
            fit=fit
        )
    
    def _parse_progress(
        self,
        __progress: Union[SlimeProgressLauncher, Missing] = MISSING
    ) -> SlimeProgressLauncher:
        if __progress is MISSING:
            __progress = SlimeProgressLauncher.create__()
        return __progress
    
    def set_progress__(self, __progress: Union[SlimeProgressLauncher, Missing] = MISSING):
        __progress = self._parse_progress(__progress)
        self.progress.replace_self__(__progress)
        self.progress = __progress
    
    def _parse_text(
        self,
        __text: Union[str, SlimeText, Missing] = MISSING
    ) -> SlimeText:
        if __text is MISSING:
            __text = SlimeText("")
        elif isinstance(__text, str):
            __text = SlimeText(__text)
        return __text
    
    def set_text__(self, __text: Union[str, SlimeText, Missing] = MISSING):
        __text = self._parse_text(__text)
        self.text.replace_self__(__text)
        self.text = __text


if TYPE_CHECKING:
    from torchslime.core.handlers import Handler

class HandlerTreeProfiler:
    
    def handler_profile(
        self,
        handler: "Handler",
        display_meta: bool = True,
        display_attr: bool = True,
        target_handlers: Union[Iterable["Handler"], NoneOrNothing] = NOTHING,
        wrap_func: Union[str, Callable[[Group, "Handler"], Group], NoneOrNothing] = NOTHING
    ) -> Group:
        renderables = [
            Text(handler.get_class_name(), style='bold blue')
        ]
        if display_meta:
            meta = handler.get_meta_dict()
            if len(meta) >= 1:
                meta_table = Table()
                meta_table.add_column('Meta', style='cyan')
                meta_table.add_column('Value', style='green')
                
                for key, value in meta.items():
                    meta_table.add_row(
                        parse_renderable(key),
                        parse_renderable(value)
                    )
                
                renderables.append(meta_table)
        
        if display_attr:
            attr = handler.get_attr_dict()
            if len(attr) >= 1:
                attr_table = Table()
                attr_table.add_column('Attr', style='cyan')
                attr_table.add_column('Value', style='green')
                
                for key, value in attr.items():
                    attr_table.add_row(
                        parse_renderable(key),
                        parse_renderable(value)
                    )
                
                renderables.append(attr_table)
        
        group = Group(*renderables)
        if not self.check_target_handler(handler, target_handlers):
            return group
        
        wrap_func = self.get_handler_profile_wrap_func(wrap_func)
        if is_none_or_nothing(wrap_func):
            from torchslime.logging.logger import logger
            logger.warning(
                'Handler profile wrap func is ``None`` or ``NOTHING``, '
                'and it will do nothing during display.'
            )
            return group
        return wrap_func(group, handler)
    
    def profile(
        self,
        handler: "Handler",
        display_meta: bool = True,
        display_attr: bool = True,
        target_handlers: Union[Iterable["Handler"], NoneOrNothing] = NOTHING,
        wrap_func: Union[str, Callable[[Group, "Handler"], Group], NoneOrNothing] = NOTHING
    ) -> Tree:
        root = Tree(self.handler_profile(handler))
        queue = [
            root
        ]
        
        def visit(node: "Handler"):
            tree = queue.pop(0)
            for child in node.composite_iterable__():
                new_tree = tree.add(self.handler_profile(
                    child,
                    display_meta=display_meta,
                    display_attr=display_attr,
                    target_handlers=target_handlers,
                    wrap_func=wrap_func
                ))
                queue.append(new_tree)
        
        CompositeBFT(handler, visit)
        return root
    
    def check_target_handler(
        self,
        handler: "Handler",
        target_handlers: Union[Iterable["Handler"], NoneOrNothing] = NOTHING
    ) -> bool:
        return handler in BaseList.create__(
            target_handlers,
            return_none=False,
            return_nothing=False,
            return_pass=False
        )
    
    def get_handler_profile_wrap_func(
        self,
        wrap_func: Union[str, Callable[[Group, "Handler"], Group], NoneOrNothing] = NOTHING
    ) -> Union[Callable[[Group, "Handler"], Group], NoneOrNothing]:
        if isinstance(wrap_func, str):
            return handler_profile_wrap_func.get(wrap_func, NOTHING)
        return wrap_func


from torchslime.components.registry import Registry
handler_profile_wrap_func = Registry[Callable[[Group, "Handler"], Group]]('handler_profile_wrap_func')

@handler_profile_wrap_func(name='exception')
def _exception_wrap(group: Group, handler: "Handler") -> Group:
    _separator_len = 10
    # ×  <---------- EXCEPTION Here ----------
    _exception_indicator = f' {chr(0x00D7)}  <{"-" * _separator_len} EXCEPTION Here {"-" * _separator_len}'
    original_text = group.renderables[0]
    group.renderables[0] = Text.assemble(original_text, Text(_exception_indicator, style='bold red'))
    return group

@handler_profile_wrap_func(name='terminate')
def _terminate_wrap(group: Group, handler: "Handler") -> Group:
    _separator_len = 10
    # ||---------- Handler TERMINATE Here ----------||
    _terminate_indicator = f' ||{"-" * _separator_len} Handler TERMINATE Here {"-" * _separator_len}||'
    original_text = group.renderables[0]
    group.renderables[0] = Text.assemble(original_text, Text(_terminate_indicator, style='bold green'))
    return group


class RenderInterface:
    def render__(self) -> RenderableType: pass

def is_render_interface(item) -> bool:
    return isinstance(item, RenderInterface) or hasattr(item, 'render__')


def parse_renderable(item) -> RenderableType:
    # NOTE: ``NOTHING`` will be inspected as ``renderable``, because 
    # ``hasattr(NOTHING, '__rich__')`` and ``hasattr(NOTHING, '__rich_console__')`` 
    # will both return True. However, it should not actually be seen as ``renderable``.
    if item is NOTHING:
        return str(item)
    elif is_render_interface(item):
        item: RenderInterface
        return item.render__()
    elif is_renderable(item):
        return item
    else:
        return str(item)


if TYPE_CHECKING:
    from torchslime.core.handlers.wrappers import HandlerWrapper, HandlerWrapperContainer

class HandlerWrapperContainerProfiler:
    
    def wrapper_profile(
        self,
        wrapper: Union["HandlerWrapper", "HandlerWrapperContainer"]
    ) -> str:
        from torchslime.utils.common import dict_to_key_value_str_list, concat_format
        
        class_name = wrapper.get_class_name()
        
        meta_display_list = dict_to_key_value_str_list(wrapper.get_meta_dict())
        meta = concat_format('[', meta_display_list, ']', item_sep=', ')
        
        attr_display_list = dict_to_key_value_str_list(wrapper.get_attr_dict())
        attr = concat_format('(', attr_display_list, ')', item_sep=', ')
        
        return f'{class_name}{meta}{attr}'
    
    def profile(self, handler_wrapper_container: "HandlerWrapperContainer") -> RenderableType:
        table = Table(show_lines=True)        
        table.add_column('index')
        table.add_column('wrapper/container')

        table.add_row('[bold]Container', f'[bold]{self.wrapper_profile(handler_wrapper_container)}')

        for index, handler_wrapper in enumerate(handler_wrapper_container):
            table.add_row(str(index), self.wrapper_profile(handler_wrapper))
        
        return table

from torchslime.utils.typing import (
    NOTHING,
    Any,
    Callable,
    Iterable,
    NoneOrNothing,
    Tuple,
    TYPE_CHECKING,
    Union,
    is_none_or_nothing
)
from torchslime.logging.rich import (
    Group,
    RenderableType,
    Table,
    Text,
    Tree,
    parse_renderable,
    escape
)
from torchslime.utils.bases import (
    BaseList,
    CompositeBFT
)
from torchslime.components.store import store
from contextlib import contextmanager

if TYPE_CHECKING:
    from torchslime.core.context import Context
    from torchslime.core.handlers import Handler
    from torchslime.core.handlers.wrappers import HandlerWrapper, HandlerWrapperContainer

#
# Handler Progress Interface
#

class ProgressInterface:

    def create_progress__(self, ctx: "Context") -> Tuple[Any, Any]: pass
    def progress_update__(self, ctx: "Context") -> None: pass

    def add_progress__(self, ctx: "Context") -> None:
        display_ctx = ctx.display_ctx
        display_ctx.live_group.append(display_ctx.handler_progress)

    def remove_progress__(self, ctx: "Context") -> None:
        ctx.display_ctx.handler_progress.remove_self__()

    @contextmanager
    def progress_context__(self, ctx: "Context"):
        progress, task_id = self.create_progress__(ctx)
        with ctx.display_ctx.assign__(
            handler_progress=progress,
            progress_task_id=task_id
        ):
            self.add_progress__(ctx)
            yield
            self.remove_progress__(ctx)


class ProfileProgressInterface(ProgressInterface):

    def progress_update__(self, ctx: "Context") -> None:
        ctx.display_ctx.handler_progress.progress.advance(
            task_id=ctx.display_ctx.progress_task_id,
            advance=1
        )
        ctx.display_ctx.handler_progress.set_text__(
            f'{ctx.hook_ctx.profiler.meter_profile(ctx)}'
        )

    def remove_progress__(self, ctx: "Context") -> None:
        super().remove_progress__(ctx)
        # detach observer
        store.builtin__().detach__(ctx.display_ctx.handler_progress.progress)

#
# Handler Structure Display
#

class HandlerTreeProfiler:

    def handler_profile(
        self,
        handler: "Handler",
        display_attr: bool = True,
        target_handlers: Union[Iterable["Handler"], NoneOrNothing] = NOTHING,
        wrap_func: Union[str, Callable[[Group, "Handler"], Group], NoneOrNothing] = NOTHING
    ) -> Group:
        renderables = [
            Text(handler.get_class_name(), style='bold blue')
        ]

        if display_attr:
            attr = handler.get_display_attr_dict()
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

#
# Handler Wrapper Display
#

class HandlerWrapperContainerProfiler:

    def wrapper_profile(
        self,
        wrapper: Union["HandlerWrapper", "HandlerWrapperContainer"]
    ) -> str:
        from torchslime.utils.common import dict_to_key_value_str_list, concat_format

        class_name = wrapper.get_class_name()

        display_attr_list = dict_to_key_value_str_list(wrapper.get_display_attr_dict())
        attr = concat_format('(', display_attr_list, ')', item_sep=', ')

        return escape(f'{class_name}{attr}')

    def profile(self, handler_wrapper_container: "HandlerWrapperContainer") -> RenderableType:
        table = Table(show_lines=True)
        table.add_column('index')
        table.add_column('wrapper/container')

        table.add_row('[bold]Container', f'[bold]{self.wrapper_profile(handler_wrapper_container)}')

        for index, handler_wrapper in enumerate(handler_wrapper_container):
            table.add_row(str(index), self.wrapper_profile(handler_wrapper))

        return table

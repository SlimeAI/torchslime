from . import HandlerContainer, H_SEQ
from torchslime.core.context import BaseContext
from torchslime.utils.bases import BaseList, Nothing
from typing import Union, Sequence
from torchslime.utils.decorators import CallDebug
from torchslime.log import logger


class HandlerWrapper(HandlerContainer):
    
    def __init__(
        self,
        handlers: H_SEQ = None,
        *args,
        wrappers: Union['HandlerWrapper', Sequence['HandlerWrapper']] = None,
        **kwargs
    ):
        super().__init__(handlers, *args, **kwargs)
        self.wrappers__: Sequence['HandlerWrapper'] = BaseList(wrappers).get_list__()

    def handle(self, ctx: BaseContext):
        self.before_handle(ctx)
        super().handle(ctx)
        self.after_handle(ctx)

    def before_handle(self, ctx: BaseContext):
        for wrapper in self.wrappers__:
            wrapper.before_handle(ctx)

    def after_handle(self, ctx: BaseContext):
        for wrapper in reversed(self.wrappers__):
            wrapper.after_handle(ctx)


class StateHandler(HandlerWrapper):
    
    def __init__(
        self,
        handlers: H_SEQ = None,
        *args,
        state: str = 'train',
        restore: bool = True,
        **kwargs
    ):
        super().__init__(handlers, *args, **kwargs)
        # get state supported
        from torchslime.core.hooks.state import ctx_state
        mode_supported = list(ctx_state.keys())
        if state not in mode_supported:
            logger.warn('An unsupported state is set, this may cause some problems.')
        self.state = state
        self.restore = restore
    
    def before_handle(self, ctx: BaseContext):
        from torchslime.core.hooks.state import ctx_state, StateHook
        # cache the state before state set
        self.restore_state: Union[StateHook, Nothing] = ctx.hook_ctx.state
        # set state to the context
        ctx.hook_ctx.state: StateHook = ctx_state.get(self.state)()
        # change pytorch model mode
        ctx.hook_ctx.state.set_model_mode(ctx)
    
    def after_handle(self, ctx: BaseContext):
        if self.restore:
            from torchslime.core.hooks.state import StateHook
            # restore state
            ctx.hook_ctx.state: StateHook = self.restore_state
            # change pytorch model mode
            ctx.hook_ctx.state.set_model_mode(ctx)
        if hasattr(self, 'restore_state'):
            # destroy cached state
            del self.restore_state
    
    def _get_display_attrs(self) -> dict:
        custom_attrs = {
            'state': 'state',
            'restore': 'restore'
        }
        return {
            **super()._get_display_attrs(),
            **custom_attrs
        }

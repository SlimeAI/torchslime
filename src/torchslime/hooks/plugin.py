from torchslime.utils.base import (
    BaseList,
    BaseGenerator,
    BaseGeneratorQueue
)
from torchslime.hooks.build import BuildInterface
from torchslime.utils.typing import (
    Generator,
    TYPE_CHECKING,
    TypeVar,
    Generic
)
from slime_core.hooks.plugin import (
    CorePluginHook,
    CorePluginContainer
)
if TYPE_CHECKING:
    from torchslime.context import Context


class PluginHook(BuildInterface, CorePluginHook["Context"]):
    
    def build_train_yield(self, ctx: "Context") -> Generator: pass
    def build_eval_yield(self, ctx: "Context") -> Generator: pass
    def build_predict_yield(self, ctx: "Context") -> Generator: pass


_PluginHookT = TypeVar("_PluginHookT", bound=PluginHook)

class PluginContainer(
    PluginHook,
    BaseList[_PluginHookT],
    CorePluginContainer["Context", _PluginHookT],
    Generic[_PluginHookT]
):
    def build_train_yield(self, ctx: "Context") -> Generator:
        with BaseGeneratorQueue((
            BaseGenerator(plugin.build_train_yield(ctx)) for plugin in self
        )):
            yield
    
    def build_eval_yield(self, ctx: "Context") -> Generator:
        with BaseGeneratorQueue((
            BaseGenerator(plugin.build_eval_yield(ctx)) for plugin in self
        )):
            yield
    
    def build_predict_yield(self, ctx: "Context") -> Generator:
        with BaseGeneratorQueue((
            BaseGenerator(plugin.build_predict_yield(ctx)) for plugin in self
        )):
            yield

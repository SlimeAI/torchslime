from torchslime.core.context import BaseContext
from torchslime.utils.bases import BaseList
from torchslime.core.hooks.build import _BuildInterface


class PluginHook(_BuildInterface): pass


class PluginContainer(PluginHook, BaseList[PluginHook]):
    
    def before_build_train(self, ctx: BaseContext) -> None:
        for plugin in self:
            plugin.before_build_train(ctx)
    
    def after_build_train(self, ctx: BaseContext) -> None:
        for plugin in self:
            plugin.after_build_train(ctx)
    
    def before_build_eval(self, ctx: BaseContext) -> None:
        for plugin in self:
            plugin.before_build_eval(ctx)
    
    def after_build_eval(self, ctx: BaseContext) -> None:
        for plugin in self:
            plugin.after_build_eval(ctx)
    
    def before_build_predict(self, ctx: BaseContext) -> None:
        for plugin in self:
            plugin.before_build_predict(ctx)
    
    def after_build_predict(self, ctx: BaseContext) -> None:
        for plugin in self:
            plugin.after_build_predict(ctx)

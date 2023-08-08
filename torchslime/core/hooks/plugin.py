from torchslime.core.context import Context
from torchslime.utils.bases import BaseList
from torchslime.core.hooks.build import _BuildInterface


class PluginHook(_BuildInterface): pass


class PluginContainer(PluginHook, BaseList):
    
    def before_build_train(self, ctx: Context) -> None:
        for plugin in self.get_list__():
            plugin: PluginHook
            plugin.before_build_train(ctx)
    
    def after_build_train(self, ctx: Context) -> None:
        for plugin in self.get_list__():
            plugin: PluginHook
            plugin.after_build_train(ctx)
    
    def before_build_eval(self, ctx: Context) -> None:
        for plugin in self.get_list__():
            plugin: PluginHook
            plugin.before_build_eval(ctx)
    
    def after_build_eval(self, ctx: Context) -> None:
        for plugin in self.get_list__():
            plugin: PluginHook
            plugin.after_build_eval(ctx)
    
    def before_build_predict(self, ctx: Context) -> None:
        for plugin in self.get_list__():
            plugin: PluginHook
            plugin.before_build_predict(ctx)
    
    def after_build_predict(self, ctx: Context) -> None:
        for plugin in self.get_list__():
            plugin: PluginHook
            plugin.after_build_predict(ctx)

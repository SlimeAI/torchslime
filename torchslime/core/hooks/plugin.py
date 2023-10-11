from torchslime.core.context import BaseContext
from torchslime.utils.bases import (
    BaseList,
    BaseGenerator
)
from torchslime.core.hooks.build import BuildInterface
from torchslime.utils.typing import Generator


class PluginHook(BuildInterface): pass


class PluginContainer(PluginHook, BaseList[PluginHook]):
    
    def build_train_yield(self, ctx: BaseContext) -> Generator:
        gen_list = [BaseGenerator(plugin.build_train_yield(ctx)) for plugin in self]
        # before
        for gen in gen_list:
            gen()
        yield
        # after
        for gen in gen_list:
            gen()
    
    def build_eval_yield(self, ctx: BaseContext) -> Generator:
        gen_list = [BaseGenerator(plugin.build_eval_yield(ctx)) for plugin in self]
        # before
        for gen in gen_list:
            gen()
        yield
        # after
        for gen in gen_list:
            gen()
    
    def build_predict_yield(self, ctx: BaseContext) -> Generator:
        gen_list = [BaseGenerator(plugin.build_predict_yield(ctx)) for plugin in self]
        # before
        for gen in gen_list:
            gen()
        yield
        # after
        for gen in gen_list:
            gen()

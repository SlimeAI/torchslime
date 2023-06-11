from torchslime.utils.bases import BaseList


class PluginHook:
    
    def __init__(self) -> None: pass
    def before_build(self, ctx): pass
    def after_build(self, ctx): pass
    def before_build_train(self, ctx): pass
    def after_build_train(self, ctx): pass
    def before_build_eval(self, ctx): pass
    def after_build_eval(self, ctx): pass
    def before_build_predict(self, ctx): pass
    def after_build_predict(self, ctx): pass


class PluginContainer(PluginHook, BaseList):
    
    pass

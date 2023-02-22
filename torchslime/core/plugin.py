from torchslime.util import BaseList


class Plugin:
    
    def __init__(self) -> None:
        pass
    
    def before_build_all(self, ctx):
        pass
    
    def after_build_all(self, ctx):
        pass
    
    def before_build_train(self, ctx):
        pass
    
    def after_build_train(self, ctx):
        pass
    
    def before_build_eval(self, ctx):
        pass
    
    def after_build_eval(self, ctx):
        pass
    
    def before_build_predict(self, ctx):
        pass
    
    def after_build_predict(self, ctx):
        pass


class PluginContainer(Plugin, BaseList):
    
    pass

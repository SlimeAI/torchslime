from typing import Any, Union, Sequence, Callable
from torchslime.utils.bases import Base, Nothing, NOTHING, is_none_or_nothing, BaseList, is_nothing
from torchslime.utils.decorators import ItemAttrBinding, ObjectAttrBinding, Singleton
from torchslime.utils import is_slime_naming, xor__
from torchslime.components.exception import APIMisused

#
# Experiment Config object
#

class Config(Base): pass

#
# Config Container
#

class _ConfigContainerType(type): pass

@ObjectAttrBinding
@ItemAttrBinding
class ConfigContainer(metaclass=_ConfigContainerType):

    def __init__(self) -> None:
        self.set_config__(Config())
        self.set_config_items__(Base())

        # parse default config items
        self.parse_config_items__()

    def set_config__(self, config: Config) -> None:
        return self.object_set__('config__', config)
    
    def get_config__(self) -> Config:
        return self.object_get__('config__')

    def set_config_items__(self, config_items: Base) -> None:
        return self.object_set__('config_items__', config_items)

    def get_config_items__(self) -> Base:
        return self.object_get__('config_items__')

    # original object operation
    def object_set__(self, __name: str, __value: Any) -> None: pass
    def object_get__(self, __name: str) -> Any: pass
    def object_del__(self, __name: str) -> None: pass

    def parse_config_items__(self):
        for _class in self.object_get__('__class__').__mro__:
            for key, value in _class.__dict__.items():
                if self.config_items__.hasattr__(key):
                    continue

                if isinstance(value, (ConfigField, _ConfigContainerType)):
                    # set config items cache
                    self.config_items__[key] = value
                    # set default config values
                    self.config__[key] = value()
                elif isinstance(value, ConfigContainer):
                    # TODO: warn and do nothing
                    pass

    def __call__(self) -> Config:
        self.config__: Config

        containers = list(filter(
            lambda item: isinstance(item[1], ConfigContainer),
            self.config__.__dict__.items()
        ))
        # get raw Config object
        for key, value in containers:
            self.config__[key] = value()
        return self.config__

    def __setattr__(self, __name: str, __value: Any) -> None:
        config_item = self.config_items__[__name]

        if isinstance(config_item, _ConfigContainerType):
            # TODO: warn
            pass
        elif isinstance(config_item, ConfigField):
            __value = config_item(__value)
        setattr(self.config__, __name, __value)

    def __getattribute__(self, __name: str) -> Any:
        # magic naming or slime naming
        if is_slime_naming(__name) is True:
            return super().__getattribute__(__name)
        # else get from config objects
        return getattr(self.config__, __name)

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.config__, __name)

    def __delattr__(self, __name: str) -> None:
        delattr(self.config__, __name)

class ConfigFactory(ConfigContainer):

    def __init__(
        self,
        loaders: Union['ConfigLoader', Sequence['ConfigLoader'], Nothing, None] = NOTHING
    ) -> None:
        super().__init__()
        
        if is_none_or_nothing(loaders):
            self.set_loaders__([FACTORY_LOADER, CLILoader()])
        else:
            self.set_loaders__(BaseList(loaders).get_list__())

    def get_loaders__(self) -> list:
        return self.object_get__('loaders__')
    
    def set_loaders__(self, loaders: list) -> None:
        self.object_set__('loaders__', loaders)

    # user-defined config setter
    def set__(self): pass

    def __call__(self) -> Config:
        for loader in self.get_loaders__():
            loader: ConfigLoader
            loader.load(self)
        
        return super().__call__()

#
# Config Field
#

class ConfigField:

    def __init__(
        self,
        default: Any = NOTHING,
        default_factory: Callable[[], Any] = NOTHING,
        validator: Callable[[Any], bool] = NOTHING,
        parser: Callable[[Any], Any] = NOTHING
    ) -> None:
        if not xor__(is_nothing(default), is_nothing(default_factory)):
            raise APIMisused('One and only one of the ``default`` and ``default_factory`` params must be specified.')
        
        self.default = default
        self.default_factory = default_factory
        # validator can raise ``Exception`` or simply return ``False`` to denote validation error.
        self.validator = validator
        self.parser = parser
        if not hasattr(self, 'fieldname'):
            self.fieldname = NOTHING

    def __call__(self, __value: Any = NOTHING) -> Any:
        # get default value
        if is_nothing(__value) is True:
            __value = self.default if not is_nothing(self.default) else self.default_factory()
        # validate
        if is_none_or_nothing(self.validator) is False and self.validator(__value) is False:
            raise ValueError('Validation error: {fieldname}'.format(fieldname=self.fieldname))
        # parse
        if is_none_or_nothing(self.parser) is False:
            __value = self.parser(__value)
        return __value
    
    def __set_name__(self, _, name):
        if is_nothing(getattr(self, 'fieldname', NOTHING)) is False:
            # TODO: warn
            pass
        self.fieldname = name

#
# Config Loader
#

class ConfigLoader:
    
    def load(self, factory: ConfigFactory) -> None: pass

class JSONLoader(ConfigLoader):
    pass

class YAMLLoader(ConfigLoader):
    pass

@Singleton
class FactoryLoader(ConfigLoader):
    
    def load(self, factory: ConfigFactory) -> None:
        factory.set__()

FACTORY_LOADER = FactoryLoader()

class CLILoader(ConfigLoader):
    pass

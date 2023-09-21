from torchslime.utils.typing import (
    NOTHING,
    Any,
    Nothing,
    NoneOrNothing,
    Union,
    Iterable,
    Callable,
    List,
    Dict,
    Type,
    Tuple,
    is_none_or_nothing,
    is_slime_naming,
    PASS
)
from torchslime.utils.bases import Base, BaseList
from torchslime.utils.decorators import ItemAttrBinding, Singleton
from torchslime.utils import xor__
from torchslime.utils.log import logger
from torchslime.components.exception import APIMisused

#
# Experiment Config object
#

class Config(Base):
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        classname = str(self.__class__.__name__)
        _dict = str(self.__dict__)
        return f'{classname}({_dict})'
    
    def to_dict__(self) -> Dict:
        # TODO: to be implemented
        pass

#
# Config Container / Container List
#

class _ConfigBase:

    def __call__(self, plain: bool = True): pass

@ItemAttrBinding
class ConfigContainer(_ConfigBase):

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

                if isinstance(value, Field):
                    # set config items cache
                    self.config_items__[key] = value
                    # set default config values
                    self.config__[key] = value()

    def __call__(self, plain: bool = True) -> Union[Config, Dict]:
        self.config__: Config

        containers = list(filter(
            lambda item: isinstance(item[1], _ConfigBase),
            self.config__.__dict__.items()
        ))
        # get raw dict or Config object
        for key, value in containers:
            self.config__[key] = value(plain)
        return self.config__.__dict__ if plain else self.config__

    def __setattr__(self, __name: str, __value: Any) -> None:
        config_item = self.config_items__[__name]

        if isinstance(config_item, Field):
            __value = config_item(__value)
        setattr(self.config__, __name, __value)

    def __getattribute__(self, __name: str) -> Any:
        # slime naming
        if is_slime_naming(__name):
            return super().__getattribute__(__name)
        # else get from config objects
        return getattr(self.config__, __name)

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.config__, __name)

    def __delattr__(self, __name: str) -> None:
        delattr(self.config__, __name)
    
    def from_dict__(self, __dict: Dict) -> None:
        pass
    
    def from_kwargs__(self, **kwargs) -> None:
        return self.from_dict__(kwargs)

class ConfigFactory(ConfigContainer):

    def __init__(
        self,
        loaders: Union[Iterable['ConfigLoader'], NoneOrNothing] = NOTHING
    ) -> None:
        super().__init__()
        
        if is_none_or_nothing(loaders):
            self.set_loaders__([FACTORY_LOADER, CLILoader()])
        else:
            self.set_loaders__(BaseList(loaders).get_list__())

    @classmethod
    def get__(
        cls,
        loaders: Union[Iterable['ConfigLoader'], NoneOrNothing] = NOTHING,
        plain: bool = True
    ) -> Config:
        return cls(loaders)(plain)

    def get_loaders__(self) -> List['ConfigLoader']:
        return self.object_get__('loaders__')
    
    def set_loaders__(self, loaders: List['ConfigLoader']) -> None:
        self.object_set__('loaders__', loaders)

    # user-defined config setter
    def set__(self): pass

    def __call__(self, plain: bool = True) -> Config:
        for loader in self.get_loaders__():
            loader.load(self)
        
        return super().__call__(plain)


class ConfigContainerList(_ConfigBase, BaseList[_ConfigBase]):
    
    def __init__(
        self,
        container_class: Union[Type[_ConfigBase], Tuple[Type[_ConfigBase]]] = _ConfigBase,
        container_list: Union[Iterable[_ConfigBase], NoneOrNothing] = NOTHING
    ):
        # set ``container_class`` before all list operations
        self.container_class = container_class
        super().__init__(container_list)
    
    def __setitem__(self, __key, __value):
        self.check__(__value)
        return super().__setitem__(__key, __value)
    
    def insert(self, __index, __object):
        self.check__(__object)
        return super().insert(__index, __object)
    
    def check__(self, __item):
        # check container class
        if not isinstance(__item, self.container_class):
            # if ``__item`` is a ``ConfigContainer`` object, use ``object_get__`` to get the real container class
            item_class = __item.object_get__('__class__') if isinstance(__item, ConfigContainer) else __item.__class__
            
            classname = str(self.__class__.__name__)
            expected = str(self.container_class.__name__)
            actual = str(item_class.__name__)
            raise ValueError(f'Validation error: ``{classname}`` only accepts specified ``{expected}`` objects, but ``{actual}`` received.')
    
    def __call__(self, plain: bool = True) -> List[Union[List, Config, Dict]]:
        return [item(plain) for item in self]

#
# Config Field
#

class Field:

    def __init__(
        self,
        default: Any = PASS,
        default_factory: Callable[[], Any] = PASS,
        validator: Callable[[Any], bool] = NOTHING,
        parser: Callable[[Any], Any] = NOTHING
    ) -> None:
        if not xor__(default is PASS, default_factory is PASS):
            raise APIMisused('One and only one of the ``default`` and ``default_factory`` params must be specified.')
        
        self.default = default
        self.default_factory = default_factory
        # validator can raise ``Exception`` or simply return ``False`` to denote validation error.
        self.validator = validator
        self.parser = parser
        if not hasattr(self, 'fieldname'):
            self.fieldname = NOTHING

    def __call__(self, __value: Any = PASS) -> Any:
        # get default value
        if __value is PASS:
            __value = self.default if self.default is not PASS else self.default_factory()
        # validate
        if not is_none_or_nothing(self.validator) and not self.validator(__value):
            raise ValueError(f'Validation error: {self.fieldname}')
        # parse
        if not is_none_or_nothing(self.parser):
            __value = self.parser(__value)
        return __value
    
    def __set_name__(self, _, name):
        if getattr(self, 'fieldname', NOTHING) is not NOTHING:
            # TODO: warn
            pass
        self.fieldname = name

class ContainerField(Field):

    def __init__(self, container_class: Type[ConfigContainer]) -> None:
        super().__init__(
            default_factory=container_class,
            validator=self._validator
        )
    
    def _validator(self, item):
        # only warning and do nothing here
        if not isinstance(item, ConfigContainer):
            logger.warning(
                'You are setting a ``ConfigContainer`` item to a plain object item, '
                'and pre-defined validators and parsers in the item will not work. '
                f'Fieldname being set: {self.fieldname}'
            )
        # always return True here
        return True

class ContainerListField(Field):
    
    def __init__(
        self,
        container_class: Union[Type[_ConfigBase], Tuple[Type[_ConfigBase]]] = _ConfigBase,
        default_factory: Union[Callable[[], Iterable[_ConfigBase]], NoneOrNothing] = NOTHING
    ) -> None:
        super().__init__(
            default_factory=self._default_factory(container_class, default_factory),
            validator=self._validator
        )
    
    def _default_factory(
        self,
        container_class: Union[Type[_ConfigBase], Tuple[Type[_ConfigBase]]],
        default_factory: Union[Callable[[], Iterable[_ConfigBase]], NoneOrNothing]
    ):
        def partial():
            if is_none_or_nothing(default_factory):
                container_list = []
            else:
                container_list = default_factory()
            return ConfigContainerList(container_class, container_list)
        return partial

    def _validator(self, item):
        # only warning and do nothing here
        if not isinstance(item, ConfigContainerList):
            logger.warning(
                'You are setting a ``ConfigContainerList`` item to a plain object item, '
                'and pre-defined type checker in the item will not work. '
                f'Fieldname being set: {self.fieldname}'
            )
        # always return True here
        return True

#
# Config Loader
#

class ConfigLoader:
    
    def load(self, factory: ConfigFactory) -> None: pass

class JSONLoader(ConfigLoader):
    
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
    
    def load(self, factory: ConfigFactory) -> None:
        # TODO
        return super().load(factory)

class YAMLLoader(ConfigLoader):
    # TODO
    pass

@Singleton
class FactoryLoader(ConfigLoader):
    
    def load(self, factory: ConfigFactory) -> None:
        factory.set__()

FACTORY_LOADER = FactoryLoader()

class CLILoader(ConfigLoader):
    pass

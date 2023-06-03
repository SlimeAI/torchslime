"""
A convenient module register util that helps you dynamically build modules.
"""
from torchslime.utils import NOTHING, Nothing, Singleton, BaseDict, is_none_or_nothing
from typing import Union


class Registry(BaseDict):

    def __init__(self, namespace: str, global_register: bool = True, strict: bool = True):
        super().__init__({})
        self.__namespace = namespace

        if global_register is True:
            registry_mapper.register(self, strict=strict)
    
    def register(self, name: Union[str, Nothing, None] = NOTHING, strict: bool = True, *, cls=NOTHING):
        def decorator(_cls):
            nonlocal name
            if is_none_or_nothing(name) is True:
                name = getattr(_cls, '__name__', NOTHING)
            if name in self and strict is True:
                raise ValueError('Name "{name}" already in registry "{namespace}".'.format(
                    name=name,
                    namespace=self.get_namespace()
                ))
            self[name] = _cls
            return _cls
        
        if is_none_or_nothing(cls) is False:
            decorator(_cls=cls)
            return cls
        return decorator

    def get_namespace(self):
        return self.__namespace


@Singleton
class RegistryMapper(BaseDict):

    def __init__(self) -> None:
        super().__init__({})
    
    def register(self, registry: Registry, strict: bool = True):
        namespace = registry.get_namespace()
        if namespace in self and strict is True:
            raise ValueError('Registry namespace "{namespace}" already in registry mapper.'.format(
                namespace=namespace
            ))
        self[namespace] = registry


registry_mapper = RegistryMapper()

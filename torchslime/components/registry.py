"""
A convenient module register util that helps you dynamically build modules.
"""
from torchslime.utils.bases import BaseDict
from typing import Union, Sequence
from torchslime.utils.bases import NOTHING, Nothing, is_none_or_nothing

from torchslime.utils.decorators import Singleton


class Registry(BaseDict):

    def __init__(
        self,
        namespace: str,
        *,
        strict: bool = True,
        mapper_register: bool = True,
        mapper_strict: bool = True
    ):
        super().__init__({})
        self.__namespace = namespace
        self.strict = strict

        if mapper_register is True:
            registry_mapper.register(self, strict=mapper_strict)
    
    def register(self, name: Union[str, Nothing, None] = NOTHING, strict: Union[bool, Nothing, None] = NOTHING):
        strict = self._get_strict(strict)

        def decorator(cls):
            nonlocal name
            if is_none_or_nothing(name) is True:
                name = getattr(cls, '__name__', NOTHING)
            if name in self.get_dict__() and strict is True:
                raise ValueError('Name "{name}" already in registry "{namespace}".'.format(
                    name=name,
                    namespace=self.get_namespace()
                ))
            self.get_dict__()[name] = cls
            return cls
        return decorator

    def register_multi(self, names: Sequence[str], strict: Union[bool, Nothing, None] = NOTHING):
        strict = self._get_strict(strict)

        def decorator(cls):
            for name in names:
                self.register(name, strict=strict)(cls)
            return cls
        return decorator

    def get(self, __name):
        return self.get_dict__().get(__name)

    def get_namespace(self):
        return self.__namespace

    def _get_strict(self, strict: Union[bool, Nothing, None]):
        return strict if is_none_or_nothing(strict) is False else self.strict


@Singleton
class RegistryMapper(BaseDict):

    def __init__(self) -> None:
        super().__init__({})
    
    def register(self, registry: Registry, strict: bool = True):
        namespace = registry.get_namespace()
        if namespace in self.get_dict__() and strict is True:
            raise ValueError('Registry namespace "{namespace}" already in registry mapper.'.format(
                namespace=namespace
            ))
        self.get_dict__()[namespace] = registry


registry_mapper = RegistryMapper()

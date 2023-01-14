"""
A convenient module register util that helps you parameterize network structure.
"""
from ..util import NOTHING, MultiConst, Singleton, SingleConst
from typing import Type, Any, Iterable
import torch.nn as nn
from .config import load_json


class Registry:

    modules = MultiConst()
    def __init__(self, namespace: str) -> None:
        super().__init__()
        self.modules = {}
        self.namespace = namespace
        # push registry to the registry mapper
        registry_mapper.push(self)

    def register(self, name: str = None):
        def decorator(cls: Type[Any]):
            nonlocal name
            if name is None:
                name = cls.__name__
            self.modules[name] = cls
            return cls
        return decorator

    def build(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def build_single(self, item):
        assert isinstance(item, dict), 'Module config should be a dict.'
        assert 'name' in item, 'Module config should have attribute "name"'
        return self.build(item['name'], *item.get('args', []), **item.get('kwargs', {}))

    def build_sequential(self, list_like: Iterable):
        blocks = []
        for item in list_like:
            for _ in range(item.get('num', 1)):
                blocks.append(self.build_single(item))
        return nn.Sequential(*blocks)

    def get(self, name):
        return self.modules.get(name, NOTHING)
    
    def __getitem__(self, name):
        return self.get(name)


@Singleton
class RegistryMapper:

    registries = SingleConst({})
    def __init__(self) -> None:
        super().__init__()
    
    def push(self, registry: Registry):
        self.registries[registry.namespace] = registry

    def delete(self, namespace: str):
        if namespace in self.registries:
            del self.registries[namespace]

    def get(self, namespace: str):
        return self.registries.get(namespace, NOTHING)

    def __getitem__(self, namespace):
        return self.get(namespace)


registry_mapper = RegistryMapper()

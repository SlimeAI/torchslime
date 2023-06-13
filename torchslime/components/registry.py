"""
A convenient module register util that helps you dynamically build modules.
"""
from torchslime.utils.bases import BaseDict, NOTHING, Nothing, is_none_or_nothing
from typing import Union, Sequence


class Registry(BaseDict):

    def __init__(
        self,
        namespace: str,
        *,
        strict: bool = True
    ):
        super().__init__({})
        self.__namespace = namespace
        self.strict = strict
    
    def register(
        self,
        _cls=NOTHING,
        *,
        name: Union[str, Nothing, None] = NOTHING,
        strict: Union[bool, Nothing, None] = NOTHING
    ):
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
        
        if is_none_or_nothing(_cls) is True:
            return decorator
        
        return decorator(cls=_cls)

    def register_multi(
        self,
        names: Sequence[str],
        *,
        _cls=NOTHING,
        strict: Union[bool, Nothing, None] = NOTHING
    ):
        strict = self._get_strict(strict)

        def decorator(cls):
            for name in names:
                self.register(_cls=cls, name=name, strict=strict)
            return cls
        
        if is_none_or_nothing(_cls) is True:
            return decorator
        
        return decorator(cls=_cls)

    def get(self, __name):
        return self.get_dict__().get(__name)

    def get_namespace(self):
        return self.__namespace

    def _get_strict(self, strict: Union[bool, Nothing, None]):
        return strict if is_none_or_nothing(strict) is False else self.strict

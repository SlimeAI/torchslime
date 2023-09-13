"""
A convenient module register util that helps you dynamically build modules.
"""
from torchslime.utils.bases import BaseDict
from torchslime.utils.decorators import DecoratorCall
from torchslime.utils.typing import (
    NOTHING,
    Nothing,
    NoneOrNothing,
    Union,
    Sequence,
    Any,
    is_none_or_nothing
)


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
    
    @DecoratorCall(index=1, keyword='_cls')
    def __call__(
        self,
        _cls: Any = NOTHING,
        *,
        name: Union[str, NoneOrNothing] = NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ):
        strict = self._get_strict(strict)

        def decorator(cls):
            nonlocal name
            if is_none_or_nothing(name):
                name = getattr(cls, '__name__', NOTHING)
            if name in self and strict:
                namespace=self.get_namespace()
                raise ValueError(f'Name "{name}" already in registry "{namespace}".')
            self[name] = cls
            return cls
        
        return decorator

    def register_multi(
        self,
        names: Sequence[str],
        *,
        _cls=NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ):
        strict = self._get_strict(strict)

        def decorator(cls):
            for name in names:
                self(_cls=cls, name=name, strict=strict)
            return cls
        
        if is_none_or_nothing(_cls):
            return decorator
        
        return decorator(cls=_cls)

    def get(self, __name):
        return super().get(__name)

    def get_namespace(self):
        return self.__namespace

    def _get_strict(self, strict: Union[bool, NoneOrNothing]):
        return strict if not is_none_or_nothing(strict) else self.strict

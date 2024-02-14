"""
A convenient module register util that helps you dynamically build modules.
"""
from torchslime.utils.bases import BaseDict
from torchslime.utils.decorators import DecoratorCall
from torchslime.utils.typing import (
    NOTHING,
    NoneOrNothing,
    Union,
    Sequence,
    is_none_or_nothing,
    TypeVar,
    Type,
    overload,
    Callable
)

_T = TypeVar('_T')


class Registry(BaseDict[str, _T]):

    def __init__(
        self,
        namespace: str,
        *,
        strict: bool = True
    ):
        super().__init__({})
        self.__namespace = namespace
        self.strict = strict
    
    @overload
    def __call__(
        self,
        _cls: NoneOrNothing = NOTHING,
        *,
        name: Union[str, NoneOrNothing] = NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ) -> Callable[[_T], _T]: pass
    @overload
    def __call__(
        self,
        _cls: _T,
        *,
        name: Union[str, NoneOrNothing] = NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ) -> _T: pass
    
    @DecoratorCall(index=1, keyword='_cls')
    def __call__(
        self,
        _cls: Union[_T, NoneOrNothing] = NOTHING,
        *,
        name: Union[str, NoneOrNothing] = NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ) -> _T:
        strict = self._get_strict(strict)

        def decorator(cls: _T) -> _T:
            nonlocal name
            if is_none_or_nothing(name):
                name = getattr(cls, '__name__', NOTHING)
            if name in self and strict:
                namespace = self.get_namespace()
                raise ValueError(f'Name "{name}" already in registry "{namespace}".')
            self[name] = cls
            return cls
        
        return decorator

    @overload
    def register_multi(
        self,
        names: Sequence[str],
        *,
        _cls: NoneOrNothing = NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ) -> Callable[[_T], _T]: pass
    @overload
    def register_multi(
        self,
        names: Sequence[str],
        *,
        _cls: _T,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ) -> _T: pass

    @DecoratorCall(keyword='_cls')
    def register_multi(
        self,
        names: Sequence[str],
        *,
        _cls: Union[_T, NoneOrNothing] = NOTHING,
        strict: Union[bool, NoneOrNothing] = NOTHING
    ) -> _T:
        strict = self._get_strict(strict)

        def decorator(cls: _T) -> _T:
            for name in names:
                self(_cls=cls, name=name, strict=strict)
            return cls
        
        return decorator

    def get_namespace(self) -> str:
        return self.__namespace

    def _get_strict(self, strict: Union[bool, NoneOrNothing]):
        return strict if not is_none_or_nothing(strict) else self.strict

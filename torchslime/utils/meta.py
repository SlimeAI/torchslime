from typing import (
    Any,
    Union,
    Tuple,
    TypeVar,
    Callable,
    Type,
    overload
)
from .bases import NOTHING, Nothing, is_none_or_nothing
from .decorators import ClassWraps, DecoratorCall

T = TypeVar('T')


class MetaData:
    
    def __or__(self, __value: 'MetaData') -> 'MetaData':
        pass


class _MetaWrapper:
    def __init__(self, cls: Type, metadata: MetaData) -> None:
        self.cls__ = cls
        self.metadata__ = metadata
        
        # set meta info
        meta_str = str(metadata)
        self.__module__ = cls.__module__
        self.__name__ = '{name}[{metadata}]'.format(
            name=cls.__name__,
            metadata=meta_str
        )
        self.__qualname__ = '{name}[{metadata}]'.format(
            name=cls.__qualname__,
            metadata=meta_str
        )
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # create a new object
        obj = self.cls__(*args, **kwargs)
        # set ``metadata__`` attribute
        obj.metadata__: MetaData = self.metadata__ if not is_none_or_nothing(self.metadata__) else MetaData()
        return obj
    
    def __str__(self) -> str: return self.__name__
    def __repr__(self) -> str: return str(self)

# type hint
@overload
def Meta(_cls: Union[None, Nothing] = NOTHING) -> Callable[[Type[T]], Type[T]]: pass
@overload
def Meta(_cls: Type[T]) -> Type[T]: pass

@DecoratorCall(index=0, keyword='_cls')
def Meta(_cls: Type[T] = NOTHING):
    def decorator(cls: Type[T]) -> Type[T]:
        class_wraps = ClassWraps(cls)
        
        class_getitem_wraps = class_wraps.__class_getitem__
        @class_getitem_wraps
        @classmethod
        def class_getitem(cls: Type[T], metadata: Union[MetaData, Tuple[MetaData]]) -> Type[T]:
            # TODO: process tuple of metadata
            return _MetaWrapper(cls, metadata)
        
        return cls
    return decorator


@Meta
class Metaclass:
    # just for type hint
    metadata__: MetaData
    def __class_getitem__(cls, metadata: Union[MetaData, Tuple[MetaData]]): return cls

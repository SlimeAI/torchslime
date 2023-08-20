from typing import (
    Any,
    Union,
    Tuple,
    TypeVar,
    Callable
)
from .bases import NOTHING, Nothing, is_none_or_nothing
from .decorators import ClassWraps, DecoratorCall

T = TypeVar('T')


@DecoratorCall(index=0, keyword='_cls')
def Meta(_cls: Union[T, None, Nothing] = NOTHING) -> Union[T, Callable[[T], T]]:
    # use class decorator to dynamically add ``__new__`` method rather than directly define it in ``Meta`` class,
    # in order to keep type hint of the user-defined ``__init__`` method.
    def decorator(cls: T) -> T:
        class_wraps = ClassWraps(cls)
        
        class_getitem_wraps = class_wraps.__class_getitem__
        @class_getitem_wraps
        @classmethod
        def class_getitem(cls: T, metadata: Union[MetaData, Tuple[MetaData]]):
            class Wrapper:
                def __call__(self, *args: Any, **kwargs: Any) -> cls:
                    return cls(*args, metadata__=metadata, **kwargs)
                
                def __str__(self) -> str: return self.__name__
                def __repr__(self) -> str: return str(self)
            
            wrapper = Wrapper()
            wrapper.__module__ = cls.__module__
            meta_str = str(metadata)
            wrapper.__name__ = '{name}[{metadata}]'.format(
                name=cls.__name__,
                metadata=meta_str
            )
            wrapper.__qualname__ = '{name}[{metadata}]'.format(
                name=cls.__qualname__,
                metadata=meta_str
            )
            return wrapper
        
        new_wraps = class_wraps.__new__
        new_cls_func = new_wraps.cls_func__
        @new_wraps
        def new(
            cls,
            *args: Any,
            metadata__: Union[MetaData, None, Nothing] = NOTHING,
            **kwargs: Any
        ):
            # create new object using the original ``__new__``
            obj = new_cls_func(cls, *args, **kwargs)
            # set ``metadata__`` attribute
            obj.metadata__: MetaData = metadata__ if not is_none_or_nothing(metadata__) else MetaData()
            return obj

        return cls
    return decorator


class MetaData:
    
    pass


@Meta
class Metaclass:
    # just for type hint
    metadata__: MetaData
    def __class_getitem__(cls, metadata: Union[MetaData, Tuple[MetaData]]): return cls

from .typing import (
    Any,
    Union,
    Tuple,
    TypeVar,
    Callable,
    Type,
    overload
)
from .bases import NOTHING, Nothing, is_none_or_nothing, BaseDict
from .decorators import ClassWraps, DecoratorCall, ClassFuncWrapper

T = TypeVar('T')


class Metadata(BaseDict):
    
    def __init__(self, __name: Union[str, None, Nothing] = NOTHING, __value: Any = NOTHING):
        super().__init__()
        if not is_none_or_nothing(__name):
            self[__name] = __value
    
    def __or__(self, __value: 'Metadata') -> 'Metadata':
        if not isinstance(__value, Metadata):
            raise ValueError('``MetaData`` can only be compatible with objects of its own class, but ``{actual_class}`` found.'.format(
                actual_class=str(__value.__class__.__name__)
            ))
        # update from other Metadata(s)
        self.update(__value)
        return self
    
    def __ror__(self, __value: 'Metadata') -> 'Metadata':
        return self | __value


class MetaWrapper:
    def __init__(self, cls: Type, metadata: Metadata) -> None:
        self.cls__ = cls
        if not isinstance(metadata, Metadata):
            raise ValueError('``Meta`` only accepts ``Metadata`` object, but ``{actual_class}`` found.'.format(
                actual_class=str(metadata.__class__.__name__)
            ))
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
    
    def __call__(self, *args: Any, **kwargs: Any):
        # create a new object
        cls = self.cls__
        new = cls.__new__
        # FIX: object.__new__ only accept one cls argument
        if new is object.__new__:
            obj = new(cls)
        else:
            obj = new(cls, *args, **kwargs)
        # set ``metadata__`` attribute
        obj.metadata__ = self.metadata__
        # ``__init__`` method call
        cls.__init__(obj, *args, **kwargs)
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
        def class_getitem(cls: Type[T], metadata: Union[Metadata, Tuple[Metadata]]) -> Type[T]:
            if isinstance(metadata, Tuple):
                result = Metadata()
                for item in metadata:
                    result |= item
            else:
                result = metadata
            return MetaWrapper(cls, result)
        
        class_new_wraps: ClassFuncWrapper = class_wraps.__new__
        new_cls_func = class_new_wraps.cls_func__
        @class_new_wraps
        def new(cls, *args, **kwargs):
            # FIX: object.__new__ only accept one cls argument
            if new_cls_func is object.__new__:
                obj = new_cls_func(cls)
            else:
                obj = new_cls_func(cls, *args, **kwargs)
            # set default metadata
            obj.metadata__ = Metadata()
            return obj
        
        return cls
    return decorator


@Meta
class Metaclass:
    # just for type hint
    metadata__: Metadata
    def __class_getitem__(cls, metadata: Union[Metadata, Tuple[Metadata]]): return cls

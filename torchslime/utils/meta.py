from .typing import (
    Any,
    Union,
    TypeVar,
    Callable,
    Type,
    overload,
    NoReturn
)
from .bases import NOTHING, Nothing, is_none_or_nothing
from .decorators import ClassWraps, DecoratorCall, ClassFuncWrapper, get_cls_func
from .formatter import dict_to_key_value_str_list, concat_format
from torchslime.components.exception import APIMisused

_T = TypeVar('_T')


class _MetaWrapper:
    def __init__(self, cls, *args: Any, **kwargs: Any) -> None:
        self.cls__ = cls
        self.args = args
        self.kwargs = kwargs
        
        # set meta info
        args_str = concat_format('', [str(arg) for arg in args], '', item_sep=', ', break_line=False)
        kwargs_str = concat_format('', dict_to_key_value_str_list(kwargs), '', item_sep=', ', break_line=False)
        meta_str = concat_format('', [item for item in [args_str, kwargs_str] if len(item) > 0], '', item_sep=', ', break_line=False)
        
        self.__module__ = cls.__module__
        self.__name__ = f'{cls.__name__}[{meta_str}]'
        self.__qualname__ = f'{cls.__qualname__}[{meta_str}]'
    
    def __call__(self, *args: Any, **kwargs: Any):
        cls = self.cls__
        # create a new object using ``m_new__``
        obj = cls.m_new__(*args, **kwargs)
        # call ``m_init__`` with args
        obj.m_init__(*self.args, **self.kwargs)
        # ``__init__`` method call
        obj.__init__(*args, **kwargs)
        return obj
    
    def __str__(self) -> str: return self.__name__
    def __repr__(self) -> str: return str(self)

# type hint
@overload
def _Meta(_cls: Union[None, Nothing] = NOTHING, *, directly_new_allowed: bool = True) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def _Meta(_cls: Type[_T], *, directly_new_allowed: bool = True) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def _Meta(
    _cls: Type[_T] = NOTHING,
    *,
    directly_new_allowed: bool = True
):
    def decorator(cls__: Type[_T]) -> Type[_T]:
        if not hasattr(cls__, 'm_init__'):
            raise TypeError(f'Class ``{cls__.__name__}`` with ``Meta`` should have a ``m_init__`` method, but not found.')
        
        class_wraps = ClassWraps(cls__)
        
        # ``m__``
        m_wraps = class_wraps.m__
        @m_wraps(use_wraps=False)
        @classmethod
        def m__(cls: Type[_T], *args, **kwargs) -> Type[_T]:
            return _MetaWrapper(cls, *args, **kwargs)
        
        # ``__init_subclass__``
        init_subclass_wraps: ClassFuncWrapper = class_wraps.__init_subclass__
        init_subclass_cls_func = init_subclass_wraps.cls_func__
        @init_subclass_wraps
        @classmethod
        def init_subclass(
            cls,
            directly_new_allowed: Union[bool, None, Nothing] = NOTHING,
            **kwargs
        ):
            init_subclass_cls_func(**kwargs)
            # set original ``m__`` method to override type hint method definition
            original_m = get_cls_func(cls__, 'm__')
            cls_m = get_cls_func(cls, 'm__')
            if cls_m is not original_m:
                cls.m__ = classmethod(original_m)
            # change ``__new__`` method if ``directly_new_allowed`` is set
            if not is_none_or_nothing(directly_new_allowed):
                cls.__new__ = new if directly_new_allowed else new_disallowed
        
        # ``__new__`` wraps
        class_new_wraps: ClassFuncWrapper = class_wraps.__new__
        new_cls_func = class_new_wraps.cls_func__
        
        def new(cls: Type[_T], *args, **kwargs) -> _T:
            # call ``m_new__`` to create a new object
            obj = cls.m_new__(*args, **kwargs)
            # call ``m_init__`` with no args
            obj.m_init__()
            return obj
        
        def new_disallowed(cls: Type, *args, **kwargs) -> NoReturn:
            raise APIMisused(
                f'Class ``{cls.__name__}`` with ``Meta`` is disallowed to directly new an instance, '
                f'please use ``{cls.__name__}.m__([args...])([args...])`` instead.'
            )
        
        # set ``__new__`` method according to whether directly calling ``__new__`` is allowed
        class_new_wraps(new if directly_new_allowed else new_disallowed)
        
        # ``m_new__`` wraps
        class_m_new_wraps = class_wraps.m_new__
        @class_m_new_wraps(use_wraps=False)
        @classmethod
        def m_new__(cls: Type[_T], *args, **kwargs) -> _T:
            # FIX: object.__new__ only accept one cls argument
            if new_cls_func is object.__new__:
                obj = new_cls_func(cls)
            else:
                obj = new_cls_func(cls, *args, **kwargs)
            return obj
        
        return cls__
    return decorator

@_Meta
class Meta:
    def m_init__(self, *args, **kwargs): pass
    @classmethod
    def m__(cls: Type[_T], *args, **kwargs) -> Type[_T]: return cls

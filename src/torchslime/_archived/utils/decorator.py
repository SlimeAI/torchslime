from functools import wraps
from torchslime.utils.typing import (
    Union,
    Callable,
    TypeVar,
    is_none_or_nothing,
    FuncOrMethod,
    NOTHING,
    Nothing,
    RawFunc,
    is_function_or_method,
    unwrap_method,
    Type,
    NoneOrNothing,
    Any,
    overload,
    get_mro,
    get_bases,
    Tuple,
    MISSING,
    Missing,
    List
)
from torchslime.utils.decorator import DecoratorCall
import threading
import multiprocessing

_T = TypeVar("_T")

#
# ClassWraps decorator
#

from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES
FUNC_CREATED = ('__module__', '__name__', '__qualname__')

def _create_func(
    func: FuncOrMethod,
    cls: type,
    name: str,
    created: Union[list, tuple]
):
    """
    Separately set function attributes.
    """
    # __module__ should be the same as cls
    if '__module__' in created and hasattr(cls, '__module__'):
        setattr(func, '__module__', getattr(cls, '__module__'))
    
    # __name__ should be set
    if '__name__' in created:
        setattr(func, '__name__', name)

    # __qualname__ should be 'cls_qualname.name'
    if '__qualname__' in created and hasattr(cls, '__qualname__'):
        setattr(func, '__qualname__', f'{getattr(cls, "__qualname__")}.{name}')
    return func

class ClassFuncWrapper:

    def __init__(
        self,
        cls: type,
        name: str
    ) -> None:
        self.cls = cls
        self.name = name
        # get functions
        self.cls_func__ = get_cls_func(cls, name)
        self.super_func__ = get_super_func(cls, name)
        self.self_func__ = get_self_func(cls, name)
    
    def __call__(
        self,
        _func=NOTHING,
        *,
        assigned=WRAPPER_ASSIGNMENTS,
        updated=WRAPPER_UPDATES,
        created=FUNC_CREATED,
        use_wraps: bool = True
    ) -> Callable:
        self_func = self.self_func__
        
        def wrapper(func: Callable):
            if not is_none_or_nothing(self_func) and use_wraps:
                func = wraps(self_func, assigned=assigned, updated=updated)(func)
            else:
                func = _create_func(func=func, cls=self.cls, name=self.name, created=created)
            # set wrapper__ attribute to denote it is a func wrapper
            func.wrapper__ = True
            # set cls_func, super_func and self_func to func wrapper
            func.cls_func__ = self.cls_func__
            func.super_func__ = self.super_func__
            func.self_func__ = self.self_func__
            # set func wrapper to cls
            setattr(self.cls, self.name, func)
            return func
        
        if is_none_or_nothing(_func):
            return wrapper
        
        return wrapper(func=_func)

class ClassWraps:
    
    def __init__(self, cls: type) -> None:
        if not isinstance(cls, type):
            from torchslime.utils.exception import APIMisused
            raise APIMisused(f'ClassWraps can only be used for class, not {str(cls)}.')
        
        self.cls = cls
    
    def __getattribute__(self, __name: str) -> 'ClassFuncWrapper':
        # use ``super`` object to get ``cls``
        cls = super().__getattribute__('cls')
        return ClassFuncWrapper(cls, __name)


def _get_self_func_or_method(cls: type, name: str) -> Union[FuncOrMethod, Nothing]:
    __item = cls.__dict__.get(name, NOTHING)
    return __item if is_function_or_method(__item) else NOTHING

def get_self_func(cls: type, name: str) -> Union[RawFunc, Nothing]:
    return unwrap_method(_get_self_func_or_method(cls, name))

def get_original_self_func(func):
    while getattr(func, 'wrapper__', False) and hasattr(func, 'self_func__'):
        func = getattr(func, 'self_func__')
    return func

def _get_func_from_mro(cls: type, name: str, start: int=0) -> Union[RawFunc, Nothing]:
    # get attr from the mro tuple
    mro_tuple = get_mro(cls)
    try:
        return unwrap_method(getattr(mro_tuple[start], name, NOTHING))
    except IndexError:
        return NOTHING

def get_cls_func(cls: type, name: str) -> Union[RawFunc, Nothing]:
    return _get_func_from_mro(cls, name, start=0)

def get_super_func(cls: type, name: str) -> Union[RawFunc, Nothing]:
    return _get_func_from_mro(cls, name, start=1)


def Singleton(cls: _T) -> _T:
    """
    [class, level-1]
    Decorator that makes decorated classes singleton.
    It makes the creation of the singleton object thread-safe by using double-checked locking.
    """
    t_lock = threading.Lock()
    p_lock = multiprocessing.Lock()
    _instance = NOTHING
    _init = False

    cls_wraps = ClassWraps(cls)
    
    # NOTE: this constraint doesn't work for subclasses
    # ``__new__`` methods in subclasses will run multiple times
    new_wraps: ClassFuncWrapper = cls_wraps.__new__
    new_cls_func = new_wraps.cls_func__
    @new_wraps
    def new(cls, *args, **kwargs):
        nonlocal _instance
        if is_none_or_nothing(_instance) is True:
            with t_lock, p_lock:
                if is_none_or_nothing(_instance) is True:
                    if new_cls_func is object.__new__:
                        # FIX: object.__new__ accept only one cls argument
                        _instance = new_cls_func(cls)
                    else:
                        _instance = new_cls_func(cls, *args, **kwargs)
        return _instance
    
    # FIX: Singleton init only once
    # NOTE: this constraint doesn't work for subclasses
    init_wraps: ClassFuncWrapper = cls_wraps.__init__
    init_cls_func = init_wraps.cls_func__
    @init_wraps
    def init(*args, **kwargs):
        nonlocal _init
        if not _init:
            init_cls_func(*args, **kwargs)
            _init = True

    return cls


# type hint
@overload
def ReadonlyAttr(attrs: list, *, _cls: NoneOrNothing = NOTHING, nothing_allowed: bool = True, empty_allowed: bool = True) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def ReadonlyAttr(attrs: list, *, _cls: Type[_T], nothing_allowed: bool = True, empty_allowed: bool = True) -> Type[_T]: pass

@DecoratorCall(keyword='_cls')
def ReadonlyAttr(attrs: list, *, _cls=NOTHING, nothing_allowed: bool = True, empty_allowed: bool = True):
    """
    [class, level-2]
    """
    def decorator(cls: Type[_T]) -> Type[_T]:
        cls_wraps = ClassWraps(cls)
        setattr_wraps: ClassFuncWrapper = cls_wraps.__setattr__
        setattr_cls_func = setattr_wraps.cls_func__

        @setattr_wraps
        def wrapper(self, __name: str, __value: Any):
            # directly set attr here for performance optimization
            if __name not in attrs:
                return setattr_cls_func(self, __name, __value)

            hasattr__ = hasattr(self, __name)
            attr__ = getattr(self, __name, None)

            if (
                (not hasattr__ and empty_allowed) or 
                (attr__ is NOTHING and nothing_allowed)
            ):
                return setattr_cls_func(self, __name, __value)
            else:
                raise AttributeError(f'``{__name}`` is readonly attribute')
        
        return cls
    return decorator


@overload
def MetaclassCheck(
    _cls: Missing = MISSING,
    *,
    ignored_metaclasses: Union[Tuple[Type[type], ...], Missing] = MISSING    
) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def MetaclassCheck(
    _cls,
    *,
    ignored_metaclasses: Union[Tuple[Type[type], ...], Missing] = MISSING    
) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def MetaclassCheck(
    _cls=MISSING,
    *,
    ignored_metaclasses: Union[Tuple[Type[type], ...], Missing] = MISSING
):
    """
    NOTE: Deprecated: The Python interpreter will automatically check the 
    metaclasses.
    
    ---
    
    Check the compatibility of metaclasses between subclasses and superclasses.
    """
    def decorator(cls: Type[_T]) -> Type[_T]:
        ignored = set(
            ignored_metaclasses if ignored_metaclasses is not MISSING else ()
        )
        
        # Metaclasses of ``cls``.
        cls_metaclass = type(cls)
        bases = get_bases(cls)

        for base in bases:
            base_metaclass_mro: List[Type] = list(get_mro(type(base)))
            
            while len(base_metaclass_mro) > 0:
                metaclass = base_metaclass_mro.pop(0)
                # Ignore the adapter metaclass.
                if getattr(metaclass, 'metaclass_adapter__', MISSING) is True:
                    continue
                # Check the metaclass.
                if metaclass not in ignored:
                    if not issubclass(cls_metaclass, metaclass):
                        from torchslime.utils.exception import APIMisused
                        cls_metaclass_mro = get_mro(cls_metaclass)
                        raise APIMisused(
                            f'Metaclass check failed. Class: {cls} - Metaclass: {cls_metaclass} - '
                            f'MRO of Metaclass: {cls_metaclass_mro} - Missing metaclass: {metaclass} '
                            f'from base class {base}'
                        )
                # Remove the mro of ``metaclass`` from ``base_metaclass_mro`` 
                # in order to improve running efficiency.
                metaclass_mro_set = set(get_mro(metaclass))
                base_metaclass_mro = list(filter(
                    lambda _cls: _cls not in metaclass_mro_set,
                    base_metaclass_mro
                ))
        return cls
    return decorator

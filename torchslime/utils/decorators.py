from functools import wraps
import multiprocessing
import threading
from .typing import (
    Any,
    Union,
    Callable,
    TypeVar,
    Type,
    is_function_or_method,
    is_none_or_nothing,
    overload,
    RawFunc,
    FuncOrMethod,
    NoneOrNothing,
    MethodType,
    NOTHING,
    Nothing,
    List,
    overload_dummy
)

_T = TypeVar('_T')

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
            from torchslime.components.exception import APIMisused
            raise APIMisused(f'ClassWraps can only be used for class, not {str(cls)}.')
        
        self.cls = cls
    
    def __getattribute__(self, __name: str) -> 'ClassFuncWrapper':
        # use ``super`` object to get ``cls``
        cls = super().__getattribute__('cls')
        return ClassFuncWrapper(cls, __name)


@overload
def _unwrap(func: FuncOrMethod) -> RawFunc: pass
@overload
def _unwrap(func: NoneOrNothing) -> NoneOrNothing: pass

def _unwrap(func: Union[FuncOrMethod, NoneOrNothing]) -> Union[RawFunc, NoneOrNothing]:
    if isinstance(func, MethodType):
        # get the original function body of the classmethod
        func = func.__func__
    return func

def _get_self_func_or_method(cls: type, name: str) -> Union[FuncOrMethod, Nothing]:
    __item = cls.__dict__.get(name, NOTHING)
    return __item if is_function_or_method(__item) else NOTHING

def get_self_func(cls: type, name: str) -> Union[RawFunc, Nothing]:
    return _unwrap(_get_self_func_or_method(cls, name))

def get_original_self_func(func):
    while getattr(func, 'wrapper__', False) and hasattr(func, 'self_func__'):
        func = getattr(func, 'self_func__')
    return func

def _get_func_from_mro(cls: type, name: str, start: int=0) -> Union[RawFunc, Nothing]:
    # get attr from the mro tuple
    mro_tuple = cls.__mro__
    try:
        return _unwrap(getattr(mro_tuple[start], name, NOTHING))
    except IndexError:
        return NOTHING

def get_cls_func(cls: type, name: str) -> Union[RawFunc, Nothing]:
    return _get_func_from_mro(cls, name, start=0)

def get_super_func(cls: type, name: str) -> Union[RawFunc, Nothing]:
    return _get_func_from_mro(cls, name, start=1)


def DecoratorCall(
    *,
    index: Union[int, Nothing] = NOTHING,
    keyword: Union[str, Nothing] = NOTHING
):
    """
    [func-decorator]
    """
    def decorator(func: _T) -> _T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_match = NOTHING

            if not is_none_or_nothing(keyword):
                arg_match = kwargs.get(str(keyword), NOTHING)
            
            if not is_none_or_nothing(index) and is_none_or_nothing(arg_match):
                arg_match = NOTHING if index >= len(args) else args[index]

            _decorator = func(*args, **kwargs)
            return _decorator if is_none_or_nothing(arg_match) else _decorator(arg_match)
        return wrapper
    return decorator


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
def CallDebug(_func: NoneOrNothing = NOTHING, *, module_name=NOTHING) -> Callable[[_T], _T]: pass
@overload
def CallDebug(_func: _T, *, module_name=NOTHING) -> _T: pass

@DecoratorCall(index=0, keyword='_func')
def CallDebug(_func: _T = NOTHING, *, module_name=NOTHING):
    """
    [func, level-2]
    A decorator that output debug information before and after a method is called.
    Args:
        func (_type_): _description_
    """
    from . import get_exec_info
    
    def decorator(func: _T) -> _T:
        from torchslime.logging.logger import logger
        from torchslime.components.store import store
        from time import time

        func_id = f'{str(id(func))}_{str(time())}'

        nonlocal module_name

        if is_none_or_nothing(module_name) is True:
            module_name = getattr(func, '__name__', NOTHING)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # do not use debug
            if not store.builtin__().call_debug:
                return func(*args, **kwargs)

            # cache debug info
            call_debug_cache = store.builtin__().call_debug_cache
            _exec_info = call_debug_cache[func_id]
            if is_none_or_nothing(_exec_info):
                _exec_info = get_exec_info(func)
                call_debug_cache[func_id] = _exec_info

            # TODO logger format
            logger.debug(f'{module_name} begins.')
            result = func(*args, **kwargs)
            logger.debug(f'{module_name} ends.')
            return result
        return wrapper
    return decorator


def MethodChaining(func):
    """
    [func, level-1]
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper


def Deprecated():
    """
    [func, level-1]
    """
    # TODO
    pass


def Experimental():
    # TODO
    pass


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

            if (not hasattr__ and empty_allowed) or \
                    (attr__ is NOTHING and nothing_allowed):
                return setattr_cls_func(self, __name, __value)
            else:
                raise AttributeError(f'``{__name}`` is readonly attribute')
        
        return cls
    return decorator


# type hint
@overload
def ItemAttrBinding(_cls: NoneOrNothing = NOTHING, *, set_binding: bool = True, get_binding: bool = True, del_binding: bool = True) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def ItemAttrBinding(_cls: Type[_T], *, set_binding: bool = True, get_binding: bool = True, del_binding: bool = True) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def ItemAttrBinding(_cls=NOTHING, *, set_binding: bool = True, get_binding: bool = True, del_binding: bool = True):
    """
    [class]
    """
    def decorator(cls: Type[_T]) -> Type[_T]:
        cls_wraps = ClassWraps(cls)

        if set_binding is True:
            setitem_wraps = cls_wraps.__setitem__

            @setitem_wraps
            def setitem(self, __name: str, __value: Any) -> None:
                return setattr(self, str(__name), __value)
        
        if get_binding is True:
            getitem_wraps = cls_wraps.__getitem__

            @getitem_wraps
            def getitem(self, __name: str) -> Any:
                return getattr(self, str(__name))
        
        if del_binding is True:
            delitem_wraps = cls_wraps.__delitem__

            @delitem_wraps
            def delitem(self, __name: str) -> None:
                return delattr(self, str(__name))
        
        return cls

    return decorator


@overload
def ContextDecoratorBinding(_cls: NoneOrNothing = NOTHING) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def ContextDecoratorBinding(_cls: Type[_T]) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def ContextDecoratorBinding(_cls=NOTHING):
    def decorator(cls):
        cls_wraps = ClassWraps(cls)
        
        @cls_wraps.__call__
        def call(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self:
                    # context ``__exit__`` method will be automatically executed before return statement
                    return func(*args, **kwargs)
            return wrapper
        
        return cls
    return decorator


@overload
def RemoveOverload(_cls: NoneOrNothing = NOTHING, *, checklist: Union[NoneOrNothing, List[str]] = NOTHING) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def RemoveOverload(_cls: Type[_T], *, checklist: Union[NoneOrNothing, List[str]] = NOTHING) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def RemoveOverload(_cls=NOTHING, *, checklist: Union[NoneOrNothing, List[str]] = NOTHING):
    def decorator(cls: Type[_T]) -> Type[_T]:
        nonlocal checklist
        
        _dict = cls.__dict__
        filter_func = lambda key: key in _dict and _unwrap(_dict[key]) is overload_dummy
        
        if is_none_or_nothing(checklist):
            checklist = filter(filter_func, _dict.keys())
        else:
            checklist = filter(filter_func, checklist)
        for attr in checklist:
            delattr(cls, attr)
        
        return cls
    return decorator

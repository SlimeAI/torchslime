from functools import wraps
from .typing import (
    Union,
    Callable,
    TypeVar,
    Type,
    is_none_or_nothing,
    overload,
    FuncOrMethod,
    NoneOrNothing,
    NOTHING,
    Nothing,
    List,
    overload_dummy,
    MISSING,
    Dict,
    Missing,
    unwrap_method
)

_T = TypeVar('_T')
_T_FuncOrMethod = TypeVar('_T_FuncOrMethod', bound=FuncOrMethod)


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


# type hint
@overload
def CallDebug(
    _func: NoneOrNothing = NOTHING,
    *,
    module_name: Union[str, NoneOrNothing] = NOTHING,
) -> Callable[[_T], _T]: pass
@overload
def CallDebug(
    _func: _T,
    *,
    module_name: Union[str, NoneOrNothing] = NOTHING,
) -> _T: pass

@DecoratorCall(index=0, keyword='_func')
def CallDebug(
    _func: _T = NOTHING,
    *,
    module_name: Union[str, NoneOrNothing] = NOTHING
):
    """
    [func, level-2]
    A decorator that output debug information before and after a method is called.
    Args:
        func (_type_): _description_
    """
    from .common import get_exec_info
    
    def decorator(func: _T) -> _T:
        from torchslime.logging.logger import logger, LoggerKwargs
        from .store import store

        # Inspect ``module_name``
        nonlocal module_name
        if is_none_or_nothing(module_name):
            module_name = getattr(func, '__qualname__', NOTHING)
        if is_none_or_nothing(module_name):
            module_name = getattr(func, '__name__', NOTHING)

        # Lazy loading.
        _exec_info = MISSING

        @wraps(func)
        def wrapper(*args, **kwargs):
            # do not use debug
            if not store.builtin__().call_debug:
                return func(*args, **kwargs)
            
            nonlocal _exec_info
            if _exec_info is MISSING:
                _exec_info = get_exec_info(func)
            
            exec_name = _exec_info["full_exec_name"] if \
                store.builtin__().call_debug_full_exec_name else \
                _exec_info["exec_name"]
            exec_info = f'Module definition -> {exec_name}:{_exec_info["lineno"]}'
            
            logger.debug(
                f'{module_name} begins. | {exec_info}',
                **LoggerKwargs(stacklevel=2)
            )
            result = func(*args, **kwargs)
            logger.debug(
                f'{module_name} ends. | {exec_info}',
                **LoggerKwargs(stacklevel=2)
            )
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


@overload
def RemoveOverload(_cls: NoneOrNothing = NOTHING, *, checklist: Union[NoneOrNothing, List[str]] = NOTHING) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def RemoveOverload(_cls: Type[_T], *, checklist: Union[NoneOrNothing, List[str]] = NOTHING) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def RemoveOverload(_cls=NOTHING, *, checklist: Union[NoneOrNothing, List[str]] = NOTHING):
    def decorator(cls: Type[_T]) -> Type[_T]:
        nonlocal checklist
        
        _dict = cls.__dict__
        filter_func = lambda key: key in _dict and unwrap_method(_dict[key]) is overload_dummy
        
        if is_none_or_nothing(checklist):
            checklist = filter(filter_func, _dict.keys())
        else:
            checklist = filter(filter_func, checklist)
        for attr in checklist:
            delattr(cls, attr)
        
        return cls
    return decorator


def InitOnce(func: _T_FuncOrMethod) -> _T_FuncOrMethod:
    """
    Used for ``__init__`` operations in multiple inheritance scenarios.
    Should be used together with ``torchslime.utils.metaclasses.InitOnceMetaclass``.
    When ``__init__`` is decorated with ``InitOnce``, it will be called only once during 
    each instance creation. NOTE that there is an exception that if one ``__init__`` call
    raises an Exception and it is successfully caught and processed, this ``__init__`` 
    method may be called again by other methods. Because of this, ``InitOnce`` only ensure 
    'at most one successful call' rather than 'one call'.
    
    Example:
    
    ```Python
    class Example(metaclass=InitOnceMetaclass):
        @InitOnce
        def __init__(self, arg1, arg2):
            print('Example.__init__', arg1, arg2)
    
    class A(Example):
        def __init__(self):
            Example.__init__(self, arg1=1, arg2=2)
    
    class B(Example):
        def __init__(self):
            Example.__init__(self, arg1=3, arg2=4)
    
    class C(A, B):
        def __init__(self):
            A.__init__(self)
            B.__init__(self)
    
    C()
    
    \"\"\"
    Output:
    Example.__init__ 1 2
    \"\"\"
    ```
    """
    func_id = str(id(func))
    
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Union[_T, None]:
        init_once__: Union[Dict, Missing] = getattr(self, 'init_once__', MISSING)
        # whether the instance is being created.
        instance_creating = init_once__ is not MISSING
        # whether the instance is being created AND this ``__init__`` method has not been called.
        uninitialized = instance_creating and not init_once__.get(func_id, False)
        
        ret = None
        if not instance_creating or uninitialized:
            # call the ``__init__`` method.
            ret = func(self, *args, **kwargs)
        
        if uninitialized:
            """
            mark this ``__init__`` has been called.
            Note that it is after ``func`` is called, so ``InitOnce`` only ensure 
            'at most one successful call' rather than 'one call'.
            """
            init_once__[func_id] = True
        
        return ret

    return wrapper


def MetaclassCheck():
    pass

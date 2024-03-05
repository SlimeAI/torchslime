from functools import wraps
from .typing import (
    Union,
    Callable,
    TypeVar,
    overload,
    MISSING,
    Missing
)
from slime_core.utils.decorator import *

_T = TypeVar("_T")


# type hint
@overload
def CallDebug(
    _func: Missing = MISSING,
    *,
    module_name: Union[str, Missing] = MISSING,
) -> Callable[[_T], _T]: pass
@overload
def CallDebug(
    _func: _T,
    *,
    module_name: Union[str, Missing] = MISSING,
) -> _T: pass

@DecoratorCall(index=0, keyword='_func')
def CallDebug(
    _func: _T = MISSING,
    *,
    module_name: Union[str, Missing] = MISSING
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
        candidate_module_names = (
            module_name,
            getattr(func, '__qualname__', MISSING),
            getattr(func, '__name__', MISSING),
            'UNKNOWN MODULE'
        )
        for candidate_name in candidate_module_names:
            if candidate_name is not MISSING:
                module_name = candidate_name
                break
        
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
            
            exec_name = (
                _exec_info["full_exec_name"] if 
                store.builtin__().call_debug_full_exec_name else 
                _exec_info["exec_name"]
            )
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

from functools import wraps
import multiprocessing
import threading
from typing import Any, Union
from types import FunctionType, MethodType
from torchslime.utils.bases import NOTHING, is_none_or_nothing
from torchslime.utils import get_exec_info, is_function_or_method

#
# ClassWraps decorator
#
FUNC_CREATED = ('__module__', '__name__', '__qualname__')

def _create_func(
    func: Union[FunctionType, MethodType],
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
        setattr(func, '__qualname__', '{}.{}'.format(getattr(cls, '__qualname__'), name))
    return func

def ClassWraps(cls):
    if isinstance(cls, type) is False:
        from torchslime.components.exception import APIMisused
        raise APIMisused('ClassWraps can only be used for class, not {cls_item}.'.format(
            cls_item=str(cls)
        ))

    from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES

    class Decorator:

        def __getattribute__(self, __name: str) -> Any:
            __func = _get_function_or_method(cls, __name)

            def FuncWrapper(
                *,
                assigned=WRAPPER_ASSIGNMENTS,
                updated=WRAPPER_UPDATES,
                created=FUNC_CREATED
            ):
                def wrapper(func):
                    if is_none_or_nothing(__func) is False:
                        func = wraps(__func, assigned=assigned, updated=updated)(func)
                    else:
                        func = _create_func(func=func, cls=cls, name=__name, created=created)
                    setattr(cls, __name, func)
                return wrapper
            return FuncWrapper, get_cls_func(cls, __name), get_super_func(cls, __name)
    
    return Decorator()

def _get_function_or_method(cls: type, name: str):
    __item = cls.__dict__.get(name, NOTHING)
    return __item if is_function_or_method(__item) else NOTHING

def _get_func_from_mro(cls: type, name: str, start: int=0):
    # get attr from the super class
    for class__ in cls.__mro__[start:]:
        return getattr(class__, name, NOTHING)
    return NOTHING

def get_cls_func(cls: type, name: str):
    return _get_func_from_mro(cls, name, start=0)

def get_super_func(cls: type, name: str):
    return _get_func_from_mro(cls, name, start=1)


def Singleton(cls):
    """
    [class]
    Decorator that makes decorated classes singleton.
    It makes the creation of the singleton object thread-safe by using double-checked locking.
    """
    t_lock = threading.Lock()
    p_lock = multiprocessing.Lock()
    _instance = NOTHING

    class_wraps = ClassWraps(cls)
    new_wraps, cls_new, _ = class_wraps.__new__

    @new_wraps()
    def _wrapper(*args, **kwargs):
        nonlocal _instance
        if is_none_or_nothing(_instance) is True:
            with t_lock, p_lock:
                if is_none_or_nothing(_instance) is True:
                    _instance = cls_new(*args, **kwargs)
        return _instance
    
    return cls


def CallDebug(module_name):
    """
    [func]
    A decorator that output debug information before and after a method is called.
    Args:
        func (_type_): _description_
    """
    def decorator(func):
        from torchslime.log import logger
        from torchslime.components.store import store

        func_id = str(id(func))

        @wraps(func)
        def wrapper(*args, **kwargs):
            # do not use debug
            if store.scope__('inner__').use_call_debug is not True:
                return func(*args, **kwargs)

            # cache debug info
            call_debug_cache = store.scope__('inner__').call_debug_cache
            _exec_info = call_debug_cache[func_id]
            if is_none_or_nothing(_exec_info) is True:
                _exec_info = get_exec_info(func)
                call_debug_cache[func_id] = _exec_info

            logger.debug('{} begins.'.format(module_name), _exec_info=_exec_info)
            result = func(*args, **kwargs)
            logger.debug('{} ends.'.format(module_name), _exec_info=_exec_info)
            return result
        return wrapper
    return decorator


def MethodChaining(func):
    """
    [func]
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper


def Deprecated():
    pass


def ReadonlyAttr():
    pass

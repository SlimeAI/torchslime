"""
This python file defines common types that are used in the project.
The module is renamed from 'type' to 'typing' to avoid namespace conflict with built-in class ``type``
"""
import multiprocessing
import re
import sys
import threading
from types import FunctionType, MethodType
from typing import *

if sys.version_info < (3, 8):
    from typing_extensions import (
        SupportsIndex,
        TypedDict,
        Literal
    )

if sys.version_info >= (3, 9):
    Dict = dict
    List = list
    Set = set
    Frozenset = frozenset
    Tuple = tuple
    Type = type
    
    import collections
    DefaultDict = collections.defaultdict
    OrderedDict = collections.OrderedDict
    ChainMap = collections.ChainMap
    Counter = collections.Counter
    Deque = collections.deque
    
    import re
    Pattern = re.Pattern
    Match = re.Match
    
    import collections.abc as abc
    AbstractSet = abc.Set
    
    # deprecated type: ByteString
    try:
        import typing_extensions
        ByteString = typing_extensions.Buffer
    except Exception:
        ByteString = Union[bytes, bytearray, memoryview]
    
    Collection = abc.Collection
    Container = abc.Container
    ItemsView = abc.ItemsView
    KeysView = abc.KeysView
    Mapping = abc.Mapping
    MappingView = abc.MappingView
    MutableMapping = abc.MutableMapping
    MutableSequence = abc.MutableSequence
    MutableSet = abc.MutableSet
    Sequence = abc.Sequence
    ValuesView = abc.ValuesView
    
    Coroutine = abc.Coroutine
    AsyncGenerator = abc.AsyncGenerator
    AsyncIterable = abc.AsyncIterable
    AsyncIterator = abc.AsyncIterator
    Awaitable = abc.Awaitable

    Iterable = abc.Iterable
    Iterator = abc.Iterator
    Callable = abc.Callable
    Generator = abc.Generator
    Hashable = abc.Hashable
    Reversible = abc.Reversible
    Sized = abc.Sized
    
    import contextlib
    ContextManager = contextlib.AbstractContextManager
    AsyncContextManager = contextlib.AbstractAsyncContextManager

try:
    from typing import _overload_dummy
    overload_dummy: FunctionType = _overload_dummy
except Exception:
    def overload_dummy(): pass
    overload_dummy = overload(overload_dummy)

#
# Nothing class, NOTHING instance and related operations.
#

class _NothingSingleton(type):
    """
    Nothing Singleton should be implemented independently, because the ``Singleton`` decorator relies on the basic NOTHING object, which may cause circular reference.
    """

    __t_lock = threading.Lock()
    __p_lock = multiprocessing.Lock()
    __instance = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.__instance is None:
            with self.__t_lock, self.__p_lock:
                if self.__instance is None:
                    self.__instance = super().__call__(*args, **kwargs)
        return self.__instance

class Nothing(metaclass=_NothingSingleton):
    """
    'Nothing' object, different from python 'None'.
    It often comes from getting properties or items that the object does not have, or simply represents a default value.
    'Nothing' allows any attribute-get or method-call operations without throwing Errors, making the program more stable.
    It will show Warnings in the console instead.
    """
    __slots__ = ()

    def __init__(self): super().__init__()
    def __call__(self, *args, **kwargs): return self
    def __getattribute__(self, *_): return self
    def __getitem__(self, *_): return self
    def __setattr__(self, *_): pass
    def __setitem__(self, *_): pass
    def __len__(self): return 0
    def __iter__(self): return self
    def __next__(self): raise StopIteration
    def __str__(self) -> str: return 'NOTHING'
    def __repr__(self) -> str: return f'NOTHING<{str(hex(id(self)))}>'
    def __format__(self, __format_spec: str) -> str: return 'NOTHING'
    def __contains__(self, _) -> bool: return False

    def __eq__(self, obj) -> bool:
        if obj is NOTHING:
            return True
        return False

    def __add__(self, _): return self
    def __sub__(self, _): return self
    def __mul__(self, _): return self
    def __truediv__(self, _): return self
    def __radd__(self, _): return self
    def __rsub__(self, _): return self
    def __rmul__(self, _): return self
    def __rtruediv__(self, _): return self
    def __int__(self) -> int: return 0
    def __index__(self) -> int: return 0
    def __float__(self): return 0.0
    def __bool__(self) -> bool: return False

NOTHING = Nothing()
NoneOrNothing = Union[None, Nothing]

def is_none_or_nothing(obj) -> bool:
    """Check whether an object is None, Nothing or neither.
    Args:
        obj (Any): object
    Returns:
        bool: check result.
    """
    return obj is None or obj is NOTHING


from torch import Tensor
from torch.nn import Module

# tensor or module
T_M = Union[Tensor, Module]
# tensor or module or their sequence
T_M_SEQ = Union[T_M, Sequence[T_M]]
# int or float
NUMBER = Union[int, float]
# int or float. tuple
NUMBER_T = (int, float)
# int or sequence of int
INT_SEQ = Union[int, Sequence[int]]

# int, sequence of int, None or NOTHING
INT_SEQ_N = Union[INT_SEQ, None]


FuncOrMethod = Union[FunctionType, MethodType]
RawFunc = FunctionType

def is_function_or_method(__item: Any) -> bool:
    return isinstance(__item, (MethodType, FunctionType))


MAGIC_PATTERN = re.compile('^_{2}[^_](?:.*[^_])?_{2}$')

def is_magic_naming(__name: str) -> bool:
    return re.match(MAGIC_PATTERN, str(__name)) is not None


SLIME_PATTERN = re.compile('^[^_](?:.*[^_])?_{2}$')

def is_slime_naming(__name: str) -> bool:
    return re.match(SLIME_PATTERN, str(__name)) is not None


def create_singleton(__name: str) -> Tuple[Type[object], object]:
    """
    Create a new singleton class with its singleton object. Mostly used in flag vars.
    """
    from .decorators import Singleton, ClassWraps

    new_class = type(__name, (object,), {})

    # set str and repr func
    class_wraps = ClassWraps(new_class)
    @class_wraps.__str__
    def str_func(self) -> str:
        return __name

    @class_wraps.__repr__
    def repr_func(self) -> str:
        return f'{__name}<{str(hex(id(self)))}>'

    new_class = Singleton(new_class)
    singleton_object = new_class()
    return new_class, singleton_object


# ``Pass`` singleton constant
Pass, PASS = create_singleton('PASS')
Pass: Type[object]

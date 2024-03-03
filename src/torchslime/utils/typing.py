"""
This Python module defines common types that are used in the project, 
provides version compatibility for Python and introduces special constants 
in torchslime.
"""
import multiprocessing
import re
import sys
import threading
from types import FunctionType, MethodType
from typing import *

#
# Typing import for version compatibility.
#

if sys.version_info < (3, 8):
    try:
        from typing_extensions import (
            SupportsIndex,
            TypedDict,
            Literal,
            Protocol,
            runtime_checkable
        )
    except Exception:
        print(
            'Loading ``typing_extensions`` module failed. '
            'Please make sure you have installed it correctly.'
        )
        raise

if sys.version_info < (3, 9):
    # FIX: ``from typing import *`` does not include the following modules under Python 3.9
    from typing import (
        BinaryIO,
        IO,
        Match,
        Pattern,
        TextIO
    )

if sys.version_info >= (3, 9):
    from builtins import (
        dict as Dict,
        list as List,
        set as Set,
        frozenset as Frozenset,
        tuple as Tuple,
        type as Type,
        # for compatibility for Python 2.x
        str as Text
    )
    
    from collections import (
        defaultdict as DefaultDict,
        OrderedDict as OrderedDict,
        ChainMap as ChainMap,
        Counter as Counter,
        deque as Deque
    )
    
    from re import (
        Pattern as Pattern,
        Match as Match
    )
    
    from collections.abc import (
        Set as AbstractSet,
        Collection as Collection,
        Container as Container,
        ItemsView as ItemsView,
        KeysView as KeysView,
        Mapping as Mapping,
        MappingView as MappingView,
        MutableMapping as MutableMapping,
        MutableSequence as MutableSequence,
        MutableSet as MutableSet,
        Sequence as Sequence,
        ValuesView as ValuesView,
        Coroutine as Coroutine,
        AsyncGenerator as AsyncGenerator,
        AsyncIterable as AsyncIterable,
        AsyncIterator as AsyncIterator,
        Awaitable as Awaitable,
        Iterable as Iterable,
        Iterator as Iterator,
        Callable as Callable,
        Generator as Generator,
        Hashable as Hashable,
        Reversible as Reversible,
        Sized as Sized
    )
    
    # deprecated type: ByteString
    try:
        from typing_extensions import (
            Buffer as ByteString
        )
    except Exception:
        ByteString = Union[bytes, bytearray, memoryview]
    
    from contextlib import (
        AbstractContextManager as ContextManager,
        AbstractAsyncContextManager as AsyncContextManager
    )

try:
    from typing import _overload_dummy as overload_dummy
except Exception:
    @overload
    def overload_dummy(): pass

#
# Special constants defined in torchslime.
#

class _SingletonMetaclass(type):
    """
    Singleton metaclass that makes a specific class a singleton class.
    
    Used for special constants. It is defined here rather than in ``torchslime.utils.metaclass``, because 
    the typing module should be an independent module and can only be imported by other torchslime modules 
    (to avoid circular import error). The ``SingletonMetaclass`` in ``torchslime.utils.metaclass`` and 
    ``Singleton`` in ``torchslime.utils.base`` are just based on this class for more general use.
    
    NOTE: The ``_SingletonMetaclass`` works for each class (even subclasses) independently, because it sets 
    locks and ``__instance`` separately for each class it creates.
    """
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.__t_lock = threading.Lock()
        cls.__p_lock = multiprocessing.Lock()
        cls.__instance = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.__instance is None:
            with self.__t_lock, self.__p_lock:
                if self.__instance is None:
                    self.__instance = super().__call__(*args, **kwargs)
        return self.__instance


# ``Nothing`` class, ``NOTHING`` instance and related functions.

class Nothing(metaclass=_SingletonMetaclass):
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
    def __enter__(self) -> 'Nothing': return self
    def __exit__(self, *args, **kwargs): return

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

#
# Flag constants.
#

class _FlagConstant(metaclass=_SingletonMetaclass):
    def __str__(self) -> str: return self.__class__.__name__.upper()
    def __repr__(self) -> str: return f'{str(self)}<{str(hex(id(self)))}>'

# ``Pass`` singleton constant
class Pass(_FlagConstant): pass
PASS = Pass()

# ``Missing`` singleton constant
class Missing(_FlagConstant):
    def __bool__(self) -> bool:
        return False

MISSING = Missing()

# ``Stop`` singleton constant
class Stop(_FlagConstant):
    def __bool__(self) -> bool:
        return False

STOP = Stop()

#
# Other types, type checking, naming checking, type parsing, etc.
#

FuncOrMethod = Union[FunctionType, MethodType]
RawFunc = FunctionType

def is_function_or_method(__item: Any) -> bool:
    return isinstance(__item, (MethodType, FunctionType))


MAGIC_PATTERN = re.compile('^_{2}[^_](?:.*[^_])?_{2}$')

def is_magic_naming(__name: str) -> bool:
    return MAGIC_PATTERN.match(str(__name)) is not None


SLIME_PATTERN = re.compile('^[^_](?:.*[^_])?_{2}$')

def is_slime_naming(__name: str) -> bool:
    return SLIME_PATTERN.match(str(__name)) is not None


@overload
def unwrap_method(func: FuncOrMethod) -> RawFunc: pass
@overload
def unwrap_method(func: NoneOrNothing) -> NoneOrNothing: pass

def unwrap_method(func: Union[FuncOrMethod, NoneOrNothing]) -> Union[RawFunc, NoneOrNothing]:
    """
    Get the original static function if the given ``func`` is a method.
    """
    if isinstance(func, MethodType):
        # get the original function body of the method
        func = func.__func__
    return func


def get_mro(cls: Type) -> Tuple[Type, ...]:
    """
    Safely get the mro of any given class. NOTE: If the class has the 
    attribute ``__mro__``, then directly return it. Otherwise, call the 
    corresponding ``mro()`` method to the the mro.
    """
    # If ``cls`` has ``__mro__``, then directly return.
    if hasattr(cls, '__mro__'):
        return cls.__mro__
    
    try:
        # NOTE: Some class (e.g., typing.Sequence) doesn't support subclass 
        # check, and ``issubclass`` will raise an exception.
        is_type_subclass = issubclass(cls, type)
    except Exception:
        is_type_subclass = False
    
    if is_type_subclass:
        # NOTE: If the given class is a metaclass, then the corresponding 
        # ``mro`` method to be called should be in the 'metaclass of the 
        # given metaclass' (i.e., type(cls)).
        return tuple(type(cls).mro(cls))
    else:
        # Normal classes simply call the ``mro`` method.
        return tuple(cls.mro())


def get_bases(cls: Type) -> Tuple[Type, ...]:
    """
    Safely get the bases of any given class. NOTE: If the class has the 
    attribute ``__bases__``, then directly return it. Otherwise (e.g., 
    typing.Sequence), this function resolves the mro and returns the 
    'minimal base class set'.
    
    The 'minimal base class set' denotes that there doesn't exist inheritance 
    relationship in this set and the set only keeps the most subclasses classes.
    
    Example:
    
    ```Python
    class A: pass
    
    class B(A): pass
    
    class C(B, A): pass
    
    # The bases of class ``C`` is (B, A)
    print(C.__bases__)
    # However, the 'minimal base class set' of ``C`` is (B,) according to the 
    # definition.
    
    # ``get_bases(C)`` still returns (B, A) because class ``C`` has attribute 
    # ``__bases__``
    print(get_bases(C))
    
    # NOTE: ``typing.Sequence`` is different from ``collections.abc.Sequence`` 
    # and it doesn't have ``__bases__`` or ``__mro__``.
    from typing import Sequence
    print(hasattr(Sequence, '__bases__'))
    print(hasattr(Sequence, '__mro__'))
    print(get_bases(Sequence))
    
    # Output:
    # False
    # False
    # (<class 'collections.abc.Sequence'>,)
    ```
    """
    # If ``cls`` has ``__bases__``, then directly return.
    if hasattr(cls, '__bases__'):
        return cls.__bases__

    # Get the mro of ``cls`` (excluding itself).
    mro_classes = list(filter(lambda _cls: _cls is not cls, get_mro(cls)))
    bases = []
    while len(mro_classes) > 0:
        # NOTE: should pop the first element in the list (index=0).
        base = mro_classes.pop(0)
        bases.append(base)
        # Get the mro of ``base`` and remove them from ``mro_classes``.
        base_mro_set = set(get_mro(base))
        mro_classes = list(filter(lambda _cls: _cls not in base_mro_set, mro_classes))
    return tuple(bases)

#
# Torch version adapter
#

try:
    from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler
except Exception:
    from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler

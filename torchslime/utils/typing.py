"""
This python file defines common types that are used in the project.
The module is renamed from 'type' to 'typing' to avoid namespace conflict with built-in class ``type``
"""
import sys
from typing import *

if sys.version_info < (3, 8):
    from typing_extensions import (
        SupportsIndex,
        TypedDict
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
    import typing_extensions
    try:
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

from . import dict_merge
import traceback
from .typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    MutableSequence,
    MutableMapping,
    Iterable,
    Iterator,
    TypeVar,
    Generic,
    overload,
    SupportsIndex,
    Type,
    Generator,
    Callable
)
from functools import partial
import threading
import multiprocessing
from types import TracebackType


# TypeVars
_T = TypeVar('_T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class Base:
    """
    Base class, making its subclasses be able to use '[]' operations(just like python dict).
    Return 'Nothing' if the object does not have the property being retrieved, without throwing Errors.
    What's more, it allows its subclasses assign properties using a dict.
    """

    def from_kwargs__(self, **kwargs):
        self.from_dict__(kwargs)

    def from_dict__(self, _dict: Dict):
        """assign properties to the object using a dict.
        Args:
            kwargs (Dict): property dict.
        """
        self.__dict__ = dict_merge(self.__dict__, _dict)

    def check__(self, item: str):
        """check whether the object has a specific attribute.
        dot operator supported.
        Args:
            items (str): _description_
        """
        attrs = item.split('.')
        temp = self
        for attr in attrs:
            try:
                temp = temp[attr]
                # if the value is NOTHING, then return False directly.
                if temp is NOTHING:
                    return False
            except Exception:
                # output error information
                self.process_exc__()
                return False
        return True

    def hasattr__(self, __name: str) -> bool:
        return str(__name) in self.__dict__

    @staticmethod
    def process_exc__():
        from torchslime.log import logger
        # output error
        logger.error(
            'Python exception raised:\n' +
            traceback.format_exc()
        )
        return NOTHING

    def pop__(self, __name: str):
        attr = getattr(self, __name)
        delattr(self, __name)
        return attr

    def __getattr__(self, *_):
        return NOTHING

    def __getattribute__(self, __name: str):
        return super().__getattribute__(str(__name))

    def __setattr__(self, __name: str, __value: Any) -> None:
        try:
            super().__setattr__(str(__name), __value)
        except Exception:
            return

    def __delattr__(self, __name: str) -> None:
        # safe delete
        try:
            super().__delattr__(str(__name))
        except Exception:
            return

    def __getitem__(self, __name: str):
        try:
            return getattr(self, __name)
        except Exception:
            return self.process_exc__()

    def __setitem__(self, __name: str, __value: Any):
        try:
            return setattr(self, __name, __value)
        except Exception:
            return self.process_exc__()

    def __delitem__(self, __name: str):
        try:
            return delattr(self, __name)
        except Exception:
            return
    
    def __str__(self) -> str:
        from .formatter import dict_to_key_value_str
        classname=str(self.__class__.__name__)
        _id=str(hex(id(self)))
        _dict=dict_to_key_value_str(self.__dict__)
        return f'{classname}<{_id}>({_dict})'


class BaseList(MutableSequence[_T], Generic[_T]):

    def __init__(
        self,
        __list_like: Union[Iterable[_T], None, 'Nothing'] = None
    ):
        self.__list: List[_T] = []
        if not is_none_or_nothing(__list_like):
            # Use ``self.extend`` here to make the initialization process controllable.
            # Otherwise, if ``self.__list = list(__list_like)`` is used here, the initialization process won't be restricted by the user-defined operations.
            self.extend(__list_like)

    @classmethod
    def create__(
        cls,
        __list_like: Union[_T, Iterable[_T], None, 'Nothing', 'Pass'] = None,
        *,
        strict = False,
        return_none: bool = True,
        return_nothing: bool = True,
        return_pass: bool = True
    ):
        # TODO: update document
        """
        If the ``list_like`` object is ``None``, ``NOTHING`` or ``...`` and the corresponding return config is True, then
        return itself, otherwise return ``BaseList`` object.
        WARNING: This changes the default behavior of ``BaseList``, which creates an empty list when the list_like object is 
        ``None`` or ``NOTHING`` and creates ``[...]`` when the list_like object is ``...``.
        """
        if (__list_like is NOTHING and return_nothing is True) or \
                (__list_like is None and return_none is True) or \
                (__list_like is PASS and return_pass is True):
            # return the item itself
            __list_like: Union[None, Nothing, Pass]
            return __list_like
        elif isinstance(__list_like, Iterable) or is_none_or_nothing(__list_like):
            return cls(__list_like)
        
        if strict:
            classname = type(__list_like).__name__
            raise TypeError(f'BaseList - ``strict`` is True and ``{classname}`` object is not iterable')
        else:
            return cls([__list_like])

    def set_list__(self, __list: List[_T]) -> None:
        self.__list = __list

    def get_list__(self) -> List[_T]:
        return self.__list
    
    @overload
    def __getitem__(self, __i: SupportsIndex) -> _T: pass
    @overload
    def __getitem__(self, __s: slice) -> List[_T]: pass
    @overload
    def __setitem__(self, __key: SupportsIndex, __value: _T) -> None: pass
    @overload
    def __setitem__(self, __key: slice, __value: Iterable[_T]) -> None: pass
    @overload
    def __delitem__(self, __key: Union[SupportsIndex, slice]) -> None: pass
    @overload
    def insert(self, __index: SupportsIndex, __object: _T) -> None: pass
    
    def __getitem__(self, __key):
        return self.__list[__key]
    
    def __setitem__(self, __key, __value):
        self.__list[__key] = __value
    
    def __delitem__(self, __key):
        del self.__list[__key]
    
    def __len__(self):
        return len(self.__list)
    
    def insert(self, __index, __object):
        return self.__list.insert(__index, __object)
    
    def __str__(self) -> str:
        classname=str(self.__class__.__name__)
        _id=str(hex(id(self)))
        _list=str(self.__list)
        return f'{classname}<{_id}>({_list})'


class BaseDict(MutableMapping[_KT, _VT], Generic[_KT, _VT]):

    def __init__(
        self,
        __dict_like: Union[Dict[_KT, _VT], Iterable[Tuple[_KT, _VT]], None, 'Nothing'] = None,
        **kwargs
    ):
        self.__dict: Dict[_KT, _VT] = {}
        if is_none_or_nothing(__dict_like):
            __dict_like = {}
        # Use ``self.update`` here to make the initialization process controllable.
        # Otherwise, if ``self.__dict = dict(__dict_like, **kwargs)`` is used here, the initialization process won't be restricted by the user-defined operations.
        self.update(__dict_like, **kwargs)

    def set_dict__(self, __dict: Dict[_KT, _VT]) -> None:
        self.__dict = __dict

    def get_dict__(self) -> Dict[_KT, _VT]:
        return self.__dict
    
    @overload
    def __getitem__(self, __key: _KT) -> _VT: pass
    @overload
    def __setitem__(self, __key: _KT, __value: _VT) -> None: pass
    @overload
    def __delitem__(self, __key: _KT) -> None: pass
    @overload
    def __iter__(self) -> Iterator[_KT]: pass
    @overload
    def __len__(self) -> int: pass
    
    def __getitem__(self, __key):
        return self.__dict[__key]
    
    def __setitem__(self, __key, __value):
        self.__dict[__key] = __value
    
    def __delitem__(self, __key):
        del self.__dict[__key]
    
    def __iter__(self):
        return iter(self.__dict)
    
    def __len__(self):
        return len(self.__dict)
    
    def __str__(self) -> str:
        classname=str(self.__class__.__name__)
        _id=str(hex(id(self)))
        _dict=str(self.__dict)
        return f'{classname}<{_id}>({_dict})'

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
    def __contains__(self) -> bool: return False

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

def is_none_or_nothing(obj) -> bool:
    """Check whether an object is None, Nothing or neither.
    Args:
        obj (Any): object
    Returns:
        bool: check result.
    """
    return obj is None or obj is NOTHING


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

# Type Vars
_YieldT_co = TypeVar('_YieldT_co', covariant=True)
_SendT_contra = TypeVar('_SendT_contra', contravariant=True)
_ReturnT_co = TypeVar('_ReturnT_co', covariant=True)

class BaseGenerator(
    Generator[_YieldT_co, _SendT_contra, _ReturnT_co],
    Generic[_YieldT_co, _SendT_contra, _ReturnT_co]
):

    def __init__(
        self,
        __gen: Generator[_YieldT_co, _SendT_contra, _ReturnT_co],
        *,
        exit_allowed: bool = True
    ) -> None:
        if not isinstance(__gen, Generator):
            raise TypeError(f'Argument ``__gen`` should be a generator.')
        self.gen = __gen
        self.exit_allowed = exit_allowed
        
        self.exit = False

    def __call__(self) -> Any:
        return next(self)

    def send(self, __value: _SendT_contra) -> _YieldT_co:
        return self.call__(partial(self.gen.send, __value))

    @overload
    def throw(
        self,
        __typ: Type[BaseException],
        __val: Union[BaseException, object] = None,
        __tb: Union[TracebackType, None] = None
    ) -> _YieldT_co: pass
    @overload
    def throw(
        self,
        __typ: BaseException,
        __val: None = None,
        __tb: Union[TracebackType, None] = None
    ) -> _YieldT_co: pass

    def throw(self, __typ, __val=None, __tb=None) -> _YieldT_co:
        return self.call__(partial(self.gen.throw, __typ, __val, __tb))

    def call__(self, __caller: Callable[[], _T]) -> Union[_T, Nothing]:
        if self.exit and not self.exit_allowed:
            from torchslime.components.exception import APIMisused
            raise APIMisused('``exit_allowed`` is set to False, and the generator already stopped but you still try to call ``next``.')
        elif self.exit:
            return NOTHING

        try:
            return __caller()
        except (StopIteration, GeneratorExit):
            self.exit = True

import traceback
from typing import Any
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
    Callable,
    NOTHING,
    Nothing,
    NoneOrNothing,
    Pass,
    PASS,
    is_none_or_nothing
)
import torchslime.utils as utils
from functools import partial
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
        self.__dict__ = utils.dict_merge(self.__dict__, _dict)

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
        from torchslime.logging.logger import logger
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

    def __delattr__(self, __name: str) -> None:
        # safe delete
        try:
            return super().__delattr__(__name)
        except AttributeError:
            return

    def __getitem__(self, __name: str):
        return getattr(self, __name)

    def __setitem__(self, __name: str, __value: Any):
        return setattr(self, __name, __value)

    def __delitem__(self, __name: str):
        return delattr(self, __name)
    
    def __str__(self) -> str:
        from .formatter import dict_to_key_value_str
        classname=str(self.__class__.__name__)
        _id=str(hex(id(self)))
        _dict=dict_to_key_value_str(self.__dict__)
        return f'{classname}<{_id}>({_dict})'


class BaseList(MutableSequence[_T], Generic[_T]):

    def __init__(
        self,
        __list_like: Union[Iterable[_T], NoneOrNothing] = None
    ):
        self.__list: List[_T] = []
        if not is_none_or_nothing(__list_like):
            # Use ``self.extend`` here to make the initialization process controllable.
            # Otherwise, if ``self.__list = list(__list_like)`` is used here, the initialization process won't be restricted by the user-defined operations.
            self.extend(__list_like)

    @classmethod
    def create__(
        cls,
        __list_like: Union[_T, Iterable[_T], NoneOrNothing, Pass] = None,
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
            __list_like: Union[NoneOrNothing, Pass]
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
        __dict_like: Union[Dict[_KT, _VT], Iterable[Tuple[_KT, _VT]], NoneOrNothing] = None,
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


class BaseProxy(Generic[_T]):
    
    def __init__(
        self,
        __obj: _T,
        __attrs: List[str]
    ) -> None:
        super().__init__()
        self.obj__ = __obj
        self.attrs__ = __attrs
    
    def __getattribute__(self, __name: str) -> Any:
        if __name in ['obj__', 'attrs__']:
            return super().__getattribute__(__name)
        # attr proxy
        if __name in self.attrs__:
            return getattr(self.obj__, __name)
        return super().__getattribute__(__name)

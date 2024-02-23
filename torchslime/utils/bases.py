import traceback
from .typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Sequence,
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
    is_none_or_nothing,
    Set,
    Missing,
    MISSING,
    unwrap_method
)
from .decorators import (
    DecoratorCall,
    InitOnce
)
from .metaclasses import (
    InitOnceMetaclass,
    SingletonMetaclass,
    _ReadonlyAttrMetaclass
)
from contextlib import ContextDecorator
from functools import partial
from types import TracebackType
import re

# TypeVars
_T = TypeVar('_T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

#
# Scoped Attribute
#

class ScopedAttr:
    
    def __init__(self) -> None: pass
    
    def assign__(self, **attr_assign) -> "ScopedAttrAssign":
        return ScopedAttrAssign(self, attr_assign)
    
    def restore__(self, *attrs: str) -> "ScopedAttrRestore":
        return ScopedAttrRestore(self, attrs)

#
# ItemAttrBinding
#

class ItemAttrSetBinding:
    
    def __setitem__(self, __name: str, __value: Any) -> None:
        return setattr(self, __name, __value)


class ItemAttrGetBinding:
    
    def __getitem__(self, __name: str) -> Any:
        return getattr(self, __name)


class ItemAttrDelBinding:
    
    def __delitem__(self, __name: str) -> None:
        return delattr(self, __name)


class ItemAttrBinding(
    ItemAttrSetBinding,
    ItemAttrGetBinding,
    ItemAttrDelBinding
):
    """
    """
    pass

#
# Base
#

class Base(ScopedAttr, ItemAttrBinding):
    """
    Base class, making its subclasses be able to use '[]' operations(just like python dict).
    Return 'Nothing' if the object does not have the property being retrieved, without throwing Errors.
    What's more, it allows its subclasses assign properties using a dict.
    """

    def __init__(self) -> None:
        super().__init__()

    def from_kwargs__(self, **kwargs):
        self.from_dict__(kwargs)

    def from_dict__(self, _dict: Dict):
        """assign properties to the object using a dict.
        Args:
            kwargs (Dict): property dict.
        """
        from .common import dict_merge
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
    
    def __str__(self) -> str:
        from .common import dict_to_key_value_str
        classname=str(self.__class__.__name__)
        _id=str(hex(id(self)))
        _dict=dict_to_key_value_str(self.__dict__)
        return f'{classname}<{_id}>({_dict})'

#
# Base List
#

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
        # NOTE: ``isinstance(NOTHING, Iterable)`` will raise TypeError, 
        # because ``NOTHING.__class__`` will still return ``NOTHING`` instance, 
        # and it will fail in the ``issubclass`` function.
        elif is_none_or_nothing(__list_like) or isinstance(__list_like, Iterable):
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

#
# Bidirectional List
#

class BiListItem:
    
    def __init__(self) -> None:
        self.__parent = NOTHING
    
    def set_parent__(self, parent) -> None:
        prev_parent = self.get_parent__()
        if not is_none_or_nothing(prev_parent) and parent is not prev_parent:
            # duplicate parent
            from torchslime.logging.logger import logger
            logger.warning(
                f'BiListItem ``{str(self)}`` has already had a parent, but another parent is set. '
                'This may be because you add a single BiListItem object to multiple BiLists '
                'and may cause some inconsistent problems.'
            )
        self.__parent = parent
    
    def get_parent__(self) -> Union["BiList", Nothing]:
        return self.__parent if hasattr(self, '_BiListItem__parent') else NOTHING
    
    def get_verified_parent__(self) -> Union["BiList", Nothing]:
        from torchslime.logging.logger import logger
        
        parent = self.get_parent__()
        if parent is NOTHING:
            # root node
            logger.warning(f'BiListItem ``{str(self)}`` does not have a parent.')
            return NOTHING
        if self not in parent:
            # unmatched parent
            logger.warning(f'BiListItem ``{str(self)}`` is not contained in its specified parent.')
            self.del_parent__()
            return NOTHING
        return parent
    
    def del_parent__(self):
        self.__parent = NOTHING


class MutableBiListItem(BiListItem):
    
    def replace_self__(self, __item: 'MutableBiListItem') -> None:
        parent = self.get_verified_parent__()
        index = parent.index(self)
        parent[index] = __item
    
    def insert_before_self__(self, __item: 'MutableBiListItem') -> None:
        parent = self.get_verified_parent__()
        index = parent.index(self)
        parent.insert(index, __item)
    
    def insert_after_self__(self, __item: 'MutableBiListItem') -> None:
        parent = self.get_verified_parent__()
        index = parent.index(self)
        parent.insert(index + 1, __item)
    
    def remove_self__(self) -> None:
        parent = self.get_verified_parent__()
        parent.remove(self)


_T_BiListItem = TypeVar('_T_BiListItem', bound=BiListItem)

class BiList(BaseList[_T_BiListItem]):
    
    def set_list__(self, __list: List[_T_BiListItem]) -> None:
        prev_list = self.get_list__()
        
        for prev_item in prev_list:
            prev_item.del_parent__()
        
        for item in __list:
            item.set_parent__(self)
        
        return super().set_list__(__list)

    @overload
    def __setitem__(self, __key: SupportsIndex, __value: _T_BiListItem) -> None: pass
    @overload
    def __setitem__(self, __key: slice, __value: Iterable[_T_BiListItem]) -> None: pass
    
    def __setitem__(
        self,
        __key: Union[SupportsIndex, slice],
        __value: Union[_T_BiListItem, Iterable[_T_BiListItem]]
    ) -> None:
        # delete parents of the replaced items and set parents to the replacing items
        if isinstance(__key, slice):
            for replaced_item in self[__key]:
                replaced_item.del_parent__()
            
            for item in __value:
                item: _T_BiListItem
                item.set_parent__(self)
        else:
            self[__key].del_parent__()
            __value: _T_BiListItem
            __value.set_parent__(self)
        return super().__setitem__(__key, __value)
    
    @overload
    def __delitem__(self, __key: SupportsIndex) -> None: pass
    @overload
    def __delitem__(self, __key: slice) -> None: pass
    
    def __delitem__(self, __key: Union[SupportsIndex, slice]) -> None:
        if isinstance(__key, slice):
            for item in self[__key]:
                item.del_parent__()
        else:
            self[__key].del_parent__()
        return super().__delitem__(__key)
    
    def insert(self, __index: SupportsIndex, __item: _T_BiListItem) -> None:
        __item.set_parent__(self)
        return super().insert(__index, __item)

#
# Base Dict
#

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

#
# Base Generator
#

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


_GeneratorT = TypeVar('_GeneratorT', bound=Generator)

class GeneratorQueue(BaseList[_GeneratorT]):
    
    def __init__(
        self,
        __generators: Union[Iterable[_GeneratorT], NoneOrNothing] = None,
        *,
        reverse: bool = False
    ):
        super().__init__(__generators)
        self.reverse = reverse
    
    def pop_gen__(self):
        while len(self) > 0:
            yield self.pop(
                -1 if self.reverse else 0
            )
    
    def __enter__(self):
        return self
    
    def __exit__(self, __type, __val, __tb):
        if not __type and not __val and not __tb:
            return True
        
        exception = (__type, __val, __tb)
        for gen in self.pop_gen__():
            try:
                gen.throw(*exception)
            except Exception as e:
                exception = (
                    e.__class__,
                    e,
                    e.__traceback__
                )
            else:
                exception = NOTHING
                break
        
        if is_none_or_nothing(exception):
            return True
        elif exception[1] is __val:
            return False
        else:
            raise exception[1]

#
# Composite Structure
#

class CompositeStructure:
    
    def composite_iterable__(self) -> Union[Iterable["CompositeStructure"], Nothing]: pass


_T_CompositeStructure = TypeVar('_T_CompositeStructure', bound=CompositeStructure)

def CompositeDFT(
    __item: _T_CompositeStructure,
    __func: Callable[[_T_CompositeStructure], None]
) -> None:
    stack: List[Union[Iterator[_T_CompositeStructure], Nothing]] = [iter([__item])]
    
    while len(stack) > 0:
        node_iter = stack[-1]
        
        try:
            node = next(node_iter)
        except StopIteration:
            stack.pop()
            continue
        
        __func(node)
        stack.append(iter(node.composite_iterable__()))


def CompositeDFS(
    __item: _T_CompositeStructure,
    __func: Callable[[_T_CompositeStructure], bool]
) -> List[_T_CompositeStructure]:
    results = []
    
    def _search(item):
        if __func(item):
            results.append(item)
    
    CompositeDFT(__item, _search)
    return results


def CompositeBFT(
    __item: _T_CompositeStructure,
    __func: Callable[[_T_CompositeStructure], None]
) -> None:
    queue: List[Union[Iterator[_T_CompositeStructure], Nothing]] = [iter([__item])]
    
    while len(queue) > 0:
        node_iter = queue[0]
        
        try:
            node = next(node_iter)
        except StopIteration:
            queue.pop(0)
            continue
        
        __func(node)
        queue.append(iter(node.composite_iterable__()))


def CompositeBFS(
    __item: _T_CompositeStructure,
    __func: Callable[[_T_CompositeStructure], bool]
) -> List[_T_CompositeStructure]:
    results = []
    
    def _search(item):
        if __func(item):
            results.append(item)
    
    CompositeBFT(__item, _search)
    return results

#
# Attr Proxy
#

class AttrProxy(Generic[_T]):
    
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

#
# Attr Observer
#

OBSERVE_FUNC_SUFFIX = '_observe__'
OBSERVE_FUNC_SUFFIX_PATTERN = re.compile(f'{OBSERVE_FUNC_SUFFIX}$')
OBSERVE_INIT = 'observe_init__'
OBSERVE_NAMESPACE = 'observe_namespace__'
ObserveFuncType = Callable[[Any, Any, "AttrObservable"], None]


class _ObservableInfo:
    
    def __init__(
        self,
        observable: "AttrObservable",
        attr_set: Union[Set[str], Missing] = MISSING
    ) -> None:
        self.observable = observable
        self.attr_set: Set[str] = set() if attr_set is MISSING else attr_set
    
    def add_attr__(self, __name: str) -> None:
        return self.attr_set.add(__name)
    
    def remove_attr__(self, __name: str) -> None:
        if __name in self.attr_set:
            self.attr_set.remove(__name)
    
    def is_empty__(self) -> bool:
        return len(self.attr_set) < 1


class _ObservableDict(BaseDict[str, _ObservableInfo]):
    
    @staticmethod
    def get_observable_id__(__observable: "AttrObservable") -> str:
        # this behavior may change through different torchslime versions
        return str(id(__observable))
    
    def add__(self, __observable: "AttrObservable", __name: str) -> None:
        observable_id = self.get_observable_id__(__observable)
        
        if observable_id not in self:
            self[observable_id] = _ObservableInfo(__observable)
        self[observable_id].add_attr__(__name)
    
    def remove__(self, __observable: "AttrObservable", __name: str) -> None:
        observable_id = self.get_observable_id__(__observable)
        
        if observable_id in self:
            observable_info = self[observable_id]
            observable_info.remove_attr__(__name)
            if observable_info.is_empty__():
                del self[observable_id]
    
    def get__(self, __observable: "AttrObservable") -> Set[str]:
        observable_id = self.get_observable_id__(__observable)
        
        if observable_id in self:
            return self[observable_id].attr_set
        else:
            return set()
    
    def contains__(self, __observable: "AttrObservable") -> bool:
        return self.get_observable_id__(__observable) in self


class AttrObserver(metaclass=InitOnceMetaclass):
    
    @InitOnce
    def __init__(self) -> None:
        self.__observable_dict = _ObservableDict()
    
    @staticmethod
    def check_namespace__(
        func: ObserveFuncType,
        namespaces: Union[Sequence[str], Missing, NoneOrNothing]
    ) -> bool:
        return (
            # ``None`` or ``NOTHING`` namespace won't match any function.
            not is_none_or_nothing(namespaces) and \
            (
                # ``MISSING`` namespace will match all the functions.
                namespaces is MISSING or \
                # Otherwise check if the function's namespace exists in ``namespaces``.
                getattr(unwrap_method(func), OBSERVE_NAMESPACE, MISSING) in namespaces
            )
        )
    
    def detach_inspect__(
        self,
        namespaces: Union[Sequence[str], Missing, NoneOrNothing] = MISSING
    ) -> Dict[str, ObserveFuncType]:
        return self.observe_inspect__(
            # Check namespace.
            lambda func: self.check_namespace__(func, namespaces)
        )
    
    def attach_inspect__(
        self,
        namespaces: Union[Sequence[str], Missing, NoneOrNothing] = MISSING
    ) -> Dict[str, ObserveFuncType]:
        return self.observe_inspect__(
            # Check namespace.
            lambda func: self.check_namespace__(func, namespaces)
        )
    
    def observe_inspect__(
        self,
        __func: Callable[[ObserveFuncType], bool]
    ) -> Dict[str, ObserveFuncType]:
        observe_dict: Dict[str, ObserveFuncType] = {}
        """
        NOTE: ``func_name`` here is actually the attribute name in the object, rather than the 
        real function name.
        
        Example:
        
        ```Python
        class A: pass
        
        def b(): pass
        
        a = A()
        a.c = b  # The ``func_name`` is ``c`` rather than ``b``
        ```
        """
        for func_name in filter(
            lambda func_name: OBSERVE_FUNC_SUFFIX_PATTERN.search(func_name) is not None,
            dir(self)
        ):
            func: ObserveFuncType = getattr(self, func_name)
            # inspect checking
            if __func(func):
                observe_attr_name = OBSERVE_FUNC_SUFFIX_PATTERN.sub('', func_name)
                observe_dict[observe_attr_name] = func
        return observe_dict
    
    def detach_all__(self) -> None:
        # NOTE: create a new list of ``__observable_dict.values()`` to avoid value change during iteration.
        for observable_info in list(self.__observable_dict.values()):
            observable_info.observable.detach__(self)
    
    def __del__(self) -> None:
        self.detach_all__()
    
    def get_observable_dict__(self) -> _ObservableDict:
        return self.__observable_dict


def get_observe_func_name(name: str) -> str:
    return f'{name}{OBSERVE_FUNC_SUFFIX}'


class _AttrObserverDict(BaseDict[str, List[AttrObserver]]):
    
    def add__(self, __name: str, __observer: AttrObserver) -> None:
        if __name not in self:
            self[__name] = []
        
        observers = self[__name]
        if __observer not in observers:
            observers.append(__observer)
    
    def remove__(self, __name: str, __observer: AttrObserver) -> None:
        if __name in self:
            observers = self[__name]
            if __observer in observers:
                observers.remove(__observer)
            if len(observers) < 1:
                del self[__name]


class AttrObservable:
    
    def __init__(self) -> None:
        # attr name to observers
        self.__attr_observer_dict: _AttrObserverDict = _AttrObserverDict()
    
    def attach__(
        self,
        __observer: AttrObserver,
        *,
        init: Union[bool, Missing] = MISSING,
        namespaces: Union[Sequence[str], Missing, NoneOrNothing] = MISSING
    ) -> None:
        observe_dict = __observer.attach_inspect__(namespaces)
        
        names = set(observe_dict.keys())
        # inspect new observe attrs
        names = names - __observer.get_observable_dict__().get__(self)
        
        for name in names:
            attr_init = getattr(
                unwrap_method(observe_dict[name]),
                OBSERVE_INIT,
                True
            ) if init is MISSING else init
            self.attach_attr__(__observer, name, init=bool(attr_init))
    
    def attach_attr__(self, __observer: AttrObserver, __name: str, *, init: bool = True):
        self.__attr_observer_dict.add__(__name, __observer)
        __observer.get_observable_dict__().add__(self, __name)
        
        if init:
            value = getattr(self, __name, MISSING)
            self.notify__(__observer, __name, value, MISSING)
    
    def detach__(
        self,
        __observer: AttrObserver,
        *,
        namespaces: Union[Sequence[str], Missing, NoneOrNothing] = MISSING
    ) -> None:
        observable_dict = __observer.get_observable_dict__()
        if not observable_dict.contains__(self):
            return
        # Check names to detach.
        detach_names = observable_dict.get__(self)
        detach_names = set(__observer.detach_inspect__(namespaces).keys()) & detach_names
        
        # NOTE: use a copy of ``detach_names`` to avoid value change during iteration
        for name in list(detach_names):
            self.detach_attr__(__observer, name)
    
    def detach_attr__(self, __observer: AttrObserver, __name: str) -> None:
        self.__attr_observer_dict.remove__(__name, __observer)
        __observer.get_observable_dict__().remove__(self, __name)
    
    def notify__(self, __observer: AttrObserver, __name: str, __new_value: Any, __old_value: Any) -> None:
        """
        Notify one observer with new value, old value and observable object.
        """
        func: ObserveFuncType = getattr(__observer, get_observe_func_name(__name))
        return func(__new_value, __old_value, self)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in self.__attr_observer_dict:
            return super().__setattr__(__name, __value)
        else:
            old_value = getattr(self, __name, MISSING)
            super().__setattr__(__name, __value)
            # observer is called only when the new value is different from the old value
            if __value is not old_value:
                for observer in self.__attr_observer_dict[__name]:
                    self.notify__(observer, __name, __value, old_value)
    
    def get_attr_observer_dict__(self) -> _AttrObserverDict:
        return self.__attr_observer_dict


@overload
def AttrObserve(
    _func: NoneOrNothing = NOTHING,
    *,
    init: bool = True,
    namespace: Union[str, Missing] = MISSING
) -> Callable[[ObserveFuncType], ObserveFuncType]: pass
@overload
def AttrObserve(
    _func: ObserveFuncType,
    *,
    init: bool = True,
    namespace: Union[str, Missing] = MISSING
) -> ObserveFuncType: pass

@DecoratorCall(index=0, keyword='_func')
def AttrObserve(
    _func=NOTHING,
    *,
    init: bool = True,
    namespace: Union[str, Missing] = MISSING
):
    def set__(item: ObserveFuncType, name: str, value: Any):
        try:
            setattr(item, name, value)
        except Exception:
            from torchslime.logging.logger import logger
            logger.warning(f'Set ``{name}`` attribute failed. Observe object: {str(item)}. Please make sure it supports attribute set.')
    
    def decorator(func: ObserveFuncType) -> ObserveFuncType:
        set__(func, OBSERVE_INIT, init)
        set__(func, OBSERVE_NAMESPACE, namespace)
        return func
    return decorator

#
# Scoped Attr Utils
#

class ScopedAttrRestore(ContextDecorator, Generic[_T]):

    def __init__(
        self,
        obj: _T,
        attrs: Iterable[str]
    ) -> None:
        self.obj = obj
        self.attrs = list(attrs)
        self.prev_value_dict: Dict[str, Any] = {}

    def __enter__(self) -> "ScopedAttrRestore":
        for attr in self.attrs:
            # Only cache existing attributes of ``obj``.
            if hasattr(self.obj, attr):
                self.prev_value_dict[attr] = getattr(self.obj, attr, NOTHING)
        return self

    def __exit__(self, *args, **kwargs):
        for attr in self.attrs:
            try:
                if attr in self.prev_value_dict:
                    # Restore previously existing attributes before the scope.
                    setattr(self.obj, attr, self.prev_value_dict[attr])
                elif hasattr(self.obj, attr):
                    # Remove previously non-existing attributes before the scope.
                    delattr(self.obj, attr)
            except Exception as e:
                from torchslime.logging.logger import logger
                logger.error(
                    f'Restoring scoped attribute failed. Object: {str(self.obj)}, '
                    f'attribute: {attr}. {str(e.__class__.__name__)}: {str(e)}'
                )


class ScopedAttrAssign(ScopedAttrRestore[_T]):

    def __init__(
        self,
        obj: _T,
        attr_assign: Dict[str, Any]
    ) -> None:
        super().__init__(obj, attr_assign.keys())
        self.attr_assign = attr_assign

    def __enter__(self) -> "ScopedAttrAssign":
        # backup previous values
        ret = super().__enter__()
        for attr, value in self.attr_assign.items():
            try:
                setattr(self.obj, attr, value)
            except Exception as e:
                from torchslime.logging.logger import logger
                logger.error(
                    f'Assigning scoped attribute failed. Object: {str(self.obj)}, '
                    f'attribute: {attr}. {str(e.__class__.__name__)}: {str(e)}'
                )
        return ret

#
# Singleton base class
#

class Singleton(metaclass=SingletonMetaclass):
    """
    Helper class that creates a Singleton class using inheritance.
    
    Note that it works for each class (even subclasses) independently.
    
    Example:
    
    ```Python
    from torchslime.utils.bases import Singleton
    class A(Singleton): pass
    
    # B inherits A
    class B(A): pass
    
    print(A() is A())  # True
    print(B() is B())  # True
    print(A() is B())  # False
    
    \"\"\"
    These two values are different, because ``SingletonMetaclass`` sets ``__instance`` 
    separately for each class it creates.
    \"\"\"
    print(A._SingletonMetaclass__instance)
    print(B._SingletonMetaclass__instance)
    ```
    """
    __slots__ = ()

#
# Readonly attributes.
#

class ReadonlyAttr(metaclass=_ReadonlyAttrMetaclass):
    
    readonly_attr__: Tuple[str, ...] = ()
    readonly_attr_computed__: Tuple[str, ...] = ()
    readonly_attr_nothing_allowed__: bool = True
    readonly_attr_empty_allowed__: bool = True
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        pass

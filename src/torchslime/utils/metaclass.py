"""
Metaclasses used in torchslime.
We name all the metaclasses with ``Metaclass`` rather than the abbreviation 
``Meta``, because there already exists the ``Meta`` feature (although it has 
been deprecated), and we want to distinguish between these two concepts.
"""
from .typing import (
    TypeVar,
    Type,
    Tuple,
    Dict,
    Any,
    _SingletonMetaclass,
    TYPE_CHECKING
)
if TYPE_CHECKING:
    from .base import ReadonlyAttr


def Metaclasses(*args: Type[type], **kwargs) -> Type[type]:
    """
    Returns a newly created ``MergedMetaclass`` that inherits all the specified metaclasses 
    as well as ``MetaclassAdapter``. It makes the adaptation of metaclasses convenient , 
    which does not need the user manually define a new metaclass.
    
    NOTE: This function only applies when each of the metaclasses to be adapted is a ``class`` 
    rather than a ``function``. Note that ``class Example(metaclass=func)`` is also accepted 
    by the Python interpreter, but it doesn't apply here. If you want to make the ``func`` 
    compatible with it, you can define a new metaclass and call the ``func`` in the ``__new__`` 
    method.
    """
    class MergedMetaclass(*args, **kwargs):
        # ``metaclass_adapter__`` works as an indicator attribute that denotes the 
        # metaclass simply inherits multiple metaclasses and does nothing else.
        metaclass_adapter__ = True
    return MergedMetaclass


class InstanceCreationHookMetaclass(type):
    """
    This metaclass breaks the inheritance chain of ``__call__`` method, so 
    it should better be the highest possible level base class.
    """
    
    def __call__(cls, *args, **kwargs):
        instance = cls.new_hook_metaclass__(*args, **kwargs)
        if isinstance(instance, cls):
            cls.init_hook_metaclass__(instance, args=args, kwargs=kwargs)
        return instance
    
    def new_hook_metaclass__(cls, *args, **kwargs):
        return cls.__new__(cls, *args, **kwargs)
    
    def init_hook_metaclass__(cls, instance, args: Tuple, kwargs: Dict[str, Any]) -> None:
        # init
        cls.__init__(instance, *args, **kwargs)


class InitOnceMetaclass(InstanceCreationHookMetaclass):
    
    def init_hook_metaclass__(cls, instance, args: Tuple, kwargs: Dict[str, Any]) -> None:
        instance.init_once__ = {}
        super().init_hook_metaclass__(instance, args, kwargs)
        if hasattr(instance, 'init_once__'):
            del instance.init_once__


class SingletonMetaclass(_SingletonMetaclass):
    """
    Makes a specific class a singleton class. Inherits ``_SingletonMetaclass`` in 
    ``torchslime.utils.typing`` for more general use.
    """
    pass


_ReadonlyAttrT = TypeVar("_ReadonlyAttrT", bound="ReadonlyAttr")


class _ReadonlyAttrMetaclass(type):
    """
    Metaclass that checks readonly attributes. It should NOT be used independently. 
    Directly inherit ``torchslime.utils.base.ReadonlyAttr`` instead.
    """

    def __new__(
        meta_cls,
        __name: str,
        __bases: Tuple[Type, ...],
        __namespace: Dict[str, Any],
        **kwargs: Any
    ):
        # Check ``readonly_attr__`` defined in the class. If undefined, set it to ``()``.
        readonly_attr__: Tuple[str, ...] = __namespace.setdefault('readonly_attr__', ())
        # Create new class.
        cls: Type[_ReadonlyAttrT] = super().__new__(meta_cls, __name, __bases, __namespace, **kwargs)
        
        readonly_attr_computed_set = set(readonly_attr__)
        for base in __bases:
            readonly_attr_computed_set.update(getattr(base, 'readonly_attr_computed__', ()))
        
        cls.readonly_attr_computed__ = frozenset(readonly_attr_computed_set)
        return cls

"""
Metaclasses used in torchslime.
We name all the metaclasses with ``Metaclass`` rather than the abbreviation 
``Meta``, because there already exists the ``Meta`` feature (although it has 
been deprecated), and we want to distinguish between these two concepts.
"""
from .typing import (
    Type,
    Tuple,
    Dict,
    Any,
    _SingletonMetaclass
)


class MetaclassAdapter(type):
    """
    This is a base class that indicates a metaclass is a ``MergedMetaclass``.
    ``MergedMetaclass`` simply inherits multiple metaclasses and does nothing else, so it 
    shouldn't be taken into account in the metaclass check (see 
    ``torchslime.utils.decorators.MetaclassCheck`` for more information).
    """
    pass


def Metaclasses(*args: Type[type], **kwargs) -> Type[type]:
    """
    Returns a newly created ``MergedMetaclass`` that inherits all the specified metaclasses 
    as well as ``MetaclassAdapter``. It makes the adaptation of metaclasses convenient , 
    which does not need the user manually define a new metaclass.
    """
    class MergedMetaclass(MetaclassAdapter, *args, **kwargs): pass
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


class _ReadonlyAttrMetaclass(type):
    """
    
    """

    def __new__(
        meta_cls,
        __name: str,
        __bases: Tuple[type, ...],
        __namespace: Dict[str, Any],
        **kwargs: Any
    ):
        cls = super().__new__(meta_cls, __name, __bases, __namespace, **kwargs)
        pass

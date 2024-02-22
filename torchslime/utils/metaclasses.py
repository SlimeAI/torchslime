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
    Any
)


def Metaclasses(*args: Type[type], **kwargs) -> Type[type]:
    class MergedMetaclass(*args, **kwargs): pass
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

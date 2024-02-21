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
    Generator,
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
        instance = cls.__new__(cls, *args, **kwargs)
        if isinstance(instance, cls):
            from .bases import BaseGenerator
            init_hook_gen = BaseGenerator(cls.init_hook_yield__(instance, args, kwargs))
            # before init
            init_hook_gen()
            # init
            cls.__init__(instance, *args, **kwargs)
            # after init
            init_hook_gen()
        return instance
    
    def init_hook_yield__(
        cls,
        instance,
        args: Tuple,
        kwargs: Dict[str, Any]
    ) -> Generator:
        yield


class InitOnceMetaclass(InstanceCreationHookMetaclass):
    
    def init_hook_yield__(cls, instance, args: Tuple, kwargs: Dict[str, Any]) -> Generator:
        from .bases import BaseGenerator
        gen = BaseGenerator(super().init_hook_yield__(instance, args, kwargs))
        gen()
        # set ``init_once__`` cache
        instance.init_once__ = {}
        # ``cls.__init__``
        yield
        gen()
        # remove ``init_once__`` cache
        if hasattr(instance, 'init_once__'):
            del instance.init_once__

"""
Archived: Meta
"""
from torchslime.utils.common import dict_to_key_value_str_list
from torchslime.utils.typing import (
    NOTHING,
    NoneOrNothing,
    Any,
    Union,
    TypeVar,
    Callable,
    Type,
    is_none_or_nothing,
    overload,
    NoReturn
)
from torchslime.utils.decorator import DecoratorCall, RemoveOverload
from .decorator import ClassWraps, ClassFuncWrapper
from torchslime.utils.exception import APIMisused

_T = TypeVar("_T")


class _MetaWrapper:
    """
    ``_MetaWrapper`` class that creates new objects.
    See ``Meta`` for more information.
    """
    
    def __init__(self, cls, *args: Any, **kwargs: Any) -> None:
        self.cls__ = cls
        self.args = args
        self.kwargs = kwargs
        
        # set meta info
        from torchslime.utils.common import concat_format
        args_str = concat_format('', [str(arg) for arg in args], '', item_sep=', ', break_line=False)
        kwargs_str = concat_format('', dict_to_key_value_str_list(kwargs), '', item_sep=', ', break_line=False)
        meta_str = concat_format('', [item for item in [args_str, kwargs_str] if len(item) > 0], '', item_sep=', ', break_line=False)
        
        self.__module__ = cls.__module__
        self.__name__ = f'{cls.__name__}[{meta_str}]'
        self.__qualname__ = f'{cls.__qualname__}[{meta_str}]'
    
    def __call__(self, *args: Any, **kwargs: Any):
        cls = self.cls__
        # create a new object using ``m_new__``
        obj = cls.m_new__(*args, **kwargs)
        # call ``m_init__`` with args
        obj.m_init__(*self.args, **self.kwargs)
        # ``__init__`` method call
        obj.__init__(*args, **kwargs)
        return obj
    
    def __str__(self) -> str: return self.__name__
    def __repr__(self) -> str: return str(self)

# type hint
@overload
def _Meta(_cls: NoneOrNothing = NOTHING, *, directly_new_allowed: bool = True) -> Callable[[Type[_T]], Type[_T]]: pass
@overload
def _Meta(_cls: Type[_T], *, directly_new_allowed: bool = True) -> Type[_T]: pass

@DecoratorCall(index=0, keyword='_cls')
def _Meta(
    _cls: Type[_T] = NOTHING,
    *,
    directly_new_allowed: bool = True
):
    """
    ``_Meta`` decorator that enables classes with ``Meta`` features.
    See ``Meta`` for more information.
    """
    
    def decorator(cls__: Type[_T]) -> Type[_T]:
        if not hasattr(cls__, 'm_init__'):
            raise TypeError(f'Class ``{cls__.__name__}`` with ``Meta`` should have a ``m_init__`` method, but not found.')
        
        class_wraps = ClassWraps(cls__)
        
        # ``m__``
        m_wraps = class_wraps.m__
        @m_wraps(use_wraps=False)
        @classmethod
        def m__(cls: Type[_T], *args, **kwargs) -> Type[_T]:
            return _MetaWrapper(cls, *args, **kwargs)
        
        # ``__init_subclass__``
        init_subclass_wraps: ClassFuncWrapper = class_wraps.__init_subclass__
        @init_subclass_wraps
        @classmethod
        def init_subclass(
            cls,
            *args,
            directly_new_allowed: Union[bool, NoneOrNothing] = NOTHING,
            **kwargs
        ):
            super(cls__, cls).__init_subclass__(*args, **kwargs)
            # change ``__new__`` method if ``directly_new_allowed`` is set
            if not is_none_or_nothing(directly_new_allowed):
                cls.__new__ = new if directly_new_allowed else new_disallowed
        
        # ``__new__`` wraps
        class_new_wraps: ClassFuncWrapper = class_wraps.__new__
        new_cls_func = class_new_wraps.cls_func__
        
        def new(cls: Type[_T], *args, **kwargs) -> _T:
            # call ``m_new__`` to create a new object
            obj = cls.m_new__(*args, **kwargs)
            # call ``m_init__`` with no args
            obj.m_init__()
            return obj
        
        def new_disallowed(cls: Type, *args, **kwargs) -> NoReturn:
            raise APIMisused(
                f'Class ``{cls.__name__}`` with ``Meta`` is disallowed to directly new an instance, '
                f'please use ``{cls.__name__}.m__([args...])([args...])`` instead.'
            )
        
        # set ``__new__`` method according to whether directly calling ``__new__`` is allowed
        class_new_wraps(new if directly_new_allowed else new_disallowed)
        
        # ``m_new__`` wraps
        class_m_new_wraps = class_wraps.m_new__
        @class_m_new_wraps(use_wraps=False)
        @classmethod
        def m_new__(cls: Type[_T], *args, **kwargs) -> _T:
            # FIX: object.__new__ only accept one cls argument
            if new_cls_func is object.__new__:
                obj = new_cls_func(cls)
            else:
                obj = new_cls_func(cls, *args, **kwargs)
            return obj
        
        return cls__
    return decorator

@_Meta
@RemoveOverload(checklist=['m__'])
class Meta:
    """
    NOTE: The ``Meta`` feature is already deprecated because of the complexity and high 
    learning cost.
    
    NOTE: Can be better implemented using Python metaclass.
    
    ---
    
    The ``Meta`` class can separate the object ``__init__`` operation into 2 processes:
    ``m_init__`` and the original ``__init__``. Commonly, The ``__init__`` method is 
    always called after the object is created (except the situation that ``__new__`` 
    does not return the instantiation of the class), with given arguments. If the 
    argument list of ``__init__`` is too long, and when we try to override ``__init__`` 
    in a subclass, we have to either copy the endless argument declarations or use 
    ``*args`` and ``**kwargs`` (which means type hint is invalid before Python 3.10).
    
    To solve this problem, we try to introduce `Meta`` class. It has 2 features:
    
    1. The ``m_init__`` method is always called after ``__new__`` and before ``__init__``, 
    which cannot be achieved in plain Python methods.
    2. The arguments in ``m_init__`` are independent from ``__init__``.
    
    These features mean that if an argument may not be used in any of the subclasses, we 
    can put it in ``m_init__``, and we also keep ``m_init__`` always called after the 
    object instantiation, unlike other plain Python methods (which should be called 
    explicitly by users). Then when overriding ``__init__`` method, we avoid dealing with 
    many of the unused arguments (and those arguments have been moved to ``m_init__``).
    
    Usage:
    
    ```Python
    # Note that ``directly_new_allowed`` is default to ``True`` and it can be omitted.
    class Foo(Meta, directly_new_allowed=True):
        def __init__(self) -> None:
            print('``__init__`` is called with no args.')
    
        def m_init__(self, arg1='default') -> None:
            print(f'``m_init__`` is called with arg: {arg1}.')

    class FooChild(Foo):
        def __init__(self) -> None:
            super().__init__()
            print('yeah, I don\'t have to copy the verbose argument list!')
    
    # First call ``m__`` and then normally create the object
    # Args passed to ``m__()`` will be passed to ``m_init__``
    # Args passed to the following ``()`` will be passed to ``__new__`` (and 
    # then ``__init__``)
    foo_obj = FooChild.m__('I\'m arg1')()
    
    \"""
    Output:
    ``m_init__`` is called with arg: I'm arg1.
    ``__init__`` is called with no args.
    yeah, I don't have to copy the verbose argument list!
    \"""
    
    # The following 2 methods are equivalent (if every argument is given a default value 
    # and ``directly_new_allowed`` is ``True``)
    foo_obj = FooChild()  # Simply instantiating ``FooChild`` object without calling ``m__``.
    foo_obj = FooChild.m__()()
    
    \"""
    Output:
    ``m_init__`` is called with arg: default.
    ``__init__`` is called with no args.
    yeah, I don't have to copy the verbose argument list!
    \"""
    ```
    
    You may find that the ``m__`` method lacks the corresponding type hint of 
    ``m_init__``, so you can override ``m__`` in ``Foo``, and one of the recommended 
    usage is:
    
    ```Python
    # Note that ``directly_new_allowed`` is default to ``True`` and it can be omitted.
    # Remove the overloaded ``m__`` method, so the real ``m__`` method is called.
    @RemoveOverload(checklist=['m__'])
    class Foo(Meta, directly_new_allowed=True):
        ...
        
        # Specify the args used in ``m_init__``.
        # Note that it only works as a type hint.
        @overload
        @classmethod
        def m__(cls: Type[_T], arg1) -> Type[_T]: pass
    ```
    
    Implementation details:
    
    Firstly, the original ``__new__`` method in class ``cls`` is saved to ``m_new__``.
    
    Secondly, the ``__new__`` method is replaced by another method (namely ``new``). 
    The ``new`` method will first call ``m_new__`` (i.e., the original ``__new__`` 
    method), and then call ``m_init__`` with no arguments (corresponds to 
    ``foo_obj = FooChild()`` in the above example). The ``__init__`` method is 
    automatically called by the Python interpreter in this situation.
    
    Thirdly, the ``m__`` method is assigned to ``cls``, and it returns ``_MetaWrapper``. 
    The ``_MetaWrapper`` will:
    
    1. call ``m_new__``
    2. call ``m_init__``
    
    (The above 2 processes are the same as ``new``)
    
    3. call ``__init__`` (Because ``__init__`` will not be automatically called in this 
    case)
    """
    def m_init__(self, *args, **kwargs): pass
    
    @overload
    @classmethod
    def m__(cls: Type[_T], *args, **kwargs) -> Type[_T]: return cls

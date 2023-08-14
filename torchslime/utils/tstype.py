"""
This python file defines common types that are used in the project.
The module is renamed from 'type' to 'tstype' to avoid namespace conflict with built-in class ``type``
"""
from typing import Sequence, Union, Callable
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
from torchslime.utils.bases import Nothing
INT_SEQ_N = Union[INT_SEQ, None, Nothing]

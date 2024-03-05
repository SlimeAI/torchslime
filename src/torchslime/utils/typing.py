"""
This Python module defines common types that are used in the project, 
provides version compatibility for Python and introduces special constants 
in torchslime.
"""
from slime_core.utils.typing import *

#
# Torch version adapter
#

try:
    from torch.optim.lr_scheduler import LRScheduler as TorchLRScheduler
except Exception:
    from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler

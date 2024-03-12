#
# version info
#
__version__ = '0.2.0'

__all__ = (
    '__version__',
)


# NOTE: The global ``store`` module should be imported first to ensure 
# a correct module initialization order.
import torchslime.utils.store as _

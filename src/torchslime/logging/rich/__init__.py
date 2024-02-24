"""
Rich Text Processor
"""

try:
    from .native import *
except Exception:
    # if rich native is not loaded correctly, then use alter instead
    
    # TODO: the alter api is not implemented in the current version
    raise NotImplementedError(
        'The rich alter api has not been implemented in this version, '
        'so the ``rich`` module is a necessary dependency, and you should install it first.'
    )

    from .alter import *

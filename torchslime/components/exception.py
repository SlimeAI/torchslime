#
# API Misused
#

class APIMisused(Exception):

    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg
    
    def __str__(self) -> str:
        return f'APIMisused: {self.msg}'


from torchslime.utils.bases import NOTHING

#
# Handler Interrupt
#

class HandlerInterrupt(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class HandlerBreak(HandlerInterrupt):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class HandlerContinue(HandlerInterrupt):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class HandlerTerminate(HandlerInterrupt):

    def __init__(self, msg: str, raise_handler=NOTHING) -> None:
        super().__init__()
        self.msg = msg
        self.raise_handler = raise_handler
    
    def __str__(self) -> str:
        return f'HandlerTerminate -> raise_handler: {str(self.raise_handler)}, msg: {self.msg}'

#
# Handler Exception
#

class HandlerException(Exception):

    def __init__(self, exception_handler, exception: Exception) -> None:
        super().__init__()
        self.exception_handler = exception_handler
        self.exception = exception
    
    def __str__(self) -> str:
        return f'HandlerException -> exception_handler: {str(self.exception_handler)}'


class WrapperException(Exception):
    pass

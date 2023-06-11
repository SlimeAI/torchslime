from torchslime.utils.bases import NOTHING


"""
Handler Interrupt
"""
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
        return 'HandlerTerminate -> raise_handler: {raise_handler}, msg: {msg}'.format(
            raise_handler=str(self.raise_handler),
            msg=str(self.msg)
        )


"""
Handler Exception
"""
class HandlerException(Exception):

    def __init__(self, exception_handler, exception: Exception) -> None:
        super().__init__()
        self.exception_handler = exception_handler
        self.exception = exception
    
    def __str__(self) -> str:
        return 'HandlerException -> exception_handler: {exception_handler}'.format(
            exception_handler=str(self.exception_handler)
        )


"""
API Misused
"""
class APIMisused(Exception):

    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg
    
    def __str__(self) -> str:
        return 'APIMisused: {msg}'.format(
            msg=self.msg
        )

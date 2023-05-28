

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

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


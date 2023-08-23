class WorkGraphBuildingException(Exception):
    """
    Raised when work graph can't be built
    """
    def __init__(self, message: str):
        super().__init__(message)


class InputDataException(Exception):
    """
    Raise when there isn't info about tasks' connection
    """
    def __init__(self, reason: str):
        super().__init__("The reason of Input Data exception is " + reason)

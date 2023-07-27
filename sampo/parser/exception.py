class WorkGraphBuildingException(Exception):
    """
    Raised when work graph can't be built
    """
    def __init__(self, message: str):
        super().__init__(message)

class SchedulingPipelineError(Exception):
    """
    Raised when scheduling pipeline error occurred.

    This is a kind of 'IllegalStateException', e.g. raising this
    indicates that the corresponding scheduling pipeline come to incorrect internal state.
    """
    def __init__(self, message: str):
        super().__init__(message)

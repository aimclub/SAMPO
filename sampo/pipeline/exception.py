class PipelineError(Exception):
    """
    Raised when any pipeline error occurred.

    This is a kind of 'IllegalStateException', e.g. raising this
    indicates that the corresponding pipeline come to incorrect internal state.
    """
    def __init__(self, message: str):
        super().__init__(message)

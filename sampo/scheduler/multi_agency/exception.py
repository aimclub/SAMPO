class NoSufficientAgents(Exception):
    """
    Raise when manager does not have enough agents
    """

    def __init__(self, message):
        super().__init__(message)

class NoSufficientContractorError(Exception):
    """
    Raise when contractor error occurred.

    It indicates when the contractors have not sufficient resources to perform schedule.
    """
    def __init__(self, message):
        super().__init__(message)


class NotEnoughMaterialsInDepots(Exception):

    def __init__(self, message):
        super().__init__(message)


class NoAvailableResources(Exception):

    def __init__(self, message):
        super().__init__(message)

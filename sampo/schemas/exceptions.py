class NoSufficientContractorError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotEnoughMaterialsInDepots(Exception):

    def __init__(self, message):
        super().__init__(message)


class NoAvailableResources(Exception):

    def __init__(self, message):
        super().__init__(message)

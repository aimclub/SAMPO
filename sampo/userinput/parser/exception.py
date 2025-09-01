"""Custom exceptions for user input processing.

Пользовательские исключения для обработки пользовательского ввода."""


class WorkGraphBuildingException(Exception):
    """Raised when work graph can't be built.

    Возникает, когда невозможно построить граф работ.
    """

    def __init__(self, message: str):
        """Store error description.

        Сохраняет описание ошибки.
        """

        super().__init__(message)


class InputDataException(Exception):
    """Raised when information about task links is missing.

    Возникает при отсутствии информации о связях задач.
    """

    def __init__(self, reason: str):
        """Store textual reason of the failure.

        Сохраняет текстовую причину ошибки.
        """

        super().__init__("The reason of Input Data exception is " + reason)

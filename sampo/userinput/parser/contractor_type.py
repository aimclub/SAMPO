from enum import Enum


class ContractorType(Enum):
    Minimal = 50
    Average = 75
    Maximal = 200

    # TODO: describe the function (annotating and comments)
    def command_capacity(self):
        return self.value

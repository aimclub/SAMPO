from dataclasses import dataclass
from typing import Optional


@dataclass
class Identifiable:
    """
    A base class for all unique entities
    :param id: unique id for the object
    :param name: name of for the object
    """
    id: str
    name: Optional[str]

    def __hash__(self):
        return hash(self.id)

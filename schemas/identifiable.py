from dataclasses import dataclass
from typing import Optional


@dataclass
class Identifiable:
    id: str
    name: Optional[str]

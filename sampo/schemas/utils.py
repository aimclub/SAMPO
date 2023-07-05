import uuid
from random import Random
from typing import Optional
from uuid import uuid4


def uuid_str(rand: Optional[Random] = None) -> str:
    """
    Transform str to uuid format.
    """
    ans = uuid4() if rand is None else uuid.UUID(int=rand.getrandbits(128))
    return str(ans)

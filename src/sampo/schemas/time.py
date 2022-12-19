from sampo.schemas.serializable import AutoJSONSerializable

TIME_INF = 2_000_000_000


# TODO Consider converting to StrSeriailzable
# TODO: describe the class (description, parameters/)
class Time(AutoJSONSerializable['Time']):
    value: int
    
    # TODO: describe the function (description, parameters, return type)
    def __init__(self, value: int = 0):
        self.value = 0
        self.set_time(value)

    # TODO: describe the function (description, return type)
    @staticmethod
    def inf():
        return Time(TIME_INF)

    # Copy-paste body between left- and right-associative functions
    # is here to avoid functional call overhead
    # TODO: describe the function (description, parameters, return type)
    def __add__(self, other: 'Time' or int) -> 'Time':
        return Time(self.value + (other.value if isinstance(other, Time) else other))

    # TODO: describe the function (description, parameters, return type)
    def __radd__(self, other: 'Time' or int) -> 'Time':
        return Time(self.value + (other.value if isinstance(other, Time) else other))

    # TODO: describe the function (description, parameters, return type)
    def __sub__(self, other: 'Time' or int) -> 'Time':
        return Time(self.value - (other.value if isinstance(other, Time) else other))

    # TODO: describe the function (description, parameters, return type)
    def __rsub__(self, other: 'Time' or int) -> 'Time':
        return Time((other.value if isinstance(other, Time) else other) - self.value)

    # TODO: describe the function (description, parameters, return type)
    def __mul__(self, other: 'Time' or int) -> 'Time':
        return Time(self.value * (other.value if isinstance(other, Time) else other))

    # TODO: describe the function (description, parameters, return type)
    def __rmul__(self, other: 'Time' or int) -> 'Time':
        return Time(self.value * (other.value if isinstance(other, Time) else other))

    # TODO: describe the function (description, parameters, return type)
    def __floordiv__(self, other: 'Time' or int) -> 'Time':
        return Time(self.value // (other.value if isinstance(other, Time) else other))

    # TODO: describe the function (description, parameters, return type)
    def __truediv__(self, other: 'Time' or int) -> float:
        return self.value / (other.value if isinstance(other, Time) else other)

    # TODO: describe the function (description, parameters, return type)
    def __lt__(self, other):
        return self.value < other

    # TODO: describe the function (description, parameters, return type)
    def __le__(self, other):
        return self.value <= other

    # TODO: describe the function (description, parameters, return type)
    def __gt__(self, other):
        return self.value > other

    # TODO: describe the function (description, parameters, return type)
    def __ge__(self, other):
        return self.value >= other

    # TODO: describe the function (description, parameters, return type)
    def __eq__(self, other):
        return self.value == other

    # TODO: describe the function (description, return type)
    def __bool__(self):
        return self.value != 0

    # TODO: describe the function (description, return type)
    def __int__(self) -> int:
        return self.value

    # TODO: describe the function (description, return type)
    def __str__(self) -> str:
        return str(self.value)

    # TODO: describe the function (description, return type)
    def __repr__(self) -> str:
        return str(self.value)

    # TODO: describe the function (description, return type)
    def __unicode__(self) -> str:
        return str(self.value)

    # TODO: describe the function (description, return type)
    def __hash__(self) -> int:
        return hash(self.value)

    # TODO: describe the function (description, return type)
    def set_time(self, value: int):
        value = int(value)
        if value > TIME_INF:
            value = TIME_INF
        elif value < -TIME_INF:
            value = -TIME_INF
        self.value = value

    # TODO: describe the function (description, return type)
    def is_inf(self) -> bool:
        return abs(self.value) == TIME_INF

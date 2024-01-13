import ast
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sampo.schemas.serializable import AutoJSONSerializable, JSONSerializable, T, JS, StrSerializable, SS
from sampo.utilities.serializers import custom_serializer, custom_type_serializer, custom_type_deserializer


class TestSimpleSerialization(AutoJSONSerializable['TestSimpleSerialization']):
    key: str
    value: Any

    def __repr__(self) -> str:
        return f'{{{self.key}: {self.value}}}'


@dataclass
class TestAutoJSONSerializable(AutoJSONSerializable['TestAutoJSONSerializable']):
    int_field: int
    float_field: float
    array_field: list
    dict_field: dict
    bool_field: bool
    none_field: None
    ndarray_field: np.ndarray
    df_field: pd.DataFrame
    class_field: TestSimpleSerialization
    class_field2: TestSimpleSerialization
    neighbor_info: 'TestAutoJSONSerializable'

    ignored_fields = ['array_field', 'bool_field']

    @custom_serializer(float)
    def floater(self, value) -> str:
        return str(value).replace('.', '|')

    @classmethod
    @custom_serializer(float, deserializer=True)
    def defloater(cls, value) -> float:
        return float(value.replace('|', '.'))

    @custom_serializer('class_field')
    def tsser(self, value) -> list:
        return [value.key, value.value]

    @classmethod
    @custom_serializer('class_field', deserializer=True)
    def detsser(cls, value) -> TestSimpleSerialization:
        r = TestSimpleSerialization()
        r.key = value[0]
        r.value = value[1]
        return r

    @custom_type_serializer('TestAutoJSONSerializable')
    def neighbourer(self, value) -> int:
        return value.int_field

    @classmethod
    @custom_type_deserializer('TestAutoJSONSerializable')
    def deneighbourer(cls, value) -> 'TestAutoJSONSerializable':
        return TestAutoJSONSerializable(value, None, None, None, None, None, None, None, None, None, None)


@dataclass
class TestJSONSerializable(JSONSerializable['TestJSONSerializable']):
    id: int
    name: str
    it1: int
    it2: str
    it3: bool

    def _serialize(self) -> T:
        return {
            'id': self.id,
            'name': self.name,
            'content': [self.it1, self.it2, self.it3]
        }

    @classmethod
    def _deserialize(cls, representation: T) -> JS:
        r = object.__new__(TestJSONSerializable)
        r.__dict__ = {
            'id': representation['id'],
            'name': representation['name'],
            'it1': representation['content'][0],
            'it2': representation['content'][1],
            'it3': representation['content'][2],
        }
        return r


@dataclass
class TestStrSerializable(StrSerializable['TestStrSerializable']):
    id: int
    name: str
    description: list

    def _serialize(self) -> str:
        return f'{self.name}\n{self.id}\n{str(self.description)}'

    @classmethod
    def _deserialize(cls, str_representation: str) -> SS:
        s = str_representation.split('\n')
        r = object.__new__(TestStrSerializable)
        r.name = s[0]
        r.id = int(s[1])
        r.description = ast.literal_eval(s[2])
        return r

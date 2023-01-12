from functools import partial
from io import StringIO
from typing import Union

import numpy as np
import pandas as pd

CUSTOM_FIELD_SERIALIZER = '_serializer_for_fields'
CUSTOM_FIELD_DESERIALIZER = '_deserializer_for_fields'
CUSTOM_TYPE_SERIALIZER = '_serializer_for_types'
CUSTOM_TYPE_DESERIALIZER = '_deserializer_for_types'


def custom_serializer(type_or_field: Union[type, str], deserializer: bool = False):
    """
    Meta-decorator for marking custom serializers or deserializers methods.<br/>
    This decorator can stack with other serializer/deserializer decorators.
    :param type_or_field: Name (str) of field or type (type) of fields, which will be serialized with this serializer in
    current class. If type should be presented in str representation, consider using custom_type_serializer or
    custom_type_deserializer decorators.
    :param deserializer:
    If True, the decorated function will be considered as a custom deserializer for type_or_field type or field<br/>
    If None, deserializer should be decorated separately with custom_serializer or custom_field_deserializer or
    custom_type_deserializer decorator
    :return:
    """
    type_of_entity = type(type_or_field)
    if type_of_entity is type:
        # if callable(deserializer):
        #     _decorate_serializer(func=deserializer,
        #                          collection_name=CUSTOM_TYPE_DESERIALIZER,
        #                          new_element=type_or_field)
        # elif deserializer:
        if deserializer:
            return custom_type_deserializer(type_or_field)
        return custom_type_serializer(type_or_field)

    elif type_of_entity is str:
        # if callable(deserializer):
        #     _decorate_serializer(func=deserializer,
        #                          collection_name=CUSTOM_FIELD_DESERIALIZER,
        #                          new_element=type_or_field)
        # elif deserializer:
        if deserializer:
            return custom_field_deserializer(type_or_field)
        return custom_field_serializer(type_or_field)

    raise TypeError(f'Unexpected type of param type_or_field: {type(type_or_field)} instead of Union[type, str]')


def custom_field_serializer(field_name: str):
    return partial(_decorate_serializer,
                   collection_name=CUSTOM_FIELD_SERIALIZER,
                   new_element=field_name)


def custom_field_deserializer(field_name: str):
    return partial(_decorate_serializer,
                   collection_name=CUSTOM_FIELD_DESERIALIZER,
                   new_element=field_name)


def custom_type_serializer(__type: type or str):
    return partial(_decorate_serializer,
                   collection_name=CUSTOM_TYPE_SERIALIZER,
                   new_element=__type)


def custom_type_deserializer(__type: type or str):
    return partial(_decorate_serializer,
                   collection_name=CUSTOM_TYPE_DESERIALIZER,
                   new_element=__type)


def _decorate_serializer(func, collection_name, new_element):
    if not hasattr(func, collection_name):
        setattr(func, collection_name, [])
    getattr(func, collection_name).append(new_element)
    return func


def default_ndarray_serializer(array: np.ndarray):
    return array.tolist()


def default_ndarray_deserializer(__list: list):
    return np.array(__list)


def default_dataframe_serializer(df: pd.DataFrame):
    return df.to_csv(encoding='utf-8')


def default_dataframe_deserializer(str_repr: str):
    return pd.read_csv(StringIO(str_repr), encoding='utf-8')


def default_np_int_serializer(n):
    return int(n)


def default_np_int_deserializer(n):
    return np.int32(n)


def default_np_long_serializer(n):
    return int(n)


def default_np_long_deserializer(n):
    return np.int64(n)

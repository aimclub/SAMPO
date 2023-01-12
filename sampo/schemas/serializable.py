import json
import os
import pydoc
from abc import ABC, abstractmethod
from itertools import chain
from typing import Generic, TypeVar, Dict, Union

import numpy as np
import pandas as pd

from sampo.utilities.serializers import CUSTOM_FIELD_SERIALIZER, CUSTOM_FIELD_DESERIALIZER, CUSTOM_TYPE_SERIALIZER, \
    CUSTOM_TYPE_DESERIALIZER, default_ndarray_serializer, default_dataframe_serializer, default_ndarray_deserializer, \
    default_dataframe_deserializer, default_np_int_deserializer, default_np_int_serializer, default_np_long_serializer, \
    default_np_long_deserializer

# TODO: describe constants
T = TypeVar('T', str, dict, list, tuple, str, bool, None)
S = TypeVar('S', bound='Serializable')
SS = TypeVar('SS', bound='StrSerializable')
JS = TypeVar('JS', bound='JSONSerializable')
AJS = TypeVar('AJS', bound='AutoJSONSerializable')

TYPE_HINTS = '_serializable_type_hints'


# TODO: Implement PartialSerializable, which can't completely deserialize itself, and just returns rich info to parent

# TODO: describe the class (description, parameters)
class Serializable(ABC, Generic[T, S]):
    @property
    @abstractmethod
    def serializer_extension(self) -> str:
        return NotImplemented

    @abstractmethod
    def _serialize(self) -> T:
        """
        Converts all the meaningful information from this instance to a generic representation
        :return: A generic representation
        """
        ...

    @classmethod
    @abstractmethod
    def _deserialize(cls, representation: T) -> S:
        """
        Creates class instance from a representation
        :param representation: Representation produced by _serialize method
        :return: New class instance
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, folder_path: str, file_name: str) -> S:
        """
        Factory method that produces a python object from the serialized version of it
        :param folder_path: Path to the folder, where the serialized file is saved
        :param file_name: File name without extension
        (the file extension should match with the one returned by serializer_extension method)
        :return: The constructed python object
        """
        ...

    # TODO: describe the function (return type)
    def dump(self, folder_path: str, file_name: str):
        """
        Serializes object and saves it to file
        :param folder_path: Path to the folder where the serialized file should be saved
        :param file_name: Name of the file without extension
        (the appended extension could be explored via serializer_extension method)
        """
        ...

    @classmethod
    def get_full_file_name(cls, folder_path: str, file_name: str):
        """
        Combines path to folder, file name and extension to get full file name
        :param folder_path: Path to folder
        :param file_name: File name without extension
        :return: Full file path, name and extension
        """
        return os.path.join(folder_path, f'{file_name}.{cls.serializer_extension}')


# TODO: describe the class (description, parameters)
class StrSerializable(Serializable[str, SS], ABC, Generic[SS]):
    serializer_extension: str = 'dat'

    @abstractmethod
    def _serialize(self) -> str:
        """
        Converts object to str representation
        :return: str representation of the object
        """
        ...

    @classmethod
    @abstractmethod
    def _deserialize(cls, str_representation: str) -> SS:
        """
        Creates class instance from a str representation
        :param str_representation: Representation produced by _serialize method
        :return: New class instance
        """
        ...

    @classmethod
    def load(cls, folder_path: str, file_name: str) -> SS:
        """
        Factory method that produces a python object from the serialized version of it
        :param folder_path: Path to the folder, where the serialized file is saved
        :param file_name: File name without extension
        (the file extension should match with the one returned by serializer_extension method)
        :return: The constructed python object
        """
        full_file_name = cls.get_full_file_name(folder_path, file_name)
        with open(full_file_name, 'r', encoding='utf-8') as read_file:
            str_representation = read_file.read()
        return cls._deserialize(str_representation)

    # TODO: describe the function (return type)
    def dump(self, folder_path: str, file_name: str) -> None:
        """
        Serializes object and saves it to file
        :param folder_path: Path to the folder where the serialized file should be saved
        :param file_name: Name of the file without extension
        (the appended extension could be explored via serializer_extension method)
        """
        full_file_name = self.get_full_file_name(folder_path, file_name)
        serialized_str = self._serialize()
        with open(full_file_name, 'w', encoding='utf-8') as write_file:
            write_file.write(serialized_str)


# TODO: describe the class (description, parameters)
class JSONSerializable(Serializable[Dict[str,
                                         Union[
                                             dict,
                                             Union[list, tuple],
                                             str,
                                             Union[int, float],
                                             bool,
                                             None]],
                                    JS], ABC, Generic[JS]):
    serializer_extension: str = 'json'

    @abstractmethod
    def _serialize(self) -> T:
        """
        Converts all the meaningful information from this instance to a generic representation
        :return: A generic representation
        """
        ...

    @classmethod
    @abstractmethod
    def _deserialize(cls, representation: T) -> JS:
        """
        Creates class instance from a representation
        :param representation: Representation produced by _serialize method
        :return: New class instance
        """
        ...

    @classmethod
    def load(cls, folder_path: str, file_name: str) -> JS:
        """
        Factory method that produces a python object from the serialized version of it
        :param folder_path: Path to the folder, where the serialized file is saved
        :param file_name: File name without extension
        (the file extension should match with the one returned by serializer_extension method)
        :return: The constructed python object
        """
        full_file_name = cls.get_full_file_name(folder_path, file_name)
        with open(full_file_name, 'r', encoding='utf-8') as read_file:
            dict_representation = json.load(read_file)
        return cls._deserialize(dict_representation)

    # TODO: describe the function (return type)
    def dump(self, folder_path: str, file_name: str) -> None:
        """
        Serializes object and saves it to file
        :param folder_path: Path to the folder where the serialized file should be saved
        :param file_name: Name of the file without extension
        (the appended extension could be explored via serializer_extension method)
        """
        full_file_name = self.get_full_file_name(folder_path, file_name)
        serialized_dict = self._serialize()
        with open(full_file_name, 'w', encoding='utf-8') as write_file:
            json.dump(serialized_dict, write_file)


# TODO: Implement automotive Enum serialization/deserialization
# TODO: Implement auto serialization of typed collection
# TODO: describe the class (parameters)
class AutoJSONSerializable(JSONSerializable[AJS], ABC):
    """
    Parent class for serialization of classes, which can be automatically converted to dict with Serializable properties
    and custom (de-)serializers, marked with custom_serializer and custom_deserializer decorators.
    """
    serializer_extension: str = 'json'

    _default_serializers_deserializers = {
        'serializers':
            {
                str(np.ndarray): default_ndarray_serializer,
                str(pd.DataFrame): default_dataframe_serializer,
                str(np.int32): default_np_int_serializer,
                str(np.int64): default_np_long_serializer,
            },
        'deserializer':
            {
                str(np.ndarray): default_ndarray_deserializer,
                str(pd.DataFrame): default_dataframe_deserializer,
                str(np.int32): default_np_int_deserializer,
                str(np.int64): default_np_long_deserializer,
            }
    }

    # TODO: describe the function (description, return type)
    @classmethod
    @property
    def _default_serializers(cls):
        return cls._default_serializers_deserializers['serializers']

    # TODO: describe the function (description, return type)
    @classmethod
    @property
    def _default_deserializers(cls):
        return cls._default_serializers_deserializers['deserializer']

    # TODO: describe the function (description, return type)
    @classmethod
    @property
    def ignored_fields(cls):
        return []

    def _serialize(self) -> Dict[str,
                                 Union[
                                     dict,
                                     Union[list, tuple],
                                     str,
                                     Union[int, float],
                                     bool,
                                     None]]:
        """
        Converts all the meaningful information from this instance to dict representation
        with values of eligible types only to be converted to JSON data structures:
            - list or tuple (to array)
            - str (to string)
            - int or float (to number)
            - True or False (to true or false)
            - None (to null)
            - dict with values of the listed types (to object)
        :return: dict representation of the object
        """
        simple_types = dict, list, tuple, str, int, float, bool, type(None)
        custom_field_serializers = dict(chain(*[[(field, attr) for field in getattr(attr, CUSTOM_FIELD_SERIALIZER)]
                                                for attr in [getattr(self, name) for name in dir(self)]
                                                if hasattr(attr, CUSTOM_FIELD_SERIALIZER)]))
        custom_type_serializers = dict(chain(*[[(__type, attr) for __type in getattr(attr, CUSTOM_TYPE_SERIALIZER)]
                                               for attr in [getattr(self, name) for name in dir(self)]
                                               if hasattr(attr, CUSTOM_TYPE_SERIALIZER)]))
        type_hints = {}

        # TODO: describe the function (description, parameters, return type)
        def serialize_field(name, value):
            if name in custom_field_serializers:
                return custom_field_serializers[name](value)

            t = type(value)
            str_t = str(t)
            name_t = t.__name__
            if t in custom_type_serializers:
                type_hints[name] = str_t
                return custom_type_serializers[t](value)
            if str_t in custom_type_serializers:
                type_hints[name] = str_t
                return custom_type_serializers[str_t](value)
            if name_t in custom_type_serializers:
                type_hints[name] = name_t
                return custom_type_serializers[name_t](value)
            if str_t in self._default_serializers:
                type_hints[name] = str_t
                return self._default_serializers[str_t](value)
            if issubclass(t, Serializable):
                type_hints[name] = pydoc.classname(t, '')
                return value._serialize()
            if t in simple_types:
                return value
            raise ValueError(f'Cannot serialize field {name}.'
                             f' Provide custom serializer for it (via decorator @custom_serializer)'
                             f' or make inherit {type(value)} from Serializable.')

        return dict({TYPE_HINTS: type_hints},
                    **{k: serialize_field(k, v) for k, v in self.__dict__.items() if k not in self.ignored_fields})

    @classmethod
    def _deserialize(cls, dict_representation: dict) -> AJS:
        """
        Creates class instance from a dict representation
        :param dict_representation: Representation produced by _serialize method
        :return: New class instance
        """
        custom_field_deserializers = dict(chain(*[[(field, attr) for field in getattr(attr, CUSTOM_FIELD_DESERIALIZER)]
                                                for attr in [getattr(cls, name) for name in dir(cls)]
                                                if hasattr(attr, CUSTOM_FIELD_DESERIALIZER)]))
        custom_type_deserializers = dict(chain(*[[(str(__type), attr) for __type in getattr(attr,
                                                                                          CUSTOM_TYPE_DESERIALIZER)]
                                               for attr in [getattr(cls, name) for name in dir(cls)]
                                               if hasattr(attr, CUSTOM_TYPE_DESERIALIZER)]))
        type_hints = {}
        if TYPE_HINTS in dict_representation:
            type_hints = dict_representation[TYPE_HINTS]
            del dict_representation[TYPE_HINTS]

        # TODO: describe the function (description, parameters, return type)
        def deserialize_field(name, value):
            if name in custom_field_deserializers:
                return custom_field_deserializers[name](value)
            if name in type_hints:
                __type = type_hints[name]
                if __type in custom_type_deserializers:
                    return custom_type_deserializers[__type](value)
                if __type in cls._default_deserializers:
                    return cls._default_deserializers[__type](value)
                c = pydoc.locate(type_hints[name])
                return c._deserialize(value)
            return value

        fields = dict({f: None for f in cls.ignored_fields},
                      **{k: deserialize_field(k, v)
                         for k, v in dict_representation.items()
                         if k not in cls.ignored_fields})

        self = cls.__new__(cls)
        try:
            self.__dict__ = fields
        except:
            for k, v in fields.items():
                object.__setattr__(self, k, v)
        return self

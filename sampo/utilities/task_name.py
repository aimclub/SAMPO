from abc import ABC
from typing import Dict

import pandas as pd


def get_task_name_unique_mapping(path: str) -> 'NameMapper':
    """
    Gets mapping of our task names to the unique names
    :param path: path to the csv file
    :return: Dict {our_name: unique_name}
    """
    df = read_tasks_df(path)
    return DictNameMapper({r[0]: r[1] for r in df.loc[:, ['task_name', 'unique_task_name']].to_numpy()})


def get_inverse_task_name_mapping(path: str) -> 'NameMapper':
    """
    Gets mapping of the unique names to our task names
    :param path: path to the csv file
    :return: Dict {unique_name: our_name}
    """
    df = read_tasks_df(path)
    return DictNameMapper({r[1]: r[0] for r in df.loc[:, ['task_name', 'unique_task_name']].to_numpy()})


def read_tasks_df(path: str) -> pd.DataFrame:
    """
    Reads DataFrame with tasks
    :param path: path to the csv file
    :return: The DataFrame read
    """
    return pd.read_csv(path, sep=';', header=0)


class NameMapper(ABC):
    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value):
        raise Exception('Trying to set a value to the NameMapper')


class DummyNameMapper(NameMapper):
    def __getitem__(self, item):
        return item

    @property
    def _source(self):
        return self


class DictNameMapper(NameMapper):
    def __init__(self, source: Dict[str, str]):
        self._source = source

    def __getitem__(self, item):
        return self._source[item] if item in self._source else item


class ModelNameMapper(NameMapper):
    """
    NameMapper for Kovalchuk's model integration
    """
    # TODO: implement
    def __init__(self):
        self.__nie()

    def __getitem__(self, item):
        self.__nie()

    def __nie(self):
        raise NotImplementedError('ModelNameMapper is not implemented')

from abc import ABC, abstractmethod
import pandas as pd


class DataSet(ABC):
    """
    Each DataSet to be built will have:
    1. setter
    2. getter
    """

    @abstractmethod
    def set_data(self):
        pass

    @abstractmethod
    def get_data(self):
        pass


class SingleDataSet(DataSet):
    """
    A SingleDataSet is basically a collection of Features.
    """


class MultiDataSet(DataSet):
    """
    A MultiDataSet can comprise of other DataSets.
    """

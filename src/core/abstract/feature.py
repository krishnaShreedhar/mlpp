from abc import ABC, abstractmethod


class Feature(ABC):
    """
    Each New Feature to be built will have:
    1. custom data requirement which will be collected using the collect_data method.
    2. custom compute function
    """

    @abstractmethod
    def collect_data(self):
        pass

    @abstractmethod
    def compute(self):
        pass

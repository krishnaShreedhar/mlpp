from abc import ABC, abstractmethod


class Visualizer(ABC):
    """
    Each New Visualizer to be built will have:
    1. custom data requirement which will be collected using the collect_data method.
    2. custom visualize function
    """

    @abstractmethod
    def collect_data(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

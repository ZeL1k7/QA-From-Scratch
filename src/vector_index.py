from abc import ABC, abstractmethod
import numpy as np


class IVectorIndex(ABC):
    @abstractmethod
    def build(self, dim: int):
        ...

    @abstractmethod
    def update(self, vector: np.array):
        ...

    @abstractmethod
    def get(self, vector: np.array, neighbors: int):
        ...

    def get_items(self, items: list[int]) -> list[str]:
        return [self.get(idx) for idx in items]

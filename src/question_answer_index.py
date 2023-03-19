from abc import ABC, abstractmethod
from collections import defaultdict
from datasets import AnswerDataset


class IQAIndex(ABC):
    @abstractmethod
    def build(self):
        ...

    @abstractmethod
    def update(self, item, idx):
        ...

    @abstractmethod
    def get(self, item):
        ...


class QAIndexHashMap(IQAIndex):
    def __init__(self, dataset: AnswerDataset) -> None:
        self._hash_map = defaultdict(list)
        self._dataset = dataset

    def build(self) -> None:
        for idx, item in enumerate(self._dataset):
            self._hash_map(item, idx)

    def update(self, item: int, idx: int) -> None:
        self._hash_map[item].append(idx)

    def get(self, item: int) -> list[int]:
        return self._hash_map[item]

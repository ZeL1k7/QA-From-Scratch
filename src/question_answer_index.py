from abc import ABC, abstractmethod


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

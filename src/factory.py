from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import torch

from qa_index import QAIndexHashMap
from utils import load_model, load_tokenizer
from vector_index import VectorIndexIVFFlat


class IFactory(ABC):
    @abstractmethod
    def create_tokenizer(self):
        ...

    @abstractmethod
    def create_model(self):
        ...

    @abstractmethod
    def create_vector_index(self):
        ...

    @abstractmethod
    def create_qa_index(self):
        ...


class QAFactory(IFactory):
    def __init__(
        self, vector_index_path: Path, qa_index_path: Path, device: torch.device
    ) -> None:
        self._vector_index_path = vector_index_path
        self._qa_index_path = qa_index_path
        self._device = device

    @lru_cache(1)
    def create_tokenizer(self):
        return load_tokenizer()

    @lru_cache(1)
    def create_model(self):
        return load_model(self._device)

    @lru_cache(1)
    def create_vector_index(self):
        return VectorIndexIVFFlat.from_pretrained(self._vector_index_path)

    @lru_cache(1)
    def create_qa_index(self):
        return QAIndexHashMap.from_pretrained(self._qa_index_path)

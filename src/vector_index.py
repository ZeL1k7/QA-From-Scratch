from abc import ABC, abstractmethod
from pathlib import Path
import faiss
import numpy as np
import torch
from utils import get_sentence_embedding, NotTrainedException
from transformers import AutoTokenizer, AutoModel
from datasets import QuestionDataset


class IVectorIndex(ABC):
    @abstractmethod
    def build(self):
        ...

    @abstractmethod
    def update(self, vector: np.array):
        ...

    @abstractmethod
    def get(self, vector: np.array, neighbors: int):
        ...

    def get_items(self, items: list[int]) -> list[str]:
        return [self.get(idx) for idx in items]


class VectorIndexIVFFlat(IVectorIndex):
    def __init__(self, n_splits: int, dim: int, neighbors: int) -> None:
        self.index = None
        self.dim = dim
        self.n_splits = n_splits
        self.neighbors = neighbors

    def build(self) -> faiss.Index:
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.n_splits)
        return index

    def update(self, vector: np.array) -> faiss.Index:
        self.index.add(vector)

    def get(self, query: np.array) -> list[list[float], list[int]]:
        distances, vectors = self.index.search(query, self.neighbors)
        return distances[0], vectors[0]

    def train(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        dataset: QuestionDataset,
        batch_size: int,
    ) -> None:
        if not self.index.is_trained:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            index_data = np.zeros((len(dataset), self.dim), dtype=np.float32)

            for idx, batch in enumerate(dataloader):
                sentence_embeddings = get_sentence_embedding(
                    batch=batch,
                    tokenizer=tokenizer,
                    model=model,
                )

                index_data[idx: (idx + 1)] = sentence_embeddings

            self.index.train(index_data)
        else:
            raise NotTrainedException(self.index)

    def save(self, index_path: Path) -> None:
        faiss.write_index(self.index, index_path)

    def load(self, index_path: Path) -> None:
        self.index = faiss.read_index(index_path)

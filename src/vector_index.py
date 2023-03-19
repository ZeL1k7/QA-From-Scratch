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


class VectorIndexIVFFlat(IVectorIndex):
    def __init__(self, n_splits: int) -> None:
        self.index = self.build()
        self.n_splits = n_splits

    def build(self, dim: int) -> faiss.Index:
        self.dim = dim
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.n_splits)
        return index

    def update(self, vector: np.array) -> faiss.Index:
        self.index.add(vector)

    def get(self, query: np.array, neighbors: int) -> list[list[float], list[int]]:
        distances, vectors = self.index.search(query, neighbors)
        return distances[0], vectors[0]

    def train_index(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        dataset: QuestionDataset,
        batch_size: int = 32,
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

    def save_index(self, index_path: Path) -> None:
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: Path) -> None:
        self.index = faiss.read_index(index_path)

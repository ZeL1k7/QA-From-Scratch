from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import QuestionDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import NotTrainedException, get_sentence_embedding


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
    def __init__(
        self,
        dim: int = 768,
        n_splits: int = 1,
        pretrained: bool = False,
    ) -> None:
        self.dim = dim
        self.n_splits = n_splits
        self.index = None
        if not pretrained:
            self.index = self.build()

    def build(self) -> faiss.Index:
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.n_splits)
        return index

    def update(self, vector: np.array) -> faiss.Index:
        self.index.add(vector)

    def get(self, query: np.array, neighbors: int) -> list[list[float], list[int]]:
        distances, vectors = self.index.search(query, neighbors)
        return distances[0], vectors[0]

    def train(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        dataset: QuestionDataset,
        batch_size: int,
    ) -> None:
        if not self.index.is_trained:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            index_data = np.zeros((len(dataset), self.dim), dtype=np.float32)

            for idx, batch in tqdm(enumerate(dataloader)):
                sentence_embeddings = get_sentence_embedding(
                    batch=batch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )

                index_data[
                    batch_size * idx : batch_size * (idx + 1)
                ] = sentence_embeddings

            self.index.train(index_data)
        else:
            raise NotTrainedException(self.index)

    def save(self, index_path: Path) -> None:
        faiss.write_index(self.index, index_path)

    def load(self, index_path: Path) -> None:
        self.index = faiss.read_index(index_path)
        self.n_splits = self.index.nlist
        self.dim = self.index.d

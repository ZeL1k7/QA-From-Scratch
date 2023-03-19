from abc import ABC, abstractmethod
from collections import defaultdict
from datasets import AnswerDataset
from transformers import AutoTokenizer, AutoModel
from vector_index import IVectorIndex
from utils import get_sentence_embedding

class IQAIndex(ABC):
    @abstractmethod
    def build(self):
        ...

    @abstractmethod
    def update(self, idx, item):
        ...

    @abstractmethod
    def get(self, idx):
        ...

    def get_items(self, items: list[int]) -> list[str]:
        return [self.get(idx) for idx in items]


class QAIndexHashMap(IQAIndex):
    def __init__(self, dataset: AnswerDataset) -> None:
        self._hash_map = defaultdict(list)
        self._dataset = dataset

    def build(self) -> None:
        for idx, item in enumerate(self._dataset):
            self.update(idx, item)

    def update(self, idx: int, item: str) -> None:
        self._hash_map[idx].append(item)

    def get(self, idx: int) -> list[str]:
        return self._hash_map[idx]

def get_answer(
    index: IVectorIndex,
    qa_index: IQAIndex,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: list[str],
    neighbors: int = 4,
) -> list[str]:
    query = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
    )

    distances, question_idxs = index.get(query, neighbors)

    return qa_index.get_items(question_idxs)
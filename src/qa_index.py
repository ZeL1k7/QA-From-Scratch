import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer

from datasets import AnswerDataset, Answer, QuestionDataset
from utils import get_sentence_embedding
from vector_index import IVectorIndex


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
    def __init__(
        self,
        question_dataset: QuestionDataset,
        answer_dataset: AnswerDataset,
        hash_map_question: Optional[dict] = defaultdict(None),
        hash_map_answer: Optional[dict] = defaultdict(list),
    ) -> None:
        self._hash_map_question = hash_map_question
        self._hash_map_answer = hash_map_answer
        self._question_dataset = question_dataset
        self._answer_dataset = answer_dataset

    @classmethod
    def from_pretrained(cls, index_path: Path) -> "QAIndexHashMap":
        with open(index_path, "rb+") as f:
            index = pickle.load(f)
        hash_map_question = index._hash_map_question
        hash_map_answer = index._hash_map_answer
        question_dataset = index._question_dataset
        answer_dataset = index._answer_dataset
        return cls(
            hash_map_question=hash_map_question,
            hash_map_answer=hash_map_answer,
            question_dataset=question_dataset,
            answer_dataset=answer_dataset,
        )

    def build(self) -> None:
        for idx, _ in enumerate(self._question_dataset):
            self._hash_map_question[idx] = self._question_dataset.__getid__(idx)

        for item in self._answer_dataset:
            self.update(item.parent_id, item)

    def update(self, idx: int, item: Answer) -> None:
        self._hash_map_answer[idx].append(item)

    def get(self, idx: int) -> Answer:
        parent_id = self._hash_map_question[idx]
        return self._hash_map_answer[parent_id]


def save_qa_index(qa_index: IQAIndex, index_save_path: Path) -> None:
    with open(index_save_path, "wb+") as f:
        pickle.dump(qa_index, f)


def get_answer(
    index: IVectorIndex,
    qa_index: IQAIndex,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    sentence: Union[str, list[str], list[list[str]]],
    neighbors: int,
) -> list[Answer]:
    query = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    _, question_idxs = index.get(query, neighbors)

    return qa_index.get_items(question_idxs)

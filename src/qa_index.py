import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import torch
from datasets import AnswerDataset, QuestionDataset, Question
from transformers import AutoModel, AutoTokenizer
from utils import get_sentence_embedding
from vector_index import IVectorIndex

from typing import Union


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
        self, question_dataset: QuestionDataset, answer_dataset: AnswerDataset
    ) -> None:
        self._hash_map_question = defaultdict(None)
        self._hash_map_answer = defaultdict(list)
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
            _hash_map_questions=hash_map_question,
            _hash_map_answer=hash_map_answer,
            _questions_dataset=question_dataset,
            _answer_dataset=answer_dataset,
        )

    def build(self) -> None:
        for idx, _ in enumerate(self._question_dataset):
            self._hash_map_question[idx] = self._question_dataset.__getid__(idx)

        for item in self._answer_dataset:
            self.update(item.parent_id, item)

    def update(self, idx: int, item: Question) -> None:
        self._hash_map_answer[idx].append(item)

    def get(self, idx: int) -> Question:
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
) -> list[Question]:
    query = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    distances, question_idxs = index.get(query, neighbors)

    return qa_index.get_items(question_idxs)

from abc import ABC, abstractmethod
from collections import defaultdict
from datasets import QuestionDataset, AnswerDataset
from transformers import AutoTokenizer, AutoModel
from vector_index import IVectorIndex
from utils import get_sentence_embedding
import torch
from pathlib import Path
import pickle


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
    def __init__(self, question_dataset: QuestionDataset, answer_dataset: AnswerDataset) -> None:
        self._hash_map_question = defaultdict(None)
        self._hash_map_answer = defaultdict(list)
        self._question_dataset = question_dataset
        self._answer_dataset = answer_dataset

    def build(self) -> None:
        for idx in range(len(self._question_dataset)):  # hash_map[vector_idx] = parent_idx
            self._hash_map_question[idx] = self._question_dataset.__getidx__(idx)

        for item in self._answer_dataset:  # hash_map[parent_id] = answer
            self.update(item.parent_id, item)

    def update(self, idx: int, item: str) -> None:  #append parent_id -> answer
        self._hash_map_answer[idx].append(item)

    def get(self, idx: int) -> list[str]:
        parent_id = self._hash_map_question[idx]
        return self._hash_map_answer[parent_id]


def save_qa_index(qa_index: IQAIndex, index_save_path: Path) -> None:
    with open(index_save_path, "wb+") as f:
        pickle.dump(qa_index, f)


def load_qa_index(index_path: Path) -> IQAIndex:
    with open(index_path, "rb+") as f:
        qa_index = pickle.load(f)
    return qa_index


def get_answer(
    index: IVectorIndex,
    qa_index: IQAIndex,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    sentence: list[str],
) -> list[str]:
    query = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    distances, question_idxs = index.get(query)

    return qa_index.get_items(question_idxs)

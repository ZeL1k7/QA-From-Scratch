from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
import torch


class Question(BaseModel):
    id: int = Field(..., description="Unique identifier")
    text: str = Field(..., description="Title of the question")


class Questions(BaseModel):
    items: List[Question]


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, questions: Questions) -> None:
        super().__init__()
        self._questions = questions

    @classmethod
    def from_json(cls, path: Path):
        questions = Questions.parse_file(path)
        return cls(questions=questions)

    def __getitem__(self, idx):
        return self.texts[idx]

    def __len__(self):
        return len(self.texts)

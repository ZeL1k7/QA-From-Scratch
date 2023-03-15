from pathlib import Path
import torch
import pandas as pd


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, questions_path: Path, answers_path: Path):
        self._question_df = pd.read_csv(questions_path, encoding="latin-1")
        self._question_df = self._question_df[["Id", "Title", "Body"]]
        self._answer_df = pd.read_csv(answers_path, encoding="latin-1")
        self._answer_df = self._answer_df[["Id", "ParentId", "Body", "Score"]]
        self._dataset = self._question_df.merge(
            self._answer_df,
            how="left",
            left_on="Id",
            right_on="ParentId",
        )
        self._dataset = self._dataset[self._dataset.Body_y.notna()]
        self._dataset = self._dataset.reset_index(drop=True)

    def __getitem__(self, idx):
        question = self._question_df.iloc[idx]
        question = [question.Title, question.Body]
        return question

    def __len__(self):
        return len(self._dataset)

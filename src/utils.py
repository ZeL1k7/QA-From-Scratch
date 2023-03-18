from pathlib import Path
import torch
import pandas as pd


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, questions_path: Path):
        self._question_df = pd.read_csv(questions_path, encoding="latin-1")
        self._question_df = self._question_df[["Id", "Title", "Body"]]

    def __getitem__(self, idx):
        question = self._question_df.iloc[idx].Title
        return question

    def __len__(self):
        return len(self._dataset)


def mean_pooling(model_output: torch.FloatTensor, attention_mask: torch.BoolTensor) -> torch.FloatTensor:
    """
    Make sentence embedding averaging word embeddings
    :param model_output:
    :param attention_mask:
    :return:
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

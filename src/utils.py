from functools import lru_cache
from pathlib import Path
import torch
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, questions_path: Path):
        self._question_df = pd.read_csv(questions_path, encoding="latin-1")
        self._question_df = self._question_df[["Id", "Title", "Body"]]

    def __getitem__(self, idx):
        question = self._question_df.iloc[idx].Title
        return question

    def __len__(self):
        return len(self._question_df)


class AnswerDataset:
    def __init__(self, answers_path: Path):
        self._answers_df = pd.read_csv(answers_path, encoding="latin-1")
        self._answers_df = self._answers_df[["ParentId", "Body", "Score"]]

    def __getitem__(self, parent_idx):
        answers = self._answers_df[self._answers_df.ParentId == parent_idx]
        if answers.empty:
            return "Answers not found"
        answers = answers.sort_values(by="Score", ascending=False)
        return answers


@lru_cache(1)
def load_model(device: torch.device = "cpu") -> torch.nn.Module:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)
    return tokenizer, model


def mean_pooling(
    model_output: torch.FloatTensor, attention_mask: torch.BoolTensor
) -> torch.FloatTensor:
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


def get_sentence_embedding(
    sentense: str, tokenizer: AutoTokenizer, model: AutoModel
) -> torch.FloatTensor:
    encoded_input = tokenizer(
        sentense,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    word_embeddings = model(**encoded_input)
    sentence_embedding = mean_pooling(word_embeddings, encoded_input["attention_mask"])
    sentence_embedding = sentence_embedding.detach().cpu().numpy()
    return sentence_embedding


def create_index(embedding_dim: int = 384, nlist: int = 4500) -> faiss.IndexIVFFlat:
    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
    return index

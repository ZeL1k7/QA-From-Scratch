from functools import lru_cache
from pathlib import Path
import torch
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self._texts = texts

    @classmethod
    def from_df(
        cls,
        path: Path,
        text_column_name: str,
        **dataframe_kwargs,
    ) -> "QuestionDataset":
        dataframe = pd.read_csv(path, **dataframe_kwargs)
        texts = dataframe[text_column_name].tolist()
        return cls(texts=texts)

    def __getitem__(self, idx):
        return self._texts[idx]

    def __len__(self):
        return len(self._texts)


@lru_cache(1)
def load_model(device: torch.device = "cpu") -> AutoModel:
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)
    return model


@lru_cache(1)
def load_tokenizer(*args, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer


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
    sentence: str, tokenizer: AutoTokenizer, model: AutoModel
) -> torch.FloatTensor:
    encoded_input = tokenizer(
        sentence,
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

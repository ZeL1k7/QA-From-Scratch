import math
from functools import lru_cache
from typing import Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer


@lru_cache(1)
def load_model(device: torch.device = "cpu") -> AutoModel:
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)
    return model


@lru_cache(1)
def load_tokenizer(**tokenizer_kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", **tokenizer_kwargs
    )
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
    batch: Union[str, list[str], list[list[str]]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.FloatTensor:
    encoded_input = tokenizer(
        batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    word_embeddings = model(**encoded_input)
    sentence_embedding = mean_pooling(
        model_output=word_embeddings,
        attention_mask=encoded_input["attention_mask"],
    )
    sentence_embedding = sentence_embedding.detach().cpu().numpy()
    return sentence_embedding


def get_n_splits(dataset_size: Optional[int] = None, n_splits: int = 1) -> int:
    """
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    :param dataset_size:
    :param n_splits:
    :return:
    """
    if dataset_size is not None:
        return int(4 * math.sqrt(dataset_size))

    return n_splits


class NotTrainedException(Exception):
    def __init__(self, index):
        self.index_type = type(index).__name__

    def __str__(self):
        return f"{self.index_type} should be trained before adding new vectors"

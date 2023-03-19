import math
from functools import lru_cache
from pathlib import Path
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


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
    batch: list[str], tokenizer: AutoTokenizer, model: AutoModel
) -> torch.FloatTensor:
    encoded_input = tokenizer(
        batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    word_embeddings = model(**encoded_input)
    sentence_embedding = mean_pooling(
        model_output=word_embeddings,
        attention_mask=encoded_input["attention_mask"],
    )
    sentence_embedding = sentence_embedding.detach().cpu().numpy()
    return sentence_embedding


def create_index(embedding_dim: int = 384, n_splits: int = 4500) -> faiss.Index:
    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_splits)
    return index


def get_n_splits(dataset_size: int, n_splits: int = None) -> int:
    """
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    :param dataset_size:
    :param n_splits:
    :return:
    """
    if n_splits is None:
        n_splits = int(4 * math.sqrt(dataset_size))
    return n_splits


def save_index(index: faiss.Index, index_path: Path) -> None:
    faiss.write_index(index, index_path)

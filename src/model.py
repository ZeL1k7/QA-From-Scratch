import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from utils import get_sentence_embedding
from datasets import QuestionDataset
from question_answer_index import IQAIndex


def train_index(
    index: faiss.Index,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    dataset: QuestionDataset,
    embedding_dim: int = 384,
    batch_size: int = 32,
) -> faiss.Index:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    index_data = np.zeros((len(dataset), embedding_dim), dtype=np.float32)

    for idx, batch in enumerate(dataloader):
        sentence_embeddings = get_sentence_embedding(
            batch=batch,
            tokenizer=tokenizer,
            model=model,
        )

        index_data[idx: (idx + 1)] = sentence_embeddings

    index.train(index_data)
    return index


def add_sentence_to_index(
    index: faiss.Index,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: list[str],
) -> faiss.Index:
    sentence_embeddings = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
    )
    index.add(sentence_embeddings)
    return index


def get_answer(
    index: faiss.Index,
    qa_index: IQAIndex,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: list[str],
    neighbors: int = 4,
) -> list[str]:
    query = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
    )

    distances, question_idxs = index.search(query, neighbors)

    return qa_index.get_items(question_idxs[0])

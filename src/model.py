import pandas as pd
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils import QuestionDataset, get_sentence_embedding


def train_index(
    index: faiss.IndexIVFFlat,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    dataset: QuestionDataset,
    embedding_dim: int = 384,
    batch_size: int = 32,
) -> faiss.IndexIVFFlat:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    index_data = np.zeros(
        (embedding_dim * len(dataset), embedding_dim), dtype=np.float32
    )

    for idx, batch in enumerate(dataloader):
        sentence_embeddings = get_sentence_embedding(batch, tokenizer, model)

        index_data[
            idx * embedding_dim : (idx + 1) * embedding_dim
        ] = sentence_embeddings

    index.train(index_data)
    return index


def add_sentence_to_index(
    index: faiss.IndexIVFFlat,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: str,
    save_index: bool = True,
) -> faiss.IndexIVFFlat:
    sentence_embeddings = get_sentence_embedding(sentence, tokenizer, model)
    index.add(sentence_embeddings)

    if save_index:
        faiss.write_index(index, "question.index")

    return index


def get_answer(
    index: faiss.IndexIVFFlat,
    question_dataset: QuestionDataset,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: str,
    neighbors: int = 4,
) -> pd.DataFrame:
    query = get_sentence_embedding(sentence, tokenizer, model)
    distances, question_idxs = index.search(query, neighbors)
    answers_idxs = question_dataset.question_df.iloc[question_idxs[0]].Id.values
    answer_df = pd.DataFrame()
    for idx in answers_idxs:
        answer_df = pd.concat([answer_df, question_dataset.get_answer(idx)])
    return answer_df

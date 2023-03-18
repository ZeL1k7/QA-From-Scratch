import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils import mean_pooling


def train_index(
    index: faiss.IndexIVFFlat,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    dataset: torch.utils.data.Dataset,
    embedding_dim: int = 384,
    batch_size: int = 32,
) -> faiss.IndexIVFFlat:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    index_data = np.zeros(
        (embedding_dim * len(dataloader), embedding_dim), dtype=np.float32
    )

    for idx, batch in enumerate(dataloader):
        encoded_input = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        word_embeddings = model(encoded_input)
        sentence_embeddings = mean_pooling(
            word_embeddings, encoded_input["attention_mask"]
        )
        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()
        index_data[
            idx * embedding_dim: (idx + 1) * embedding_dim
        ] = sentence_embeddings

    index.train(index_data)
    return index


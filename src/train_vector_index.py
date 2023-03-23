from pathlib import Path
import torch
import typer
from utils import (
    load_model,
    load_tokenizer,
    get_n_splits,
)
from vector_index import VectorIndexIVFFlat
from datasets import QuestionDataset
from utils import get_sentence_embedding


def main(question_path: Path, index_save_path: Path, batch_size: int, device: str):
    dataset = QuestionDataset.from_json(question_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    tokenizer = load_tokenizer()
    model = load_model(device)

    n_splits = get_n_splits(dataset_size=len(dataset))

    index = VectorIndexIVFFlat(dim=384, n_splits=n_splits)
    index.build()

    index.train(
        tokenizer=tokenizer,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
    )

    for batch in dataloader:
        batch = get_sentence_embedding(
            batch=batch,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        index.update(batch)

    index.save(index_save_path)


if __name__ == "__main__":
    typer.run(main)

from pathlib import Path
import torch
import typer
from utils import (
    load_model,
    load_tokenizer,
    create_index,
    get_n_splits,
    save_index,
)
from model import train_index, add_sentence_to_index
from datasets import QuestionDataset


def main(question_path: Path, index_path: Path, batch_size: int, device: str):
    dataset = QuestionDataset.from_json(question_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    tokenizer = load_tokenizer()
    model = load_model(device)

    n_splits = get_n_splits(dataset_size=len(dataset))

    index = create_index(embedding_dim=384, n_splits=n_splits)
    index = train_index(
        index=index,
        tokenizer=tokenizer,
        model=model,
        dataset=dataset,
        embedding_dim=384,
        batch_size=32,
    )

    for batch in dataloader:
        add_sentence_to_index(
            index=index,
            tokenizer=tokenizer,
            model=model,
            sentence=batch,
        )

    save_index(index=index, index_path=index_path)


if __name__ == "__main__":
    typer.run(main)

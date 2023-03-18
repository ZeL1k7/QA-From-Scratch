import argparse
import math
import torch
from utils import QuestionDataset, load_model, load_tokenizer, create_index, get_n_splits, save_index
from model import train_index, add_sentence_to_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question_path", help="Path to question dataset")
    parser.add_argument("answer_path", help="Path to answer dataset")
    parser.add_argument("index_path", help="Path where save index")
    parser.add_argument("batch_size", help="batch_size")
    parser.add_argument("device", help="device")
    args = parser.parse_args()

    dataset = QuestionDataset(args.question_path, args.answer_path)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size)
    device = args.device
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

    save_index(index=index, index_path=args.index_path)

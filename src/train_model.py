import argparse
import math
import torch
import faiss
from utils import QuestionDataset, load_model, create_index
from model import train_index, add_sentence_to_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question_path", help="Path to question dataset")
    parser.add_argument("index_path", help="Path where save index")
    parser.add_argument("batch_size", help="batch_size")
    parser.add_argument("device", help="device")
    args = parser.parse_args()

    dataset = QuestionDataset(args.question_path)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size)
    device = args.device
    model, tokenizer = load_model(device)
    nlist = int(4 * math.sqrt(len(dataset)))

    index = create_index(384, nlist)
    index = train_index(index, tokenizer, model, dataset, 384, 32)

    for batch in dataloader:
        add_sentence_to_index(index, tokenizer, model, batch, False)

    faiss.write_index(index, args.index_path)

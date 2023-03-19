import typer
from pathlib import Path
from datasets import AnswerDataset
from qa_index import QAIndexHashMap, save_qa_index


def main(answer_path: Path, index_save_path: Path):
    answer_dataset = AnswerDataset.from_json(answer_path)
    qa_index = QAIndexHashMap(answer_dataset)
    qa_index.build()
    save_qa_index(qa_index=qa_index, index_save_path=index_save_path)


if __name__ == "__main__":
    typer.run(main)

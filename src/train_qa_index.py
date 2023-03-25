from pathlib import Path

import typer

from datasets import AnswerDataset, QuestionDataset
from qa_index import QAIndexHashMap, save_qa_index


def main(question_path: Path, answer_path: Path, index_save_path: Path):
    answer_dataset = AnswerDataset.from_json(answer_path)
    question_dataset = QuestionDataset.from_json(question_path)
    qa_index = QAIndexHashMap(question_dataset, answer_dataset)
    qa_index.build()
    save_qa_index(qa_index=qa_index, index_save_path=index_save_path)


if __name__ == "__main__":
    typer.run(main)

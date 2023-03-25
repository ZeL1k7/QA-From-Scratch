from typing import Union

import torch
from fastapi import FastAPI

from factory import QAFactory
from qa_index import get_answer

app = FastAPI()


device = "cuda" if torch.cuda.is_available() else "cpu"

factory = QAFactory(
    vector_index_path="./app/data/vector.index",
    qa_index_path="./app/data/qa_index.pkl",
    device=device,
)

tokenizer = factory.create_tokenizer()
model = factory.create_model()
index = factory.create_vector_index()
qa_index = factory.create_qa_index()


@app.get("/send_answer")
def send_answer(question: str, num_answers: int):
    answer = get_answer(
        index=index,
        qa_index=qa_index,
        tokenizer=tokenizer,
        model=model,
        device=device,
        sentence=question,
        neighbors=num_answers,
    )
    return answer

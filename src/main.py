import torch
from fastapi import FastAPI
from qa_index import get_answer, QAIndexHashMap
from utils import load_model, load_tokenizer
from vector_index import VectorIndexIVFFlat

from typing import Union

app = FastAPI()


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = load_tokenizer()
model = load_model(device=device)
index = VectorIndexIVFFlat.from_pretrained("data/vector.index")
qa_index = QAIndexHashMap.from_pretrained("data/qa_index.pkl")


@app.get("/send_answer")
def send_answer(question: Union[str, list[str], list[list[str]]], num_answers: int):
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

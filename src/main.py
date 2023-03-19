import torch
import subprocess
from fastapi import FastAPI
from utils import load_tokenizer, load_model
from vector_index import VectorIndexIVFFlat
from qa_index import QAIndexHashMap, load_qa_index, get_answer

app = FastAPI()


def create_vector_index():
    subprocess.run(
        [
            "python",
            "train_vector_index.py",
            "../data/Questions.json",
            "../data/q.index",
            "32",
            "cuda",
        ]
    )


def create_qa_index():
    subprocess.run(
        ["python", "train_qa_index.py", "../data/Answers.json", "../data/qa.pkl"]
    )


@app.on_event("startup")
def startup_event():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model = load_model(device=device)
    create_vector_index()
    create_qa_index()


@app.post("/send_question")
def send_question(question: str):
    return question


@app.get("/send_answer")
def send_answer(question: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model = load_model(device=device)
    index = VectorIndexIVFFlat(..., ..., ..., pretrained=True)
    index.load("../data/q.index")
    qa_index = QAIndexHashMap(...)
    qa_index = load_qa_index("../data/qa.pkl")
    answer = get_answer(
        index=index,
        qa_index=qa_index,
        tokenizer=tokenizer,
        model=model,
        device=device,
        sentence=[question],
    )

import torch
from fastapi import FastAPI
from utils import load_tokenizer, load_model
from vector_index import VectorIndexIVFFlat
from qa_index import load_qa_index, get_answer


app = FastAPI()


@app.get("/send_answer")
def send_answer(question: str, num_answers: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model = load_model(device=device)
    index = VectorIndexIVFFlat(neighbors=num_answers, pretrained=True)
    index.load("../data/question.index")
    qa_index = load_qa_index("../data/test_qa.pkl")
    answer = get_answer(
        index=index,
        qa_index=qa_index,
        tokenizer=tokenizer,
        model=model,
        device=device,
        sentence=[question],
    )

    return answer

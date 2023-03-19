from transformers import AutoTokenizer, AutoModel
from utils import get_sentence_embedding
from question_answer_index import IQAIndex
from vector_index import IVectorIndex


def get_answer(
    index: IVectorIndex,
    qa_index: IQAIndex,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sentence: list[str],
    neighbors: int = 4,
) -> list[str]:
    query = get_sentence_embedding(
        batch=sentence,
        tokenizer=tokenizer,
        model=model,
    )

    distances, question_idxs = index.get(query, neighbors)

    return qa_index.get_items(question_idxs)

import streamlit as st
import pandas as pd
import requests


@st.cache_data
def load_answers(question, num_answers=15):
    request = requests.get(
        "http://backend:8080/send_answer",
        params={"question": question, "num_answers": num_answers},
    ).json()

    df = pd.DataFrame(request)

    answers_base = [
        answer["text"] for item in df for answer in df[item] if answer is not None
    ]

    return answers_base


question = st.text_input("Enter your question")

if question:
    answers_base = load_answers(question)
    count = st.number_input(
        "How many answers do you want to show?",
        min_value=0,
        max_value=len(answers_base),
    )
    for i in range(count):
        st.markdown("---")
        st.markdown(
            "<div style='margin-top: 10px'>" + answers_base[i] + "</div>",
            unsafe_allow_html=True,
        )

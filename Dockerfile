FROM python:3.10

WORKDIR /app

COPY src/ /app/src/
COPY data/ /app/data/

COPY requirements.txt /app/

RUN pip install -r requirements.txt

WORKDIR /app/src

CMD ["uvicorn", "main:app", "--reload", "--port=8080"]

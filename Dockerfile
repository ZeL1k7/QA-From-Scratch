FROM python:3.9-alpine

WORKDIR /app

COPY src/ /app/src/
COPY data/ /app/data/

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

WORKDIR /app/src
CMD ["uvicorn", "main:app", "--reload", "--port=8080"]

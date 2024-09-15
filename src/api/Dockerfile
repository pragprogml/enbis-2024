FROM python:3.10.15 as base

ENV PORT=8080

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY src/api/predict.py .
COPY models/best_model.pt models/best_model.pt

ENV MODEL /app/models/best_model.pt

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]

FROM python:3.10.13 as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file and install Python dependencies
COPY src/api/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY src/api/predict.py .
COPY models/best_model.pt models/best_model.pt

ENV MODEL /app/models/best_model.pt

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9090"]

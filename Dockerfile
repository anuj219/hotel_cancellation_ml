FROM python:3.11-slim

WORKDIR /app

# Install curl for downloading the model
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# COPY model ./model

RUN mkdir -p /app/model && \
    curl -fL \
    -o /app/model/hotel_cancellation_pipeline.pkl \
    "https://github.com/anuj219/hotel_cancellation_ml/releases/download/v1.0-model/hotel_cancellation_pipeline.pkl"

EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]   
# starts FastAPI inside the container
# this new version allows render or other platform to configure the container port based on their infra
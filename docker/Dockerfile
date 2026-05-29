FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["sh", "-c", "python -m mlflow server \
     --backend-store-uri ${DATABASE_URL} \
     --artifacts-destination ./mlartifacts \
     --serve-artifacts \
     --host 0.0.0.0 --port 5000 \
     --allowed-hosts '*'"]
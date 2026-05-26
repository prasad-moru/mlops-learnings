# Dockerfile — MLflow Tracking Server
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Start MLflow tracking server
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow_user:your_password@db/mlflow_db", \
     "--default-artifact-root", "./mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

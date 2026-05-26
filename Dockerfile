# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow_user:password@db/mlflow_db", \
     "--default-artifact-root", "./mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
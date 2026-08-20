FROM python:3.12.13-slim-bookworm
WORKDIR /app
COPY cloud/ml_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cloud/ml_worker/worker.py .
CMD ["python", "worker.py"]

FROM python:3.12.13-slim-bookworm
WORKDIR /app
COPY cloud/ingestor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cloud/ingestor/ingest.py cloud/ingestor/transform.py cloud/ingestor/validation.py cloud/ingestor/ack.py ./
COPY schemas/telemetry.schema.json /app/schema/telemetry.schema.json
CMD ["python", "ingest.py"]

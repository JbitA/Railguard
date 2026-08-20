import io
import os
from minio import Minio

ENDPOINT=os.getenv("MINIO_ENDPOINT","minio:9000")
ACCESS=os.getenv("MINIO_ROOT_USER","railguard")
SECRET=os.getenv("MINIO_ROOT_PASSWORD","railguard_minio_dev")
BUCKET=os.getenv("MINIO_BUCKET","railguard-events")

client=Minio(ENDPOINT, access_key=ACCESS, secret_key=SECRET, secure=False)

def ensure_bucket():
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

def put_bytes(name: str, data: bytes, content_type: str) -> str:
    ensure_bucket()
    client.put_object(BUCKET, name, io.BytesIO(data), len(data), content_type=content_type)
    return f"{BUCKET}/{name}"

def get_bytes(name: str) -> tuple[bytes,str]:
    ensure_bucket()
    response=client.get_object(BUCKET,name)
    try:
        return response.read(), response.headers.get("content-type","application/octet-stream")
    finally:
        response.close(); response.release_conn()

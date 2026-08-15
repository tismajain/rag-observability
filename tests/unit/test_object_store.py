from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import ObjectStorageSettings
from src.storage.object_store import S3ObjectStore


async def test_minio_put_does_not_require_unconfigured_kms() -> None:
    client = MagicMock()
    with patch("src.storage.object_store.boto3.client", return_value=client):
        store = S3ObjectStore(ObjectStorageSettings(endpoint_url="http://minio:9000"))
    await store.put("documents/id", b"body", "text/plain")
    kwargs = client.put_object.call_args.kwargs
    assert "ServerSideEncryption" not in kwargs


async def test_s3_put_applies_configured_server_side_encryption() -> None:
    client = MagicMock()
    with patch("src.storage.object_store.boto3.client", return_value=client):
        store = S3ObjectStore(
            ObjectStorageSettings(endpoint_url=None, server_side_encryption="AES256")
        )
    await store.put("documents/id", b"body", "text/plain")
    assert client.put_object.call_args.kwargs["ServerSideEncryption"] == "AES256"

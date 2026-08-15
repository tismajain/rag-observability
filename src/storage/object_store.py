"""S3-compatible object storage with an injectable, provider-neutral interface."""

from __future__ import annotations

import asyncio
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.config.settings import ObjectStorageSettings, settings


class ObjectNotFoundError(Exception):
    pass


class ObjectStore(Protocol):
    async def ensure_bucket(self) -> None: ...
    async def put(self, key: str, body: bytes, content_type: str) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


class S3ObjectStore:
    """Async facade over boto3; endpoint_url makes the same code work with MinIO."""

    def __init__(self, config: ObjectStorageSettings | None = None) -> None:
        self._config = config or settings.object_storage
        kwargs: dict[str, object] = {
            "service_name": "s3",
            "region_name": self._config.region,
            "config": Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        }
        if self._config.endpoint_url:
            kwargs["endpoint_url"] = self._config.endpoint_url
        if self._config.access_key_id is not None:
            kwargs["aws_access_key_id"] = self._config.access_key_id.get_secret_value()
        if self._config.secret_access_key is not None:
            kwargs["aws_secret_access_key"] = self._config.secret_access_key.get_secret_value()
        self._client = boto3.client(**kwargs)
        self._bucket = self._config.bucket

    async def ensure_bucket(self) -> None:
        def ensure() -> None:
            try:
                self._client.head_bucket(Bucket=self._bucket)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code not in {"404", "NoSuchBucket", "NotFound"}:
                    raise
                args: dict[str, object] = {"Bucket": self._bucket}
                if self._config.region != "us-east-1":
                    args["CreateBucketConfiguration"] = {"LocationConstraint": self._config.region}
                self._client.create_bucket(**args)

        await asyncio.to_thread(ensure)

    async def put(self, key: str, body: bytes, content_type: str) -> None:
        args: dict[str, object] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": body,
            "ContentType": content_type,
        }
        if self._config.server_side_encryption is not None:
            args["ServerSideEncryption"] = self._config.server_side_encryption
        await asyncio.to_thread(self._client.put_object, **args)

    async def get(self, key: str) -> bytes:
        def read() -> bytes:
            try:
                response = self._client.get_object(Bucket=self._bucket, Key=key)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code in {"404", "NoSuchKey", "NotFound"}:
                    raise ObjectNotFoundError from None
                raise
            return bytes(response["Body"].read())

        return await asyncio.to_thread(read)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._client.delete_object, Bucket=self._bucket, Key=key)


_object_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    global _object_store
    if _object_store is None:
        _object_store = S3ObjectStore()
    return _object_store


def set_object_store_for_tests(store: ObjectStore | None) -> None:
    global _object_store
    _object_store = store

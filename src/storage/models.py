"""ORM models for the observability tables.

Three tables, one row per (trace × event-kind):

* :class:`Trace` — one row per ``/query`` request, headline fields only.
* :class:`DefectEvent` — N rows per query, one per detected defect.
* :class:`EvalScore` — 0..1 rows per query (only when the async evaluator
  ran and produced or failed scores).

Index choices match the dashboard query patterns the spec calls out:
time-range scans on ``created_at``, filtered fetches by defect type/severity,
and quality-gate distribution histograms.

JSONB is used for free-form fields (``DefectEvent.event_metadata``). We use
the SQLAlchemy ``JSON`` type which maps to JSONB on Postgres and JSON on
SQLite — letting tests run against in-memory SQLite.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Single declarative base for all ORM models."""


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Trace(Base):
    __tablename__ = "trace"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    query_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    trace_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    context_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_strategy: Mapped[str] = mapped_column(String(32), nullable=False, default="hybrid")
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_used: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    organization_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    defects: Mapped[list[DefectEvent]] = relationship(
        back_populates="trace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    evals: Mapped[list[EvalScore]] = relationship(
        back_populates="trace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    __table_args__ = (Index("ix_trace_created_at_desc", created_at.desc()),)


class DefectEvent(Base):
    __tablename__ = "defect_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    query_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("trace.query_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trace_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    defect_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # SQLAlchemy's reserved attribute is `metadata` on the declarative base, so
    # we name the column event_metadata and map it to a JSON payload column.
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    organization_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    trace: Mapped[Trace] = relationship(back_populates="defects")


class EvalScore(Base):
    __tablename__ = "eval_score"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    query_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("trace.query_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trace_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_gate_result: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    evaluation_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    organization_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    trace: Mapped[Trace] = relationship(back_populates="evals")


class Organization(Base):
    __tablename__ = "organization"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AppUser(Base):
    __tablename__ = "app_user"

    subject: Mapped[str] = mapped_column(String(255), primary_key=True)
    organization_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("organization.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentRecord(Base):
    __tablename__ = "document"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    organization_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("organization.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    owner_user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("app_user.subject", ondelete="RESTRICT"), nullable=False, index=True
    )
    visibility: Mapped[str] = mapped_column(String(16), nullable=False, default="private")
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(255), nullable=False, default="application/octet-stream"
    )
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lifecycle_state: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "visibility IN ('private', 'organization', 'shared')",
            name="ck_document_visibility",
        ),
        CheckConstraint(
            "lifecycle_state IN ('active', 'deleting', 'deleted')",
            name="ck_document_lifecycle_state",
        ),
    )


class DocumentShare(Base):
    __tablename__ = "document_share"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("app_user.subject", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("app_user.subject", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("document_id", "user_id", name="uq_document_share_user"),)


class IngestionJob(Base):
    __tablename__ = "ingestion_job"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True
    )
    organization_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    requested_by_user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="ck_ingestion_job_status",
        ),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True
    )
    organization_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    owner_user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    qdrant_point_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )

"""document lifecycle and object metadata

Revision ID: d4e81f70a922
Revises: 8b6f4b2d19a1
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e81f70a922"
down_revision: str | None = "8b6f4b2d19a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document",
        sa.Column("source_filename", sa.String(512), nullable=False, server_default="document.txt"),
    )
    op.add_column(
        "document",
        sa.Column(
            "content_type",
            sa.String(255),
            nullable=False,
            server_default="application/octet-stream",
        ),
    )
    op.add_column(
        "document", sa.Column("byte_size", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "document",
        sa.Column("lifecycle_state", sa.String(16), nullable=False, server_default="active"),
    )
    op.add_column("document", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_check_constraint(
        "ck_document_lifecycle_state",
        "document",
        "lifecycle_state IN ('active', 'deleting', 'deleted')",
    )
    op.drop_constraint("ck_ingestion_job_status", "ingestion_job", type_="check")
    op.execute("UPDATE ingestion_job SET status='queued' WHERE status='pending'")
    op.execute("UPDATE ingestion_job SET status='processing' WHERE status='running'")
    op.execute("UPDATE ingestion_job SET status='failed' WHERE status='cancelled'")
    op.create_check_constraint(
        "ck_ingestion_job_status",
        "ingestion_job",
        "status IN ('queued', 'processing', 'completed', 'failed')",
    )
    op.alter_column("ingestion_job", "status", server_default="queued")
    if op.get_bind().dialect.name == "postgresql":
        # Organization membership alone never grants document content access.
        op.execute("DROP POLICY IF EXISTS document_access ON document")
        op.execute(
            """
            CREATE POLICY document_access ON document
            USING (
              lifecycle_state = 'active' AND (
                current_setting('app.document_content_admin', true) = 'true'
                OR owner_user_id = current_setting('app.user_id', true)
                OR (visibility = 'shared' AND EXISTS (
                  SELECT 1 FROM document_share s
                  WHERE s.document_id = document.id
                    AND s.user_id = current_setting('app.user_id', true)
                ))
              )
            )
            WITH CHECK (
              owner_user_id = current_setting('app.user_id', true)
              AND organization_id = current_setting('app.organization_id', true)
            )
            """
        )
        op.execute("DROP POLICY IF EXISTS ingestion_job_tenant_access ON ingestion_job")
        op.execute(
            """
            CREATE POLICY ingestion_job_tenant_access ON ingestion_job
            USING (
              requested_by_user_id = current_setting('app.user_id', true)
              OR EXISTS (
                SELECT 1 FROM document d
                WHERE d.id = ingestion_job.document_id
              )
            )
            """
        )
        op.execute("DROP POLICY IF EXISTS document_chunk_tenant_access ON document_chunk")
        op.execute(
            """
            CREATE POLICY document_chunk_tenant_access ON document_chunk
            USING (
              owner_user_id = current_setting('app.user_id', true)
              OR EXISTS (
                SELECT 1 FROM document_share s
                WHERE s.document_id = document_chunk.document_id
                  AND s.user_id = current_setting('app.user_id', true)
              )
            )
            """
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP POLICY IF EXISTS document_access ON document")
        op.execute(
            """
            CREATE POLICY document_access ON document
            USING (
              owner_user_id = current_setting('app.user_id', true)
              OR (organization_id = current_setting('app.organization_id', true) AND visibility = 'organization')
              OR (visibility = 'shared' AND EXISTS (
                SELECT 1 FROM document_share s
                WHERE s.document_id = document.id
                  AND s.user_id = current_setting('app.user_id', true)
              ))
            )
            WITH CHECK (
              owner_user_id = current_setting('app.user_id', true)
              AND organization_id = current_setting('app.organization_id', true)
            )
            """
        )
        op.execute("DROP POLICY IF EXISTS ingestion_job_tenant_access ON ingestion_job")
        op.execute(
            """
            CREATE POLICY ingestion_job_tenant_access ON ingestion_job
            USING (
              requested_by_user_id = current_setting('app.user_id', true)
              OR organization_id = current_setting('app.organization_id', true)
            )
            """
        )
        op.execute("DROP POLICY IF EXISTS document_chunk_tenant_access ON document_chunk")
        op.execute(
            """
            CREATE POLICY document_chunk_tenant_access ON document_chunk
            USING (
              owner_user_id = current_setting('app.user_id', true)
              OR organization_id = current_setting('app.organization_id', true)
            )
            """
        )
    op.drop_constraint("ck_ingestion_job_status", "ingestion_job", type_="check")
    op.execute("UPDATE ingestion_job SET status='pending' WHERE status='queued'")
    op.execute("UPDATE ingestion_job SET status='running' WHERE status='processing'")
    op.create_check_constraint(
        "ck_ingestion_job_status",
        "ingestion_job",
        "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
    )
    op.alter_column("ingestion_job", "status", server_default="pending")
    op.drop_constraint("ck_document_lifecycle_state", "document", type_="check")
    op.drop_column("document", "deleted_at")
    op.drop_column("document", "lifecycle_state")
    op.drop_column("document", "byte_size")
    op.drop_column("document", "content_type")
    op.drop_column("document", "source_filename")

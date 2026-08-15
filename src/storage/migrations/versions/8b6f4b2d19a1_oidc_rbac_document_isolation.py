"""OIDC identities, document authorization, and RLS

Revision ID: 8b6f4b2d19a1
Revises: cbb0182b0217
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8b6f4b2d19a1"
down_revision: str | None = "cbb0182b0217"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "organization",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "app_user",
        sa.Column("subject", sa.String(255), primary_key=True),
        sa.Column(
            "organization_id",
            sa.String(255),
            sa.ForeignKey("organization.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_app_user_organization_id", "app_user", ["organization_id"])
    op.create_table(
        "document",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "organization_id",
            sa.String(255),
            sa.ForeignKey("organization.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "owner_user_id",
            sa.String(255),
            sa.ForeignKey("app_user.subject", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("visibility", sa.String(16), nullable=False, server_default="private"),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'organization', 'shared')", name="ck_document_visibility"
        ),
    )
    op.create_index("ix_document_organization_id", "document", ["organization_id"])
    op.create_index("ix_document_owner_user_id", "document", ["owner_user_id"])
    op.create_table(
        "document_share",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(255),
            sa.ForeignKey("app_user.subject", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_by_user_id",
            sa.String(255),
            sa.ForeignKey("app_user.subject", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("document_id", "user_id", name="uq_document_share_user"),
    )
    op.create_index("ix_document_share_document_id", "document_share", ["document_id"])
    op.create_index("ix_document_share_user_id", "document_share", ["user_id"])
    op.create_table(
        "ingestion_job",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(255), nullable=False),
        sa.Column("requested_by_user_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_ingestion_job_status",
        ),
    )
    op.create_index("ix_ingestion_job_document_id", "ingestion_job", ["document_id"])
    op.create_index("ix_ingestion_job_organization_id", "ingestion_job", ["organization_id"])
    op.create_index(
        "ix_ingestion_job_requested_by_user_id", "ingestion_job", ["requested_by_user_id"]
    )
    op.create_table(
        "document_chunk",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(255), nullable=False),
        sa.Column("owner_user_id", sa.String(255), nullable=False),
        sa.Column("qdrant_point_id", sa.String(36), nullable=False, unique=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )
    for table in ("trace", "defect_event", "eval_score"):
        op.add_column(table, sa.Column("user_id", sa.String(255), nullable=True))
        op.add_column(table, sa.Column("organization_id", sa.String(255), nullable=True))
        op.create_index(f"ix_{table}_user_id", table, ["user_id"])
        op.create_index(f"ix_{table}_organization_id", table, ["organization_id"])

    # NULL identity deliberately quarantines legacy rows. Application transactions
    # must SET LOCAL app.user_id/app.organization_id after authenticating.
    if op.get_bind().dialect.name == "postgresql":
        for table in (
            "document",
            "document_share",
            "ingestion_job",
            "document_chunk",
            "trace",
            "defect_event",
            "eval_score",
        ):
            op.execute(sa.text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))
            op.execute(sa.text(f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY'))
        op.execute(
            sa.text("""
            CREATE POLICY document_access ON document
            USING (
              owner_user_id = current_setting('app.user_id', true)
              OR (organization_id = current_setting('app.organization_id', true) AND visibility = 'organization')
              OR (visibility = 'shared' AND EXISTS (
                SELECT 1 FROM document_share s
                WHERE s.document_id = document.id AND s.user_id = current_setting('app.user_id', true)
              ))
            )
            WITH CHECK (
              owner_user_id = current_setting('app.user_id', true)
              AND organization_id = current_setting('app.organization_id', true)
            )
        """)
        )
        for table in ("ingestion_job", "document_chunk", "trace", "defect_event", "eval_score"):
            op.execute(
                sa.text(
                    f"""
                CREATE POLICY {table}_tenant_access ON {table}
                USING (
                  user_id = current_setting('app.user_id', true)
                  OR organization_id = current_setting('app.organization_id', true)
                )
            """.replace(
                        "user_id",
                        "owner_user_id"
                        if table == "document_chunk"
                        else "requested_by_user_id"
                        if table == "ingestion_job"
                        else "user_id",
                    )
                )
            )
        op.execute(
            sa.text("""
            CREATE POLICY document_share_access ON document_share
            USING (user_id = current_setting('app.user_id', true) OR created_by_user_id = current_setting('app.user_id', true))
        """)
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(sa.text("DROP POLICY IF EXISTS document_access ON document"))
        op.execute(sa.text("DROP POLICY IF EXISTS document_share_access ON document_share"))
        for table in ("ingestion_job", "document_chunk", "trace", "defect_event", "eval_score"):
            op.execute(sa.text(f"DROP POLICY IF EXISTS {table}_tenant_access ON {table}"))
    for table in ("trace", "defect_event", "eval_score"):
        op.drop_index(f"ix_{table}_organization_id", table_name=table)
        op.drop_index(f"ix_{table}_user_id", table_name=table)
        op.drop_column(table, "organization_id")
        op.drop_column(table, "user_id")
    op.drop_table("document_chunk")
    op.drop_table("ingestion_job")
    op.drop_table("document_share")
    op.drop_table("document")
    op.drop_table("app_user")
    op.drop_table("organization")

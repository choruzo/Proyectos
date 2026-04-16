"""Add ownership columns

Revision ID: 0002_ownership_columns
Revises: 0001_initial
Create Date: 2026-04-16

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_ownership_columns"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # projects
    op.add_column("projects", sa.Column("owner_user_id", sa.String(length=36), nullable=True))
    op.add_column("projects", sa.Column("created_by", sa.String(length=36), nullable=True))
    op.add_column("projects", sa.Column("updated_by", sa.String(length=36), nullable=True))

    op.create_foreign_key(
        "fk_projects_owner_user_id_users",
        "projects",
        "users",
        ["owner_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_projects_created_by_users",
        "projects",
        "users",
        ["created_by"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_projects_updated_by_users",
        "projects",
        "users",
        ["updated_by"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_index(op.f("ix_projects_owner_user_id"), "projects", ["owner_user_id"], unique=False)
    op.create_index(op.f("ix_projects_created_by"), "projects", ["created_by"], unique=False)
    op.create_index(op.f("ix_projects_updated_by"), "projects", ["updated_by"], unique=False)

    # tasks
    op.add_column("tasks", sa.Column("created_by", sa.String(length=36), nullable=True))
    op.add_column("tasks", sa.Column("updated_by", sa.String(length=36), nullable=True))

    op.create_foreign_key(
        "fk_tasks_created_by_users",
        "tasks",
        "users",
        ["created_by"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_tasks_updated_by_users",
        "tasks",
        "users",
        ["updated_by"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_index(op.f("ix_tasks_created_by"), "tasks", ["created_by"], unique=False)
    op.create_index(op.f("ix_tasks_updated_by"), "tasks", ["updated_by"], unique=False)

    # attachments
    op.add_column("attachments", sa.Column("created_by", sa.String(length=36), nullable=True))
    op.add_column("attachments", sa.Column("updated_by", sa.String(length=36), nullable=True))

    op.create_foreign_key(
        "fk_attachments_created_by_users",
        "attachments",
        "users",
        ["created_by"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_attachments_updated_by_users",
        "attachments",
        "users",
        ["updated_by"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_index(op.f("ix_attachments_created_by"), "attachments", ["created_by"], unique=False)
    op.create_index(op.f("ix_attachments_updated_by"), "attachments", ["updated_by"], unique=False)


def downgrade() -> None:
    # attachments
    op.drop_index(op.f("ix_attachments_updated_by"), table_name="attachments")
    op.drop_index(op.f("ix_attachments_created_by"), table_name="attachments")
    op.drop_constraint("fk_attachments_updated_by_users", "attachments", type_="foreignkey")
    op.drop_constraint("fk_attachments_created_by_users", "attachments", type_="foreignkey")
    op.drop_column("attachments", "updated_by")
    op.drop_column("attachments", "created_by")

    # tasks
    op.drop_index(op.f("ix_tasks_updated_by"), table_name="tasks")
    op.drop_index(op.f("ix_tasks_created_by"), table_name="tasks")
    op.drop_constraint("fk_tasks_updated_by_users", "tasks", type_="foreignkey")
    op.drop_constraint("fk_tasks_created_by_users", "tasks", type_="foreignkey")
    op.drop_column("tasks", "updated_by")
    op.drop_column("tasks", "created_by")

    # projects
    op.drop_index(op.f("ix_projects_updated_by"), table_name="projects")
    op.drop_index(op.f("ix_projects_created_by"), table_name="projects")
    op.drop_index(op.f("ix_projects_owner_user_id"), table_name="projects")
    op.drop_constraint("fk_projects_updated_by_users", "projects", type_="foreignkey")
    op.drop_constraint("fk_projects_created_by_users", "projects", type_="foreignkey")
    op.drop_constraint("fk_projects_owner_user_id_users", "projects", type_="foreignkey")
    op.drop_column("projects", "updated_by")
    op.drop_column("projects", "created_by")
    op.drop_column("projects", "owner_user_id")

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.database import Base


class DatabaseHealth(Base):
    """Minimal database model used to verify ORM configuration."""

    __tablename__ = "database_health"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )


class Dataset(Base):
    """Persist metadata for an uploaded dataset."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    rows: Mapped[int] = mapped_column(
        nullable=False,
    )

    columns: Mapped[int] = mapped_column(
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    runs: Mapped[list["Run"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class Run(Base):
    """Persist metadata, lifecycle state, progress, and results."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
    )

    compute_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="local",
    )

    current_stage: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="queued",
    )

    progress: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
    )

    stage_message: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="Waiting for the ML worker.",
    )

    target_column: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    test_size: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.2,
    )

    random_state: Mapped[int] = mapped_column(
        nullable=False,
        default=42,
    )

    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    dataset: Mapped[Dataset] = relationship(
        back_populates="runs",
    )
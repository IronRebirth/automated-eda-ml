from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

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
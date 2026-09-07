from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


class DatabaseHealth(Base):
    """Minimal database model used to verify ORM configuration."""

    __tablename__ = "database_health"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )
from sqlalchemy import inspect

from backend.app.db.database import Base, SessionLocal, engine, init_db
from backend.app.db.models import DatabaseHealth


def test_database_engine_is_configured():
    """Verify that the SQLAlchemy engine is configured."""

    assert engine is not None
    assert engine.url.get_backend_name() == "postgresql"


def test_database_base_contains_health_model():
    """Verify that the health model is registered with the ORM base."""

    assert DatabaseHealth.__tablename__ == "database_health"
    assert DatabaseHealth.__table__ in Base.metadata.tables.values()


def test_database_initialization_creates_tables():
    """Verify that database initialization creates registered tables."""

    init_db()

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "database_health" in tables


def test_database_session_can_execute_query():
    """Verify that a SQLAlchemy session can communicate with PostgreSQL."""

    db = SessionLocal()

    try:
        result = db.execute(
            DatabaseHealth.__table__.select()
        )

        assert result is not None
    finally:
        db.close()
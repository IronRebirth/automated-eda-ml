from sqlalchemy import inspect, text

from backend.app.db.database import engine


def migrate() -> None:
    """Add the started_at column to the runs table when needed."""

    inspector = inspect(engine)

    if "runs" not in inspector.get_table_names():
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("runs")
    }

    if "started_at" in columns:
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE runs "
                "ADD COLUMN started_at TIMESTAMP NULL"
            )
        )


if __name__ == "__main__":
    migrate()
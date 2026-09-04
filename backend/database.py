"""
database.py
------------
Sets up the SQLAlchemy plumbing: the engine (knows how to reach MySQL),
a session factory (opens a "conversation" with the DB per request), and
the declarative Base that ORM models inherit from.

Nothing PDF/Gemini-specific lives here — this file only knows about "how
do I talk to the database", same separation-of-concerns idea as pdf_service.py.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

try:
    from backend.config import DATABASE_URL
except ImportError:
    from config import DATABASE_URL

# The engine manages the actual connection pool to MySQL.
# pool_pre_ping=True checks a connection is still alive before using it —
# avoids "MySQL server has gone away" errors after idle periods.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# SessionLocal is a factory: calling SessionLocal() gives you a new session.
# autocommit=False / autoflush=False are SQLAlchemy 2.0's sane defaults —
# you explicitly call .commit() when you want changes saved.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# All ORM model classes (see models.py) inherit from this Base.
# It's what lets SQLAlchemy know "this Python class maps to a SQL table".
Base = declarative_base()


def get_db():
    """
    FastAPI dependency: opens a session for the duration of one request,
    and guarantees it's closed afterward (even if an error occurs).

    Used in endpoints like: def upload_pdf(db: Session = Depends(get_db))
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Creates tables if they don't exist, and ensures all required columns exist
    in existing tables (automatic self-healing migration).
    """
    Base.metadata.create_all(bind=engine)

    try:
        from sqlalchemy import inspect, text
        with engine.begin() as conn:
            inspector = inspect(engine)
            if "documents" in inspector.get_table_names():
                columns = {col["name"] for col in inspector.get_columns("documents")}
                if "gemini_cache_name" not in columns:
                    conn.execute(text("ALTER TABLE documents ADD COLUMN gemini_cache_name VARCHAR(255) NULL;"))
                if "gemini_cache_expires_at" not in columns:
                    conn.execute(text("ALTER TABLE documents ADD COLUMN gemini_cache_expires_at DATETIME NULL;"))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Schema verification/migration note: %s", exc)

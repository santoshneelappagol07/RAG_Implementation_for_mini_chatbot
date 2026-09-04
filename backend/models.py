"""
models.py
----------
Defines the `documents` table as a Python class. This is the ORM part:
Document.filename maps to a `filename` column, Document.extracted_text
maps to an `extracted_text` column, etc. SQLAlchemy translates
`db.add(doc)` / `db.query(Document)` into actual SQL for you.
"""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.sql import func

try:
    from backend.database import Base
except ImportError:
    from database import Base


class Document(Base):
    __tablename__ = "documents"

    # Auto-incrementing primary key — this becomes the document_id
    # your API returns and expects on /ask, replacing the old
    # "filename as document_id" approach.
    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String(255), nullable=False)

    # Where the actual PDF bytes live on disk (see config.STORAGE_DIR).
    file_path = Column(String(500), nullable=False)

    # LONGTEXT: handles long PDF text extractions without hitting MySQL's 64KB TEXT limit.
    extracted_text = Column(LONGTEXT, nullable=False)

    created_at = Column(DateTime, server_default=func.now(), default=func.now())


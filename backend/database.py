"""
database.py
------------
Sets up the MongoDB Atlas connection plumbing using PyMongo:
- get_mongo_client(): manages connection pooling to MongoDB Atlas
- get_db(): FastAPI dependency that yields the MongoDB database per request
- init_db(): validates connection and ensures indexes on collections
"""

import logging
from typing import Generator
from pymongo import MongoClient
from pymongo.database import Database

try:
    from backend.config import MONGODB_URI, MONGODB_DB
except ImportError:
    from config import MONGODB_URI, MONGODB_DB

logger = logging.getLogger(__name__)

# Singleton MongoDB client instance
_client: MongoClient = None


def get_mongo_client() -> MongoClient:
    """
    Returns a singleton MongoClient configured for MongoDB Atlas or local MongoDB.
    """
    global _client
    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
    return _client


def get_db() -> Generator[Database, None, None]:
    """
    FastAPI dependency: yields the MongoDB Database instance for a request.

    Used in endpoints like: def upload_pdf(db: Database = Depends(get_db))
    """
    client = get_mongo_client()
    db = client[MONGODB_DB]
    yield db


def init_db() -> None:
    """
    Validates connection to MongoDB Atlas and ensures collection indexes exist.
    """
    try:
        client = get_mongo_client()
        # Verify connection to Atlas
        client.admin.command("ping")
        db = client[MONGODB_DB]

        # Ensure indexes on documents collection for fast queries
        db.documents.create_index("filename")
        db.documents.create_index("created_at")

        logger.info(" Connected successfully to MongoDB Atlas (Database: %s)", MONGODB_DB)
    except Exception as exc:
        logger.warning(" MongoDB Atlas connection check/index setup note: %s", exc)

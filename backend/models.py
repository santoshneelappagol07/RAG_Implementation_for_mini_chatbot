"""
models.py
----------
Defines MongoDB Atlas Document model and operations for PDF records.
Replaces previous SQL/MySQL tables with native MongoDB BSON documents.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from bson import ObjectId
from pymongo.database import Database


class Document:
    """
    Represents a stored PDF document record in MongoDB Atlas `documents` collection.
    """

    def __init__(
        self,
        filename: str,
        file_path: str,
        extracted_text: str,
        gemini_cache_name: Optional[str] = None,
        gemini_cache_expires_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        _id: Optional[Union[ObjectId, str]] = None,
        id: Optional[str] = None,
    ):
        self._id = _id
        self.id = str(id or _id or "")
        self.filename = filename
        self.file_path = file_path
        self.extracted_text = extracted_text
        self.gemini_cache_name = gemini_cache_name
        self.gemini_cache_expires_at = gemini_cache_expires_at
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Converts model instance to MongoDB document dict."""
        data: Dict[str, Any] = {
            "filename": self.filename,
            "file_path": self.file_path,
            "extracted_text": self.extracted_text,
            "gemini_cache_name": self.gemini_cache_name,
            "gemini_cache_expires_at": self.gemini_cache_expires_at,
            "created_at": self.created_at,
        }
        if self._id is not None:
            data["_id"] = self._id
        return data

    @classmethod
    def from_mongo(cls, doc: Dict[str, Any]) -> "Document":
        """Builds a Document instance from MongoDB document."""
        _id = doc.get("_id")
        return cls(
            _id=_id,
            id=str(_id) if _id is not None else None,
            filename=doc.get("filename", ""),
            file_path=doc.get("file_path", ""),
            extracted_text=doc.get("extracted_text", ""),
            gemini_cache_name=doc.get("gemini_cache_name"),
            gemini_cache_expires_at=doc.get("gemini_cache_expires_at"),
            created_at=doc.get("created_at"),
        )

    @classmethod
    def insert(cls, db: Database, doc: "Document") -> "Document":
        """
        Inserts document into MongoDB Atlas `documents` collection and returns updated model.
        """
        data = doc.to_dict()
        result = db.documents.insert_one(data)
        doc._id = result.inserted_id
        doc.id = str(result.inserted_id)
        return doc

    @classmethod
    def find_by_id(
        cls, db: Database, document_id: Union[str, int, ObjectId]
    ) -> Optional["Document"]:
        """
        Finds a document record by MongoDB _id (accepts hex string ObjectId, ObjectId, or int/str).
        """
        doc = None
        # Check if valid ObjectId hex string
        if isinstance(document_id, str) and ObjectId.is_valid(document_id):
            doc = db.documents.find_one({"_id": ObjectId(document_id)})

        # Check raw _id matching
        if doc is None:
            doc = db.documents.find_one({"_id": document_id})

        # Check custom or legacy 'id' field if present
        if doc is None and isinstance(document_id, (int, str)):
            doc = db.documents.find_one({"id": document_id})

        if doc:
            return cls.from_mongo(doc)
        return None

    def update_cache(
        self,
        db: Database,
        gemini_cache_name: Optional[str],
        gemini_cache_expires_at: Optional[datetime] = None,
    ) -> None:
        """
        Updates Gemini context cache fields in MongoDB Atlas.
        """
        self.gemini_cache_name = gemini_cache_name
        self.gemini_cache_expires_at = gemini_cache_expires_at

        query: Dict[str, Any]
        if self._id is not None:
            query = {"_id": self._id}
        elif ObjectId.is_valid(self.id):
            query = {"_id": ObjectId(self.id)}
        else:
            query = {"id": self.id}

        db.documents.update_one(
            query,
            {
                "$set": {
                    "gemini_cache_name": gemini_cache_name,
                    "gemini_cache_expires_at": gemini_cache_expires_at,
                }
            },
        )

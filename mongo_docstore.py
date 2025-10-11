from typing import List, Tuple, Optional, Sequence
from langchain_core.stores import BaseStore
from langchain.schema import Document
from pymongo.collection import Collection
from pymongo import UpdateOne

class MongoDocStore(BaseStore[str, Document]):
    """
    LangChain BaseStore implementation using MongoDB for document storage.

    This class allows the ParentDocumentRetriever to use MongoDB as its
    docstore for storing parent documents.
    """
    def __init__(self, collection: Collection):
        """Initialize with a pymongo collection."""
        self.collection = collection

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        """
        Get documents from MongoDB by their _id.
        """
        results = self.collection.find({"_id": {"$in": keys}})
        found_docs = {doc["_id"]: Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in results}
        return [found_docs.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """
        Set documents in MongoDB using their _id.
        """
        operations = []
        for key, doc in key_value_pairs:
            operations.append(
                UpdateOne(
                    {"_id": key},
                    {"$set": {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }},
                    upsert=True
                )
            )
        if operations:
            self.collection.bulk_write(operations)

    def mdelete(self, keys: List[str]) -> None:
        """Delete documents from MongoDB by their _id."""
        self.collection.delete_many({"_id": {"$in": keys}})

    def yield_keys(self, prefix: Optional[str] = None):
        """Yield all keys (not typically needed for this retriever)."""

        cursor = self.collection.find({}, {"_id": 1})
        for doc in cursor:
            yield doc["_id"]
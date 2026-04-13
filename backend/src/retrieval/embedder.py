"""
ChromaDB embedder — stores and retrieves training data:
  - add_ddl()           → ddl-collection
  - add_documentation() → documentation-collection
  - add_question_sql()  → sql-collection

Embeddings use sentence-transformers all-MiniLM-L6-v2.
"""

import hashlib
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from src.ingestion.loader import SchemaLoader


EMBED_MODEL = "all-MiniLM-L6-v2"
DDL_COLLECTION = "ddl-collection"
DOC_COLLECTION = "documentation-collection"
SQL_COLLECTION = "sql-collection"


class SchemaEmbedder:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Local sentence-transformers — zero API cost
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

        # Three separate collections
        self.ddl_col = self.client.get_or_create_collection(
            name=DDL_COLLECTION, embedding_function=self.ef
        )
        self.doc_col = self.client.get_or_create_collection(
            name=DOC_COLLECTION, embedding_function=self.ef
        )
        self.sql_col = self.client.get_or_create_collection(
            name=SQL_COLLECTION, embedding_function=self.ef
        )

    # ── Training methods (called once at startup) ──────────────────────────────

    def add_ddl(self, ddl: str) -> str:
        """
        Store a DDL statement in the DDL collection.
        Called once per table during startup.
        ID is a deterministic hash of the content — same DDL always
        gets the same ID, preventing duplicates on restart.
        """
        doc_id = self._make_id(ddl)
        self._upsert(self.ddl_col, doc_id, ddl)
        return doc_id

    def add_documentation(self, doc: str) -> str:
        """
        Store a documentation string in the documentation collection.
        Called for:
          - Relationship descriptions (from loader.py)
          - Service-level descriptions (from loader.py)
          - User-confirmed metric definitions (from metric_dict.py)
        """
        doc_id = self._make_id(doc)
        self._upsert(self.doc_col, doc_id, doc)
        return doc_id

    def add_question_sql(self, question: str, sql: str) -> str:
        """
        Store a Q-SQL pair in the SQL collection.
        Called for:
          - Seeded example pairs at startup
          - User-corrected pairs from the fix loop (Stage 5)
        The question is what gets embedded and searched.
        The SQL is stored as metadata.
        """
        doc_id = self._make_id(question)
        self._upsert(self.sql_col, doc_id, question, metadata={"sql": sql})
        return doc_id

    # ── Retrieval methods (called on every user question) ─────────────────────

    def get_related_ddl(self, question: str, n_results: int = 6) -> List[str]:
        """
        Find DDL statements most relevant to the question.
        Returns a list of DDL strings — the schema context for the LLM.
        """
        return self._query(self.ddl_col, question, n_results)

    def get_related_documentation(self, question: str, n_results: int = 5) -> List[str]:
        """
        Find documentation most relevant to the question.
        Returns relationship descriptions + matching metric definitions.
        """
        return self._query(self.doc_col, question, n_results)

    def get_similar_question_sql(self, question: str, n_results: int = 3) -> List[Dict]:
        """
        Find past Q-SQL pairs most similar to the current question.
        Returns list of {"question": ..., "sql": ...} dicts.
        These become the few-shot examples in the LLM prompt.
        """
        count = self.sql_col.count()
        if count == 0:
            return []

        results = self.sql_col.query(
            query_texts=[question], n_results=min(n_results, count)
        )

        pairs = []
        if results["ids"] and results["ids"][0]:
            for i, _ in enumerate(results["ids"][0]):
                pairs.append(
                    {
                        "question": results["documents"][0][i],
                        "sql": results["metadatas"][0][i].get("sql", ""),
                    }
                )
        return pairs

    # ── Bulk loading (called at startup from app.py) ───────────────────────────

    def load_from_schema(self, loader: SchemaLoader):
        """
        Bulk load DDL and documentation from a SchemaLoader instance.
        This is called once when the app starts.
        Clears existing data first to allow clean restarts.
        """
        self._clear_collection(self.ddl_col)
        self._clear_collection(self.doc_col)

        # Load DDL — one statement per table
        ddl_statements = loader.to_ddl_statements()
        for ddl in ddl_statements:
            self.add_ddl(ddl)

        # Load documentation — relationships + service descriptions
        doc_strings = loader.to_documentation_strings()
        for doc in doc_strings:
            self.add_documentation(doc)

        return {"ddl_loaded": len(ddl_statements), "docs_loaded": len(doc_strings)}

    def seed_qa_pairs(self, pairs: List[Dict]):
        """
        Seed the SQL collection with example Q-SQL pairs.
        Called at startup so the system has few-shot examples
        from day one, before any user has asked anything.
        pairs: list of {"question": str, "sql": str}
        """
        self._clear_collection(self.sql_col)
        for pair in pairs:
            self.add_question_sql(pair["question"], pair["sql"])
        return len(pairs)

    # ── Counts (for UI display) ────────────────────────────────────────────────

    def ddl_count(self) -> int:
        return self.ddl_col.count()

    def doc_count(self) -> int:
        return self.doc_col.count()

    def sql_count(self) -> int:
        return self.sql_col.count()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_id(self, text: str) -> str:
        """
        Deterministic ID from content hash.
        Same text always produces the same ID.
        Prevents duplicate entries on app restart.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _upsert(self, collection, doc_id: str, document: str, metadata: Dict = None):
        """
        Insert or update a document in a collection.
        Uses upsert pattern — safe to call multiple times with same content.
        """
        kwargs = {"ids": [doc_id], "documents": [document]}
        if metadata:
            kwargs["metadatas"] = [metadata]

        try:
            collection.upsert(**kwargs)
        except Exception as e:
            # Log but don't crash — missing one document is recoverable
            print(f"Warning: upsert failed for id {doc_id}: {e}")

    def _query(self, collection, question: str, n_results: int) -> List[str]:
        """
        Query a collection and return document strings.
        Returns empty list if collection is empty.
        """
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_texts=[question], n_results=min(n_results, count)
        )

        if results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

    def _clear_collection(self, collection):
        """Clear all documents from a collection before reloading."""
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

"""
Schema linker — combines ChromaDB vector retrieval with Neo4j graph traversal.

This is the core of the Graph-RAG architecture. It answers two questions:
  1. Which tables are relevant to this question? (ChromaDB semantic search)
  2. How do those tables connect to each other? (Neo4j shortest path)

The output is a SchemaContext object that the SQL generator consumes directly.

Design informed by:
  - Vanna's get_sql_prompt() which assembles ddl_list, doc_list, question_sql_list
  - LinkedIn SQL Bot's schema linking step (table pruning + join path discovery)
  - CHESS paper's schema linking: candidate generation → pruning → join path
  - Our validated Graph-RAG architecture decision (ChromaDB finds, Neo4j connects)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict

from src.retrieval.embedder import SchemaEmbedder
from src.retrieval.graph_builder import SchemaGraphBuilder

logger = logging.getLogger(__name__)


# ── Context object ─────────────────────────────────────────────────────────────


@dataclass
class SchemaContext:
    """
    Unified context produced by SchemaLinker for a single user question.
    Passed directly to sql_gen.py for prompt construction.

    Every field has a clear consumer:
      relevant_ddls  → LLM knows table structure and column names
      relevant_docs  → LLM knows business definitions and metric formulas
      similar_sql    → LLM has few-shot examples of correct SQL
      join_paths     → LLM knows exactly how to JOIN tables (no hallucination)
      join_context   → pre-formatted JOIN string ready to embed in prompt
      table_names    → list of all tables the LLM should consider
    """

    question: str
    relevant_ddls: List[str] = field(default_factory=list)
    relevant_docs: List[str] = field(default_factory=list)
    similar_sql: List[Dict] = field(default_factory=list)
    join_paths: List[Dict] = field(default_factory=list)
    join_context: str = ""
    table_names: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """True if no schema context was retrieved — signals retrieval failure."""
        return not self.relevant_ddls and not self.relevant_docs

    def summary(self) -> str:
        """Single-line summary for logging."""
        return (
            f"tables={len(self.table_names)}, "
            f"ddls={len(self.relevant_ddls)}, "
            f"docs={len(self.relevant_docs)}, "
            f"sql_examples={len(self.similar_sql)}, "
            f"join_hops={len(self.join_paths)}"
        )


# ── SchemaLinker ───────────────────────────────────────────────────────────────


class SchemaLinker:
    """
    Combines ChromaDB semantic retrieval with Neo4j graph traversal
    to produce a SchemaContext for a given user question.

    ChromaDB answers: which tables are semantically relevant?
    Neo4j answers:    how do those tables connect via foreign keys?

    Neither is sufficient alone:
      - ChromaDB misses intermediate join tables not mentioned in the question
      - Neo4j cannot score semantic relevance — it needs a starting set of tables
    Together they produce complete, join-path-aware schema context.
    """

    def __init__(
        self,
        embedder: SchemaEmbedder,
        graph: SchemaGraphBuilder,
        n_ddl: int = 6,
        n_docs: int = 5,
        n_sql: int = 3,
        n_hops: int = 1,
    ):
        """
        Args:
            embedder: initialised SchemaEmbedder with populated ChromaDB
            graph:    initialised SchemaGraphBuilder with connected Neo4j driver
            n_ddl:    number of DDL statements to retrieve per question
            n_docs:   number of documentation strings to retrieve per question
            n_sql:    number of similar Q-SQL pairs to retrieve per question
            n_hops:   neighbourhood expansion depth in Neo4j graph
        """
        self.embedder = embedder
        self.graph = graph
        self.n_ddl = n_ddl
        self.n_docs = n_docs
        self.n_sql = n_sql
        self.n_hops = n_hops

    # ── Main entry point ───────────────────────────────────────────────────────

    def link(self, question: str) -> SchemaContext:
        """
        Full schema linking pipeline for a user question.
        Returns a populated SchemaContext ready for SQL generation.

        Steps:
          1. Retrieve relevant DDLs from ChromaDB
          2. Extract table names from those DDLs
          3. Expand table set via Neo4j neighbourhood traversal
          4. Fetch DDLs for any newly added tables
          5. Retrieve relevant documentation from ChromaDB
          6. Retrieve similar Q-SQL pairs from ChromaDB
          7. Find join paths between all relevant tables
          8. Format join context string for LLM prompt
          9. Return SchemaContext
        """
        logger.info(f"SchemaLinker: linking question: '{question[:80]}'")

        # Step 1 — ChromaDB: semantic DDL retrieval
        raw_ddls = self.embedder.get_related_ddl(question, n_results=self.n_ddl)
        logger.debug(f"ChromaDB returned {len(raw_ddls)} DDLs")

        # Step 2 — Extract short table names from DDL comment headers
        chroma_tables = self._extract_table_names(raw_ddls)
        logger.debug(f"Tables from ChromaDB: {chroma_tables}")

        # Step 3 — Neo4j: expand table set with 1-hop neighbours
        # This catches intermediate join tables (e.g. Order between Account/Trade)
        # that are structurally necessary but not mentioned in the question
        if chroma_tables:
            neighbours = self.graph.get_neighborhood(chroma_tables, hops=self.n_hops)
            all_tables = list(dict.fromkeys(chroma_tables + neighbours))
        else:
            all_tables = chroma_tables
        logger.debug(f"Tables after graph expansion: {all_tables}")

        # Step 4 — Fetch DDLs for any newly added tables from neighbourhood
        expanded_ddls = self._fetch_ddls_for_tables(raw_ddls, all_tables)

        # Step 5 — ChromaDB: retrieve relevant documentation
        # Covers relationship descriptions AND user-confirmed metric definitions
        docs = self.embedder.get_related_documentation(question, n_results=self.n_docs)

        # Step 6 — ChromaDB: retrieve similar past Q-SQL pairs (few-shot)
        similar_sql = self.embedder.get_similar_question_sql(
            question, n_results=self.n_sql
        )

        # Step 7 — Neo4j: find join paths between all relevant table pairs
        join_paths = self._find_all_join_paths(all_tables)

        # Step 8 — Format join conditions into a single string for LLM prompt
        join_context = self._format_join_context(join_paths)

        ctx = SchemaContext(
            question=question,
            relevant_ddls=expanded_ddls,
            relevant_docs=docs,
            similar_sql=similar_sql,
            join_paths=join_paths,
            join_context=join_context,
            table_names=all_tables,
        )

        logger.info(f"SchemaLinker result: {ctx.summary()}")
        return ctx

    # ── Step 2: Extract table names from DDL strings ───────────────────────────

    def _extract_table_names(self, ddl_list: List[str]) -> List[str]:
        """
        Parse short table names from DDL comment headers.

        Each DDL string starts with a comment line of the form:
          -- CustSrv.CustomerDB.Customer: Master customer profile
          CREATE TABLE Customer ( ...

        We extract the short name from the CREATE TABLE line — this is
        what Neo4j stores as node.name and what SQL uses.
        """
        names = []
        for ddl in ddl_list:
            # Match: CREATE TABLE <TableName> (
            match = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\(", ddl, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name not in names:
                    names.append(name)
        return names

    # ── Step 4: Fetch DDLs for expanded tables ─────────────────────────────────

    def _fetch_ddls_for_tables(
        self, existing_ddls: List[str], all_tables: List[str]
    ) -> List[str]:
        """
        Ensure every table in all_tables has a DDL in the final list.
        Tables added by Neo4j neighbourhood expansion may not have been
        returned by ChromaDB's semantic search. We add their DDLs by
        querying ChromaDB with the table name directly.
        """
        existing_names = self._extract_table_names(existing_ddls)
        result_ddls = list(existing_ddls)

        for table_name in all_tables:
            if table_name not in existing_names:
                # Query ChromaDB with the table name as the search term
                extra = self.embedder.get_related_ddl(table_name, n_results=1)
                for ddl in extra:
                    # Verify it actually matches the table we want
                    if re.search(
                        rf"CREATE\s+TABLE\s+{re.escape(table_name)}\s*\(",
                        ddl,
                        re.IGNORECASE,
                    ):
                        result_ddls.append(ddl)
                        logger.debug(f"Added expanded DDL for table: {table_name}")
                        break

        return result_ddls

    # ── Step 7: Find join paths ────────────────────────────────────────────────

    def _find_all_join_paths(self, table_names: List[str]) -> List[Dict]:
        """
        Find Neo4j shortest paths between all pairs of relevant tables.
        Deduplicates paths so the same hop does not appear twice.

        For N tables, we check N*(N-1)/2 pairs.
        For our typical retrieval of 6-8 tables this is 15-28 pairs —
        fast enough for real-time use.

        Each returned dict has:
          from_table, to_table, relation, join_condition, description
        """
        seen_conditions = set()
        all_hops = []

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                src = table_names[i]
                tgt = table_names[j]

                try:
                    path = self.graph.find_join_path(src, tgt)
                    if not path:
                        continue

                    for hop in path:
                        jc = hop.get("join_condition", "")
                        if jc and jc not in seen_conditions:
                            seen_conditions.add(jc)
                            all_hops.append(hop)

                except Exception as e:
                    logger.warning(f"Join path query failed for {src}->{tgt}: {e}")
                    continue

        logger.debug(f"Found {len(all_hops)} unique join conditions")
        return all_hops

    # ── Step 8: Format join context for LLM ───────────────────────────────────

    def _format_join_context(self, join_paths: List[Dict]) -> str:
        """
        Format join paths into a clean string the LLM can read directly.

        Output format:
          JOIN CONDITIONS:
          - Account.customer_id = Customer.customer_id  (HAS_ACCOUNT: Customer owns account)
          - Order.account_id = Account.account_id       (HAS_ORDER: Account creates orders)
          - Trade.order_id = Order.order_id             (EXECUTES_TO: Order executes to trade)

        This string is injected into the SQL generation prompt verbatim.
        The LLM uses it to write correct JOIN clauses without hallucinating
        column names or join conditions.
        """
        if not join_paths:
            return ""

        lines = ["JOIN CONDITIONS:"]
        for hop in join_paths:
            jc = hop.get("join_condition", "")
            rel = hop.get("relation", "")
            desc = hop.get("description", "")
            if jc:
                lines.append(f"  - {jc}  ({rel}: {desc})")

        return "\n".join(lines)

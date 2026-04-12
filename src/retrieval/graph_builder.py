"""
Neo4j schema graph builder for banking microservice schema.

Populates AuraDB with:
  - Table nodes (one per table, label :Table)
  - Relationship edges (directed, typed by relation name)
  - Join conditions derived from FK data in tables file

Design decisions informed by:
  - LinkedIn SQL Bot knowledge graph architecture (tables, fields, relationships)
  - Neo4j Python driver best practices (execute_write, single session per operation)
  - Our validated JSONL files (25 tables, 25 relationships, all FKs confirmed)

"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.ingestion.loader import SchemaLoader

load_dotenv()

logger = logging.getLogger(__name__)


class SchemaGraphBuilder:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self):
        """
        Establish connection to Neo4j AuraDB.
        Called once at startup. Driver is reused for all operations.
        Raises clearly if credentials are missing or connection fails.
        """
        if not all([self.uri, self.username, self.password]):
            raise ValueError(
                "Missing Neo4j credentials. "
                "Ensure NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD are in .env"
            )
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            logger.info(f"Neo4j connected: {self.uri}")
        except AuthError as e:
            raise ConnectionError(f"Neo4j authentication failed: {e}")
        except ServiceUnavailable as e:
            raise ConnectionError(f"Neo4j unreachable: {e}")

    def close(self):
        """Close driver. Always call on shutdown."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    # ── Build graph ────────────────────────────────────────────────────────────

    def build_from_loader(self, loader: SchemaLoader) -> dict:
        """
        Main entry point. Clears existing data and rebuilds the full graph.
        Returns stats dict with node and edge counts.

        Order:
          1. Create uniqueness constraint (idempotent)
          2. Clear all existing nodes and edges
          3. Create 25 Table nodes
          4. Create 25 relationship edges with join conditions
          5. Verify and return stats
        """
        self._create_constraint()
        self._clear_graph()
        self._create_table_nodes(loader)
        self._create_relationship_edges(loader)
        stats = self._verify(loader)
        logger.info(f"Graph built: {stats['nodes']} nodes, {stats['edges']} edges")
        return stats

    # ── Step 1: Constraint ─────────────────────────────────────────────────────

    def _create_constraint(self):
        """
        Create uniqueness constraint on Table.full_name.
        Prevents duplicate nodes on restart.
        Also creates an index automatically — fast lookups for MATCH queries.
        IF NOT EXISTS makes this idempotent — safe to call every startup.
        """
        with self.driver.session() as session:
            session.execute_write(self._constraint_tx)
        logger.info("Constraint on Table.full_name: ready")

    @staticmethod
    def _constraint_tx(tx):
        tx.run("""
            CREATE CONSTRAINT table_full_name_unique IF NOT EXISTS
            FOR (t:Table) REQUIRE t.full_name IS UNIQUE
        """)

    # ── Step 2: Clear ──────────────────────────────────────────────────────────

    def _clear_graph(self):
        """
        Delete all nodes and relationships.
        DETACH DELETE removes edges before nodes — avoids constraint errors.
        Called every startup so graph always reflects current JSONL files exactly.
        """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        logger.info("Existing graph cleared")

    # ── Step 3: Table nodes ────────────────────────────────────────────────────

    def _create_table_nodes(self, loader: SchemaLoader) -> int:
        """
        Create one :Table node per table in schema.
        Properties:
          full_name   — unique key e.g. CustSrv.CustomerDB.Customer
          name        — short name used in SQL e.g. Customer
          service     — microservice name e.g. CustSrv
          db          — database name e.g. CustomerDB
          primary_key — PK column name
          columns     — comma-joined column names
          col_types   — pipe-joined col:TYPE pairs
          description — plain English description
        """
        count = 0
        with self.driver.session() as session:
            for full_name, record in loader.tables.items():
                parts = full_name.split(".")
                service = parts[0] if len(parts) > 0 else ""
                db = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else parts[-1]

                cols = record.get("columns", {})
                col_list = ",".join(cols.keys())
                col_types = "|".join(f"{k}:{v}" for k, v in cols.items())

                session.execute_write(
                    self._create_table_tx,
                    full_name=full_name,
                    name=name,
                    service=service,
                    db=db,
                    primary_key=record.get("primary_key", ""),
                    columns=col_list,
                    col_types=col_types,
                    description=record.get("description", ""),
                )
                count += 1

        logger.info(f"Table nodes created: {count}")
        return count

    @staticmethod
    def _create_table_tx(
        tx, full_name, name, service, db, primary_key, columns, col_types, description
    ):
        """MERGE on full_name prevents duplicates.
        SET overwrites properties on every run — keeps data fresh."""
        tx.run(
            """
            MERGE (t:Table {full_name: $full_name})
            SET t.name        = $name,
                t.service     = $service,
                t.db          = $db,
                t.primary_key = $primary_key,
                t.columns     = $columns,
                t.col_types   = $col_types,
                t.description = $description
        """,
            full_name=full_name,
            name=name,
            service=service,
            db=db,
            primary_key=primary_key,
            columns=columns,
            col_types=col_types,
            description=description,
        )

    # ── Step 4: Relationship edges ─────────────────────────────────────────────

    def _create_relationship_edges(self, loader: SchemaLoader) -> int:
        """
        Create directed relationship edges between Table nodes.
        Edge label = relation name from relationships file (e.g. HAS_ACCOUNT).
        Edge properties:
          - description: plain English
          - source_col: FK column on source table
          - target_col: referenced column on target table
          - join_condition: "SourceTable.col = TargetTable.col" — used directly
                            by the SQL generator to build JOIN clauses

        Join condition is derived from FK data in the tables file.
        We check both directions (source FK to target, target FK to source).
        This is guaranteed to succeed for all 25 relationships because we
        validated every one has a backing FK during data preparation.
        """
        count = 0
        with self.driver.session() as session:
            for rel in loader.relations:
                source_full = rel["source"]
                target_full = rel["target"]
                relation = rel["relation"]
                description = rel.get("description", "")
                source_short = source_full.split(".")[-1]
                target_short = target_full.split(".")[-1]

                # Derive join condition from FK data
                source_col, target_col, join_condition = self._derive_join(
                    loader, source_full, target_full, source_short, target_short
                )

                session.execute_write(
                    self._create_edge_tx,
                    source_full=source_full,
                    target_full=target_full,
                    relation=relation,
                    description=description,
                    source_col=source_col,
                    target_col=target_col,
                    join_condition=join_condition,
                )
                count += 1
                logger.debug(
                    f"Edge: {source_short} -[{relation}]-> "
                    f"{target_short} | {join_condition}"
                )

        logger.info(f"Relationship edges created: {count}")
        return count

    def _derive_join(
        self,
        loader: SchemaLoader,
        source_full: str,
        target_full: str,
        source_short: str,
        target_short: str,
    ):
        """
        Derive the SQL join condition for an edge.

        Logic:
          Pass 1 — Does source table have an FK pointing to target?
                   e.g. Account.customer_id -> Customer.customer_id
          Pass 2 — Does target table have an FK pointing to source?
                   e.g. AccountBalance.account_id -> Account.account_id

        Returns (source_col, target_col, join_condition).
        """
        src_record = loader.tables.get(source_full, {})
        tgt_record = loader.tables.get(target_full, {})
        src_fks = src_record.get("foreign_keys", {})
        tgt_fks = tgt_record.get("foreign_keys", {})

        # Pass 1: source has FK to target
        for fk_col, ref in src_fks.items():
            ref_parts = ref.split(".")
            if len(ref_parts) >= 4 and ref_parts[2] == target_short:
                ref_col = ref_parts[3]
                join_cond = f"{source_short}.{fk_col} = {target_short}.{ref_col}"
                return fk_col, ref_col, join_cond

        # Pass 2: target has FK to source (reverse FK)
        for fk_col, ref in tgt_fks.items():
            ref_parts = ref.split(".")
            if len(ref_parts) >= 4 and ref_parts[2] == source_short:
                ref_col = ref_parts[3]
                join_cond = f"{target_short}.{fk_col} = {source_short}.{ref_col}"
                return ref_col, fk_col, join_cond

        # Fallback — should not reach here given validated data
        logger.warning(
            f"No FK found for {source_short} -> {target_short}. "
            f"Join condition left empty."
        )
        return "", "", ""

    @staticmethod
    def _create_edge_tx(
        tx,
        source_full,
        target_full,
        relation,
        description,
        source_col,
        target_col,
        join_condition,
    ):
        """
        MATCH both Table nodes by full_name (indexed — fast).
        MERGE the relationship to avoid duplicates on restart.
        SET overwrites properties to keep them current.

        Relationship type is injected via f-string — this is the standard
        Neo4j Python driver pattern for dynamic relationship types.
        Cypher does not support parameterised relationship labels.
        """
        tx.run(
            f"""
            MATCH (source:Table {{full_name: $source_full}})
            MATCH (target:Table {{full_name: $target_full}})
            MERGE (source)-[r:{relation}]->(target)
            SET r.description    = $description,
                r.source_col     = $source_col,
                r.target_col     = $target_col,
                r.join_condition = $join_condition
        """,
            source_full=source_full,
            target_full=target_full,
            description=description,
            source_col=source_col,
            target_col=target_col,
            join_condition=join_condition,
        )

    # ── Step 5: Verify ─────────────────────────────────────────────────────────

    def _verify(self, loader: SchemaLoader) -> dict:
        """Query the graph and return node/edge counts.
        Also checks that counts match expected values from loader."""
        with self.driver.session() as session:
            node_count = session.execute_read(
                lambda tx: tx.run("MATCH (t:Table) RETURN COUNT(t) AS c").single()["c"]
            )
            edge_count = session.execute_read(
                lambda tx: tx.run("MATCH ()-[r]->() RETURN COUNT(r) AS c").single()["c"]
            )

        if node_count != loader.table_count():
            logger.warning(
                f"Node mismatch: got {node_count}, expected {loader.table_count()}"
            )
        if edge_count != loader.relation_count():
            logger.warning(
                f"Edge mismatch: got {edge_count}, expected {loader.relation_count()}"
            )

        return {"nodes": node_count, "edges": edge_count}

    # ── Query methods (used by SchemaLinker at query time) ─────────────────────

    def find_join_path(self, source_short: str, target_short: str) -> Optional[list]:
        """
        Find shortest path between two tables in the graph.
        Returns list of dicts with hop details including join conditions.
        Used by SchemaLinker (Stage 1) to build JOIN clauses for the LLM.

        Uses Cypher shortestPath — handles multi-hop cross-service paths.
        Max depth 6 covers the longest realistic path in our schema
        (e.g. Batch -> AggregateOrder -> Order -> Account -> Customer).
        """
        with self.driver.session() as session:
            return session.execute_read(
                self._shortest_path_tx, source_short, target_short
            )

    @staticmethod
    def _shortest_path_tx(tx, source_short: str, target_short: str):
        """
        Returns each hop in the shortest path as a dict:
          - from_table, to_table: short names
          - relation: relationship label
          - join_condition: SQL join clause
          - description: business description
        """
        result = tx.run(
            """
            MATCH (start:Table {name: $source}),
                  (end:Table {name: $target})
            MATCH path = shortestPath((start)-[*1..6]-(end))
            RETURN path
            LIMIT 1
        """,
            source=source_short,
            target=target_short,
        )

        record = result.single()
        if not record:
            return None

        path = record["path"]
        nodes = list(path.nodes)
        rels = list(path.relationships)
        hops = []

        for i, rel in enumerate(rels):
            hops.append(
                {
                    "from_table": nodes[i]["name"],
                    "to_table": nodes[i + 1]["name"],
                    "relation": rel.type,
                    "join_condition": rel.get("join_condition", ""),
                    "description": rel.get("description", ""),
                }
            )

        return hops

    def get_neighborhood(self, table_names: list, hops: int = 1) -> list:
        """
        Get all tables within `hops` hops of the given starting tables.
        Used by SchemaLinker to expand ChromaDB vector results with
        structurally connected tables that semantic search may have missed.
        Returns list of short table names.
        """
        if not table_names:
            return []
        with self.driver.session() as session:
            return session.execute_read(self._neighborhood_tx, table_names, hops)

    @staticmethod
    def _neighborhood_tx(tx, table_names: list, hops: int):
        result = tx.run(
            f"""
            MATCH (start:Table)
            WHERE start.name IN $names
            MATCH (start)-[*1..{hops}]-(neighbor:Table)
            WHERE NOT neighbor.name IN $names
            RETURN DISTINCT neighbor.name AS name
        """,
            names=table_names,
        )
        return [r["name"] for r in result]

    def get_join_conditions_for_path(self, path_names: list) -> list:
        """
        Given ordered list of table short names, return join condition
        for each consecutive hop. Used by SchemaLinker to build
        the JOIN context string sent to LLM.
        """
        if len(path_names) < 2:
            return []
        conditions = []
        for i in range(len(path_names) - 1):
            hops = self.find_join_path(path_names[i], path_names[i + 1])
            if hops:
                conditions.extend(hops)
        return conditions

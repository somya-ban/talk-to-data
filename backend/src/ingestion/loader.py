"""
Schema loader for banking microservice schema.
Reads JSONL files and converts to three training data types:
  - DDL statements (CREATE TABLE) from banking_tables_typed.jsonl
  - Documentation strings from banking_relationships_v2.jsonl
  - Q-SQL pairs added separately via store_qa_pair()

"""

import json
from pathlib import Path
from typing import List, Dict


class SchemaLoader:
    def __init__(
        self,
        tables_path: str = "data/banking_tables_typed.jsonl",
        relations_path: str = "data/banking_relationships_v2.jsonl",
    ):
        self.tables_path = Path(tables_path)
        self.relations_path = Path(relations_path)
        self.tables: Dict = {}
        self.relations: List = []

    def load(self) -> "SchemaLoader":
        self._load_tables()
        self._load_relations()
        return self

    def _load_tables(self):
        with open(self.tables_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.tables[record["table"]] = record

    def _load_relations(self):
        with open(self.relations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.relations.append(json.loads(line))

    def to_ddl_statements(self) -> List[str]:
        """
        Convert each table record to a CREATE TABLE DDL statement.
        DDL is the most powerful training type — specifies table names,
        column names, types, and relationships explicitly.
        """
        ddl_list = []
        for full_name, record in self.tables.items():
            short_name = full_name.split(".")[-1]
            cols = record.get("columns", {})
            pk = record.get("primary_key", "")
            fks = record.get("foreign_keys", {})
            desc = record.get("description", "")

            lines = [f"-- {full_name}: {desc}"]
            lines.append(f"CREATE TABLE {short_name} (")

            col_defs = []
            fk_comments = []  # column definition get commas, fk comments do not
            for col, dtype in cols.items():
                sql_type = self._map_type(dtype)
                if col == pk:
                    col_defs.append(f"    {col} {sql_type} PRIMARY KEY")
                else:
                    col_defs.append(f"    {col} {sql_type}")

            # Add FK constraints as comments (not enforced in DuckDB but informative)
            if fks:
                for fk_col, ref in fks.items():
                    ref_parts = ref.split(".")
                    ref_table = ref_parts[2] if len(ref_parts) >= 4 else ref_parts[-2]
                    ref_col = ref_parts[-1]
                    fk_comments.append(
                        f"    -- FK: {fk_col} references {ref_table}.{ref_col}"
                    )

            body = ",\n".join(col_defs)
            if fk_comments:
                body += "\n" + "\n".join(fk_comments)

            lines.append(body)
            lines.append(");")
            ddl_list.append("\n".join(lines))

        return ddl_list

    def to_documentation_strings(self) -> List[str]:
        """
        Convert relationships to documentation strings.
        Documents business definitions, metric definitions,
        and cross-service relationships.
        """
        docs = []

        # Relationship docs — cross-service business semantics
        for rel in self.relations:
            src = rel["source"].split(".")[-1]
            tgt = rel["target"].split(".")[-1]
            relation = rel["relation"]
            desc = rel.get("description", "")
            docs.append(
                f"Relationship: {src} {relation} {tgt}. {desc}. "
                f"To join these tables, use the foreign key connection "
                f"between {src} and {tgt}."
            )

        # Service-level documentation (Hardcoded)
        service_docs = {
            "CustSrv": "CustSrv contains customer data: Customer (master profile), "
            "CustomerAddress (addresses), CustomerType (classification), "
            "Branch (bank branches), Company (banking entities).",
            "CoreSrv": "CoreSrv contains core banking data: Account (investment accounts), "
            "AccountBalance (balance snapshots), CashMovement (debits/credits), "
            "TransferRequest (transfer requests between accounts).",
            "WealthSrv": "WealthSrv contains wealth management data: Advisor (relationship managers), "
            "ProductWrapper (investment wrappers), WrapProvider (providers), "
            "ProductWrapperType (wrapper types).",
            "TradeSrv": "TradeSrv contains trading data: Order (trade orders), Trade (executed trades), "
            "Instrument (tradable instruments), SecurityType, SecuritySubType, "
            "AggregateOrder (grouped orders), Batch (trading batches).",
            "AuthSrv": "AuthSrv contains authentication data: User (system users), "
            "UserGroup (access groups), UserGroupMembership (role mappings), "
            "Application (applications), Session (user sessions).",
        }
        docs.extend(service_docs.values())

        return docs

    def get_table(self, full_name: str) -> Dict:
        return self.tables.get(full_name, {})

    def all_table_names(self) -> List[str]:
        return list(self.tables.keys())

    def get_services(self) -> List[str]:
        return list({n.split(".")[0] for n in self.tables})

    def table_count(self) -> int:
        return len(self.tables)

    def relation_count(self) -> int:
        return len(self.relations)

    def _map_type(self, jsonl_type: str) -> str:
        """Map JSONL type strings to SQL types."""
        mapping = {
            "INT": "INTEGER",
            "STRING": "VARCHAR(255)",
            "DECIMAL": "DECIMAL(18,4)",
            "DATE": "DATE",
            "DATETIME": "TIMESTAMP",
        }
        return mapping.get(
            jsonl_type.upper(), "VARCHAR(255)"
        )  # If type not specified then default to VARCHAR()

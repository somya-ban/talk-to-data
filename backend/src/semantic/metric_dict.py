"""
Metric dictionary — the semantic layer of the pipeline.

This is the mechanism that makes the Trust pillar real.

Without a semantic layer:
  The LLM guesses what 'revenue' means on every query. In banking, 'net interest
  income' has a regulatory definition. 'Total balance' could mean current balance,
  available balance, or overdraft-adjusted balance. Each query may calculate
  differently. The CFO asks for "total exposure" in January and March and gets
  two different numbers. Trust collapses.

With a semantic layer:
  The user defines their metrics once, before any query runs. Every subsequent
  query is grounded against these definitions. The LLM receives "total_balance:
  SUM(ab.balance) FROM AccountBalance ab" and uses that formula exactly, not
  whatever it infers from column names.

This is the Wren AI AI Studio pattern:
  AI proposes metric definitions based on the schema (not the user writes from
  scratch), the user reviews and confirms in 60 seconds, and the confirmed
  definitions ground every query from that point on. The AI proposes, the human
  governs.

The dbt MetricFlow research finding:
  AI answered 83% of addressable NL questions correctly when grounded through
  a semantic layer, vs much lower without. Our metric dictionary is the
  hackathon-viable implementation of this principle.

Storage:
  - metrics.yaml: version-controlled human-readable definitions
  - ChromaDB documentation-collection: embedded for semantic retrieval
    (the embedder's add_documentation() call stores each metric description
    so the schema linker can retrieve relevant metric context alongside DDL)

Banking metrics proposed by default:
  The BANKING_METRIC_HINTS below are injected into the LLM prompt to guide
  proposals toward metrics that actually make sense in the Banking context.
  Without hints, the LLM may propose generic metrics that don't reflect
  real banking use cases.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional

from groq import Groq


# ── Banking-specific metric hints ─────────────────────────────────────────────
# Injected into the LLM proposal prompt to anchor suggestions in real banking
# domain concepts rather than generic aggregations.

BANKING_METRIC_HINTS = """
For a banking system, the most analytically valuable metrics include:
- Balance metrics: total balance, average balance per account, overdraft utilisation
- Cash flow metrics: total cash inflow (CREDIT), total cash outflow (DEBIT), net cash movement
- Trade metrics: total trade volume (price × quantity), number of trades, average trade size
- Customer metrics: active customer count, high-risk customer count, KYC approval rate
- Account metrics: account count by type, accounts opened this period, closed account rate
- Transfer metrics: total transfer value, failed transfer rate
"""


# ── LLM prompts ────────────────────────────────────────────────────────────────

_PROPOSAL_SYSTEM = """You are a banking data analyst helping define business metrics
for a text-to-SQL system. You will examine a database schema and propose metric
definitions that analysts would actually use.

Rules:
1. Use ONLY the exact table names and column names shown in the schema.
2. Every SQL formula must be valid DuckDB SQL.
3. Table names with spaces or reserved words must be quoted: "Order".
4. When joining tables, use explicit JOIN syntax and table aliases.
5. Return ONLY a valid JSON array. No markdown. No explanation. No backticks.
"""

_PROPOSAL_USER = """Schema context (relevant tables and columns):
{schema_context}

{hints}

Propose exactly 6-8 business metrics relevant to this banking schema.
Each metric must have:
  - name: snake_case identifier (e.g. total_balance)
  - description: one plain English sentence explaining what it measures
  - sql_formula: valid DuckDB SQL expression or subquery that computes this metric
  - tables: list of table names involved
  - columns: list of column names involved

Important: the sql_formula should be a SELECT statement or aggregation expression
that can stand alone as a query or be embedded in a SELECT list.

Return ONLY this JSON structure, no other text:
[
  {{
    "name": "total_balance",
    "description": "Sum of all account balances across all AccountBalance records",
    "sql_formula": "SELECT SUM(ab.balance) AS total_balance FROM AccountBalance ab",
    "tables": ["AccountBalance"],
    "columns": ["balance"]
  }}
]"""


# ── MetricDictionary ───────────────────────────────────────────────────────────


class MetricDictionary:
    """
    Manages the semantic layer: propose → user confirms → store → retrieve.

    Lifecycle in the pipeline:
      1. At startup: load_metrics() — if metrics.yaml exists, load it
      2. First run: propose_metrics(schema_context) → present to user in UI
      3. User confirms: save_confirmed_metrics(confirmed, embedder) → YAML + ChromaDB
      4. Every query: format_for_prompt() → injected as Layer 3 of sql_gen prompt

    The embedder parameter in save_confirmed_metrics() is the SchemaEmbedder instance
    from embedder.py. Each confirmed metric is stored as a documentation entry in
    ChromaDB's documentation-collection via embedder.add_documentation(). This means
    the schema linker can retrieve metric definitions alongside DDL when a question
    involves a known metric concept.
    """

    def __init__(
        self,
        api_key: str = None,
        yaml_path: str = "metrics.yaml",
    ):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.yaml_path = Path(yaml_path)
        self.metrics: Dict[str, Dict] = {}  # keyed by metric name

    # ── Proposal (pre-confirmation) ────────────────────────────────────────────

    def propose_metrics(self, schema_context: str) -> List[Dict]:
        """
        Call Groq with the schema context and return LLM-proposed metric definitions.
        The user reviews and edits these in the UI before they are saved.

        This is the Wren AI AI Studio pattern: AI proposes, human governs.
        The user never writes metric SQL from scratch — they review and confirm.

        Args:
            schema_context: The DDL + join context string from schema_linker.link()
                            or a concatenation of all 25 table DDL statements.

        Returns:
            List of proposal dicts, each with: name, description, sql_formula,
            tables, columns. Validated to ensure required keys exist.
        """
        prompt = _PROPOSAL_USER.format(
            schema_context=schema_context[:4000],  # stay well within context limits
            hints=BANKING_METRIC_HINTS,
        )

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _PROPOSAL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # low — metric definitions need to be precise
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_proposals(raw)

    def propose_from_ddl_list(self, ddl_statements: List[str]) -> List[Dict]:
        """
        Convenience method: propose metrics from a list of DDL strings
        (as returned by loader.to_ddl_statements()).

        Selects the most analytically relevant DDL statements to stay within
        the context window while covering the key banking tables.
        """
        # Prioritise the tables most likely to appear in analytical queries
        priority_keywords = [
            "accountbalance",
            "transaction",
            "trade",
            "cashmove",
            "account",
            "customer",
            "transfer",
        ]

        selected = []
        remaining = []
        for ddl in ddl_statements:
            ddl_lower = ddl.lower()
            if any(kw in ddl_lower for kw in priority_keywords):
                selected.append(ddl)
            else:
                remaining.append(ddl)

        # Take priority tables first, then fill up to context limit
        combined = selected + remaining
        schema_context = "\n\n".join(combined[:15])  # ~15 tables fits comfortably
        return self.propose_metrics(schema_context)

    # ── Confirmation and persistence ───────────────────────────────────────────

    def save_confirmed_metrics(
        self,
        confirmed: List[Dict],
        embedder=None,  # SchemaEmbedder — passed in to avoid circular import
    ) -> int:
        """
        Persist confirmed metrics to:
          1. metrics.yaml (version-controlled single source of truth)
          2. ChromaDB documentation-collection (for semantic retrieval)

        Args:
            confirmed:  List of confirmed metric dicts from the UI
            embedder:   SchemaEmbedder instance. If provided, each metric
                        description is stored as documentation in ChromaDB.
                        If None, only YAML is written (useful for testing).

        Returns:
            Number of metrics saved.
        """
        self.metrics = {}
        for m in confirmed:
            name = m.get("name", "").strip()
            if not name:
                continue
            self.metrics[name] = {
                "name": name,
                "description": m.get("description", ""),
                "sql_formula": m.get("sql_formula", ""),
                "tables": m.get("tables", []),
                "columns": m.get("columns", []),
            }

        # ── Write to YAML ──────────────────────────────────────────────────────
        with open(self.yaml_path, "w") as f:
            yaml.dump(
                {"version": "1.0", "metrics": self.metrics},
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        # ── Write to ChromaDB documentation-collection ─────────────────────────
        # Each metric becomes a documentation string in the embedder.
        # The schema linker's get_related_documentation() call will retrieve
        # relevant metric definitions alongside DDL when a question mentions
        # a known metric concept like "total balance" or "trade volume".
        if embedder is not None:
            for name, m in self.metrics.items():
                doc_string = (
                    f"Metric: {name}\n"
                    f"Description: {m['description']}\n"
                    f"Formula: {m['sql_formula']}\n"
                    f"Tables: {', '.join(m['tables'])}"
                )
                try:
                    embedder.add_documentation(doc_string)
                except Exception:
                    pass  # embedder may not have add_documentation in all versions

        return len(self.metrics)

    def load_metrics(self) -> Dict[str, Dict]:
        """
        Load confirmed metrics from metrics.yaml.
        Called at session startup — if the file exists, metrics are ready
        immediately without needing the user to go through the confirmation flow.
        """
        if not self.yaml_path.exists():
            return {}

        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.metrics = data.get("metrics", {}) if data else {}
        return self.metrics

    # ── Formatting for sql_gen ─────────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """
        Format confirmed metrics for injection into the SQL generator prompt.

        This is Layer 3 of the 6-layer prompt in sql_gen.py. It tells the LLM:
        "when the user asks about total_balance, use SUM(ab.balance) FROM
        AccountBalance ab — not whatever you infer from column names."

        Returns an empty string if no metrics are confirmed yet.
        The sql_gen handles an empty string gracefully.
        """
        if not self.metrics:
            return ""

        lines = [
            "Confirmed business metric definitions.",
            "Use THESE exact formulas when the question involves these concepts:\n",
        ]
        for name, m in self.metrics.items():
            lines.append(f"{name}:")
            lines.append(f"  Description: {m['description']}")
            lines.append(f"  SQL formula: {m['sql_formula']}")
            lines.append("")

        return "\n".join(lines)

    # ── Metric access ──────────────────────────────────────────────────────────

    def get_metric(self, name: str) -> Optional[Dict]:
        """Get a single metric by name."""
        return self.metrics.get(name)

    def metric_count(self) -> int:
        """Number of confirmed metrics currently loaded."""
        return len(self.metrics)

    def is_empty(self) -> bool:
        """True if no metrics have been confirmed yet."""
        return len(self.metrics) == 0

    def metric_names(self) -> List[str]:
        """List of confirmed metric names — used in the UI display."""
        return list(self.metrics.keys())

    # ── Seeded banking metrics (used when no YAML exists) ─────────────────────

    def load_banking_defaults(self) -> Dict[str, Dict]:
        """
        Load a set of sensible default metrics for the banking schema.

        Called when metrics.yaml does not exist and the UI cannot show the
        confirmation flow (e.g. in test_sql_gen.py or automated tests).
        These defaults cover the most common analytical questions a NatWest
        analyst would ask against the 25-table banking schema.

        Does NOT write to disk or ChromaDB — call save_confirmed_metrics()
        if you want to persist them.
        """
        defaults = [
            {
                "name": "total_balance",
                "description": "Sum of all account balances across all AccountBalance records",
                "sql_formula": "SELECT SUM(ab.balance) AS total_balance FROM AccountBalance ab",
                "tables": ["AccountBalance"],
                "columns": ["balance"],
            },
            {
                "name": "average_balance",
                "description": "Average account balance across all AccountBalance records",
                "sql_formula": "SELECT AVG(ab.balance) AS average_balance FROM AccountBalance ab",
                "tables": ["AccountBalance"],
                "columns": ["balance"],
            },
            {
                "name": "trade_volume",
                "description": "Total value of all trades calculated as price multiplied by quantity",
                "sql_formula": "SELECT SUM(t.price * t.quantity) AS trade_volume FROM Trade t",
                "tables": ["Trade"],
                "columns": ["price", "quantity"],
            },
            {
                "name": "cash_inflow",
                "description": "Total cash credited to accounts (direction = CREDIT)",
                "sql_formula": "SELECT SUM(cm.amount) AS cash_inflow FROM CashMovement cm WHERE cm.direction = 'CREDIT'",
                "tables": ["CashMovement"],
                "columns": ["amount", "direction"],
            },
            {
                "name": "cash_outflow",
                "description": "Total cash debited from accounts (direction = DEBIT)",
                "sql_formula": "SELECT SUM(cm.amount) AS cash_outflow FROM CashMovement cm WHERE cm.direction = 'DEBIT'",
                "tables": ["CashMovement"],
                "columns": ["amount", "direction"],
            },
            {
                "name": "active_customer_count",
                "description": "Number of distinct customers who have at least one account",
                "sql_formula": "SELECT COUNT(DISTINCT a.customer_id) AS active_customer_count FROM Account a",
                "tables": ["Account"],
                "columns": ["customer_id"],
            },
            {
                "name": "high_risk_customer_count",
                "description": "Number of customers classified as HIGH or VERY_HIGH risk",
                "sql_formula": "SELECT COUNT(*) AS high_risk_customer_count FROM Customer c WHERE c.risk_rating IN ('HIGH', 'VERY_HIGH')",
                "tables": ["Customer"],
                "columns": ["risk_rating"],
            },
            {
                "name": "total_transfer_value",
                "description": "Total value of all completed transfer requests",
                "sql_formula": "SELECT SUM(tr.amount) AS total_transfer_value FROM TransferRequest tr WHERE tr.status = 'COMPLETED'",
                "tables": ["TransferRequest"],
                "columns": ["amount", "status"],
            },
        ]

        self.metrics = {m["name"]: m for m in defaults}
        return self.metrics

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _parse_proposals(self, raw: str) -> List[Dict]:
        """
        Parse LLM response into a list of validated metric dicts.
        Strips markdown fences if the model added them despite instructions.
        Falls back to banking defaults if parsing fails.
        """
        # Strip markdown code fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.strip().startswith("["):
                    raw = part
                    break

        try:
            proposals = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON array with a regex fallback
            import re

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    proposals = json.loads(match.group())
                except json.JSONDecodeError:
                    return list(self.load_banking_defaults().values())
            else:
                return list(self.load_banking_defaults().values())

        # Validate structure — require name, description, sql_formula at minimum
        validated = []
        for p in proposals:
            if not isinstance(p, dict):
                continue
            if not all(k in p for k in ["name", "description", "sql_formula"]):
                continue
            p.setdefault("tables", [])
            p.setdefault("columns", [])
            # Clean the name to snake_case
            p["name"] = p["name"].lower().replace(" ", "_").replace("-", "_")
            validated.append(p)

        if not validated:
            return list(self.load_banking_defaults().values())

        return validated

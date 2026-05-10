"""
DuckDB SQL executor with self-correction loop — Stage 3 of the query pipeline.

This file does three things and exactly three things:
  1. Execute the SQL string from sql_gen.generate() against DuckDB
  2. Return results as a pandas DataFrame
  3. On failure: classify the error, call sql_gen.regenerate_with_feedback(),
     retry — up to MAX_RETRIES rounds (standard is 3 across all production systems)

Design decisions grounded in research:
  - DIN-SQL self-correction (NeurIPS 2023): execution feedback + typed error classification
    adds ~10 percentage points over vanilla generation. The key insight: the LLM benefits
    from knowing WHAT type of error occurred, not just the raw message.
  - LinkedIn SQL Bot "Fix with AI": 80% session usage, their single highest-ROI feature.
    This is the production proof that execution-based correction works at scale.
  - SQL-of-Thought (Sept 2025): taxonomy-guided dynamic error modification. Each ErrorType
    value is written as an actionable instruction — the LLM reads it and acts on it.
  - ExCoT-DPO (Snowflake, ACL 2025): models learn from execution failures. Our correction
    loop stores successful corrections back to ChromaDB (fix loop), which is the same
    principle applied without fine-tuning.

Empty result handling:
  A query that returns 0 rows is a SOFT FAILURE. The SQL ran without error but the WHERE
  conditions, JOIN paths, or value casing is wrong. The most common real-world case:
  status = 'active' when the data has status = 'ACTIVE'. We re-prompt rather than silently
  returning an empty DataFrame — the user asked a question, they deserve an answer.

User-confirmation integration:
  This module returns CorrectionHistory entries so the Flask/React layer can surface
  the correction to the user: "The query failed with X, I rewrote it as Y — does this
  look right?" If the user confirms, the UI stores the pair in ChromaDB (fix loop).
  The executor does not call ChromaDB directly — separation of concerns.

Privacy guard:
  The guard (src/privacy/guard.py) runs AFTER execution, BEFORE returning to the user.
  This module ensures there is a result worth masking before the guard runs.
"""

import duckdb
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

from backend.src.generation.sql_gen import SQLGenerator, SQLResult

MAX_RETRIES = 3


# ── Error taxonomy ─────────────────────────────────────────────────────────────


class ErrorType(str, Enum):
    """
    Taxonomy-guided error classification — SQL-of-Thought (Sept 2025).

    Each value is the exact string injected into the regeneration prompt.
    They are written as actionable instructions, not just labels.
    The LLM reads the value and knows precisely what to fix.
    """

    SYNTAX = (
        "SYNTAX_ERROR — rewrite the SQL. Check: unclosed parentheses, missing commas, "
        "invalid DuckDB syntax. DuckDB date functions: strftime('%Y-%m', col), "
        "EXTRACT(YEAR FROM col), DATE_TRUNC('month', col)."
    )
    COLUMN_NOT_FOUND = (
        "COLUMN_NOT_FOUND — use ONLY the exact column names in the schema context above. "
        "Never invent or abbreviate column names. Check spelling and casing exactly. "
        "If you need a column that isn't in the schema, use a different approach."
    )
    TABLE_NOT_FOUND = (
        "TABLE_NOT_FOUND — use ONLY the short table names shown in the schema context. "
        "Do not qualify with service names (use 'Customer' not 'CustSrv.Customer'). "
        "Do not alias tables you were not given."
    )
    EMPTY_RESULT = (
        "EMPTY_RESULT — the query ran but returned 0 rows. Likely causes: "
        "(1) WHERE clause uses wrong value casing — check sample values in the schema; "
        "(2) JOIN condition references wrong column names — use the join path provided; "
        "(3) date range does not overlap with the data (dates are in 2023-2024). "
        "Relax or remove filters and retry."
    )
    TYPE_ERROR = (
        "TYPE_ERROR — data type mismatch. Check: comparing VARCHAR to INT, "
        "arithmetic on VARCHAR columns. Cast explicitly if needed: CAST(col AS INTEGER). "
        "DuckDB INT columns: do not quote. VARCHAR columns: use single quotes."
    )
    AMBIGUOUS = (
        "AMBIGUOUS_COLUMN — the column name exists in multiple joined tables. "
        "Qualify every column reference with its table alias: t.trade_id, a.account_id. "
        "Add aliases for every table in the FROM and JOIN clauses."
    )
    JOIN_ERROR = (
        "JOIN_ERROR — the join condition is wrong or the join path is incomplete. "
        "Use the join paths shown in the schema context above. "
        "Verify that the FK column on the child table matches the PK column on the parent."
    )
    UNKNOWN = (
        "UNKNOWN_ERROR — the SQL failed for an unexpected reason. "
        "Rewrite from scratch using only the schema context and join paths provided."
    )


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class CorrectionEntry:
    """One round of the self-correction loop."""

    attempt: int
    sql: str
    error_type: ErrorType
    error_message: str


@dataclass
class ExecutionResult:
    """
    Complete result from one end-to-end question → execution cycle.

    The Flask/React layer uses this to:
      - Display the answer (df)
      - Show the final SQL and whether it was corrected (for source transparency)
      - Surface correction history to the user for confirmation
      - Store confirmed corrections back to ChromaDB (fix loop)
    """

    df: Optional[pd.DataFrame]  # None if all retries failed
    final_sql_result: SQLResult  # Final SQLResult (may be corrected)
    was_corrected: bool  # True if any retry was needed
    attempts: int  # Total LLM calls made
    correction_history: List[CorrectionEntry]  # Every failed round, for the UI
    row_count: int  # 0 if failed
    success: bool  # True if df is not None


# ── Main executor class ────────────────────────────────────────────────────────


class SQLExecutor:
    """
    Stage 3 of the query pipeline.

    Holds a DuckDB connection that has been pre-populated by data_seeder.py.
    The seeder and executor share the same in-memory DuckDB instance — they
    must use the same connection object, not two separate connections.

    Usage:
        conn = duckdb.connect(":memory:")
        seeder = BankingDataSeeder(conn)
        seeder.seed_all()                   # populates all 25 tables
        executor = SQLExecutor(conn)

        schema_ctx = linker.link(question)
        sql_result = sql_gen.generate(question, schema_ctx)
        exec_result = executor.execute(question, sql_result, sql_gen, schema_ctx)

        if exec_result.success:
            df = exec_result.df             # pass to privacy guard
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def execute(
        self,
        question: str,
        initial_sql_result: SQLResult,
        generator: SQLGenerator,
        schema_context,  # SchemaContext from schema_linker.link()
    ) -> ExecutionResult:
        """
        Main entry point: execute SQL, self-correct on failure, max MAX_RETRIES rounds.

        Round 1: run the initial SQL from sql_gen.generate()
        Round 2+: classify error, call regenerate_with_feedback(), run again

        Args:
            question:            Original natural language question (passes through to regenerate)
            initial_sql_result:  SQLResult from sql_gen.generate()
            generator:           SQLGenerator instance (called for retries)
            schema_context:      SchemaContext from schema_linker — needed for regeneration

        Returns:
            ExecutionResult with full metadata for the UI layer
        """
        current_result = initial_sql_result
        correction_history: List[CorrectionEntry] = []
        was_corrected = False

        for attempt in range(1, MAX_RETRIES + 1):
            df, error_msg = self._run(current_result.sql)

            if df is not None:
                # ── Success ──────────────────────────────────────────────────
                return ExecutionResult(
                    df=df,
                    final_sql_result=current_result,
                    was_corrected=was_corrected,
                    attempts=attempt,
                    correction_history=correction_history,
                    row_count=len(df),
                    success=True,
                )

            # ── Failure — classify and maybe retry ───────────────────────────
            error_type = self._classify(error_msg)

            correction_history.append(
                CorrectionEntry(
                    attempt=attempt,
                    sql=current_result.sql,
                    error_type=error_type,
                    error_message=error_msg,
                )
            )

            if attempt < MAX_RETRIES:
                # Re-generate with full error context.
                # The error_type.value string is a direct instruction to the LLM.
                # Temperature is slightly higher on retry (set inside regenerate_with_feedback)
                # to escape the failure mode without going fully exploratory.
                full_error = (
                    f"Error type: {error_type.value.split('—')[0].strip()}\n"
                    f"Error: {error_msg}"
                )
                current_result = generator.regenerate_with_feedback(
                    context=schema_context,
                    previous_sql=current_result.sql,
                    error_message=full_error,
                )
                was_corrected = True

        # ── All retries exhausted ────────────────────────────────────────────
        return ExecutionResult(
            df=None,
            final_sql_result=current_result,
            was_corrected=True,
            attempts=MAX_RETRIES,
            correction_history=correction_history,
            row_count=0,
            success=False,
        )

    def execute_direct(self, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL with no retry.

        Used for:
          1. User-submitted manual corrections in the fix loop
          2. The metric_dict validation step (verifying a formula executes before saving)
          3. Tests and debugging
        """
        return self._run(sql)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _run(self, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL against DuckDB. Returns (DataFrame, None) on success,
        (None, error_string) on any failure.

        Empty result → soft failure. We return a descriptive message explaining
        what likely went wrong so the LLM knows what to fix in the retry prompt.
        """
        try:
            df = self.conn.execute(sql).df()

            if len(df) == 0:
                # Soft failure: SQL ran but matched nothing.
                # This is the most common real-world failure mode in banking schemas:
                # status = 'active' when the column has status = 'ACTIVE',
                # or a JOIN condition that produces a Cartesian-product-then-filter-to-zero.
                return None, (
                    "EMPTY_RESULT: query executed without error but returned 0 rows. "
                    "Likely causes: (1) WHERE clause value casing is wrong — the banking "
                    "data uses uppercase status values like 'ACTIVE', 'PENDING', 'CLOSED'; "
                    "(2) JOIN condition references the wrong FK column; "
                    "(3) Date filter excludes all rows — data range is 2023-01-01 to 2024-12-31."
                )

            return df, None

        except duckdb.CatalogException as e:
            # Table or column doesn't exist
            return None, f"CATALOG_ERROR: {e}"

        except duckdb.ParserException as e:
            # Syntax error — DuckDB couldn't parse the SQL at all
            return None, f"PARSER_ERROR: {e}"

        except duckdb.BinderException as e:
            # Ambiguous column or column not found after parsing
            return None, f"BINDER_ERROR: {e}"

        except duckdb.ConversionException as e:
            # Type mismatch during execution
            return None, f"CONVERSION_ERROR: {e}"

        except duckdb.IOException as e:
            return None, f"IO_ERROR: {e}"

        except Exception as e:
            return None, f"EXECUTION_ERROR: {type(e).__name__}: {e}"

    def classify(self, error_msg: str) -> ErrorType:
        """
        Map a raw DuckDB error message to a typed ErrorType.

        Order matters — check most specific patterns first.
        The matched ErrorType.value goes directly into the regeneration prompt.
        """
        err = error_msg.lower()

        # Ambiguous must come before COLUMN_NOT_FOUND (both mention "column")
        if "ambiguous" in err:
            return ErrorType.AMBIGUOUS

        if any(
            k in err
            for k in [
                "parser_error",
                "syntax",
                "parse error",
                "unexpected token",
            ]
        ):
            return ErrorType.SYNTAX

        # Table not found — check BEFORE column (both match "does not exist")
        if ("catalog_error" in err or "binder_error" in err) and any(
            k in err for k in ["table", "relation", "view"]
        ):
            return ErrorType.TABLE_NOT_FOUND

        # Column not found — DuckDB raises CatalogException for missing columns
        if ("catalog_error" in err or "binder_error" in err) and any(
            k in err for k in ["column", "attribute", "does not exist", "not found"]
        ):
            return ErrorType.COLUMN_NOT_FOUND

        if "empty_result" in err or "0 rows" in err or "returned 0" in err:
            return ErrorType.EMPTY_RESULT

        if any(
            k in err
            for k in [
                "conversion_error",
                "type",
                "cast",
                "cannot",
                "invalid",
                "overflow",
            ]
        ):
            return ErrorType.TYPE_ERROR

        if any(k in err for k in ["join", "foreign key", "reference"]):
            return ErrorType.JOIN_ERROR

        return ErrorType.UNKNOWN

    # ── Introspection helpers (used by tests + seeder verification) ────────────

    def list_tables(self) -> List[str]:
        """List all tables loaded in DuckDB — for debugging and seeder verification."""
        try:
            result = self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchdf()
            return result["table_name"].tolist()
        except Exception:
            return []

    def row_count(self, table_name: str) -> int:
        """Return row count for a table — used in seeder verification output."""
        try:
            return self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except Exception:
            return 0

    def column_names(self, table_name: str) -> List[str]:
        """Return column names for a table — used in seeder + guard integration."""
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            return result["column_name"].tolist()
        except Exception:
            return []

    def validate_metric_formula(self, formula_sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a metric formula executes without error.
        Called by metric_dict.py before saving a user-confirmed metric.

        The formula is wrapped in a SELECT against any available table so DuckDB
        can parse and bind it. If it fails, the error is shown to the user.
        """
        tables = self.list_tables()
        if not tables:
            return False, "No tables loaded. Run the data seeder first."

        # Try wrapping in a SELECT against the first available table
        test_sql = f"SELECT {formula_sql} FROM {tables[0]} LIMIT 1"
        df, error = self._run(test_sql)
        if df is not None:
            return True, None
        return False, error

"""
SQL generator — converts a SchemaContext into a SQL query using Groq LLM.

Follows Vanna's get_sql_prompt() pattern:
  1. System prompt: role, rules, database context
  2. Few-shot examples: similar Q-SQL pairs from ChromaDB
  3. Schema context: relevant DDLs + join conditions
  4. Documentation: business definitions + metric formulas
  5. User question: the actual question to answer

LLM: Groq llama-3.3-70b-versatile (free tier, 6000 RPM)
Extraction: regex-based SQL block extraction with fallback
Validation: syntax check before returning

Design informed by:
  - Vanna's base.py get_sql_prompt() and extract_sql()
  - DIN-SQL paper: schema linking → SQL decomposition → generation
  - CHESS paper: evidence-based prompting with join conditions
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from src.retrieval.schema_linker import SchemaContext

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.0  # Deterministic — SQL generation needs consistency
MAX_RETRIES = 3  # Self-correction loop attempts


# ── Result object ──────────────────────────────────────────────────────────────


@dataclass
class SQLResult:
    """
    Output of the SQL generator for a single question.

    Fields:
      sql        — the generated SQL query, clean and executable
      prompt     — full prompt sent to LLM (used by fix loop to regenerate)
      explanation— plain English explanation of what the SQL does
      success    — True if valid SQL was extracted
      error      — error message if success is False
      attempts   — number of LLM calls made (1 normally, up to 3 with retries)
    """

    sql: str
    prompt: str
    explanation: str = ""
    success: bool = True
    error: str = ""
    attempts: int = 1


# ── SQL Generator ──────────────────────────────────────────────────────────────


class SQLGenerator:
    """
    Converts a SchemaContext into executable SQL using Groq LLM.

    The prompt is constructed in layers following Vanna's architecture:
      Layer 1 — System: role and strict rules
      Layer 2 — Schema: DDL statements (table structure)
      Layer 3 — Joins: join conditions from Neo4j graph
      Layer 4 — Docs: business definitions and metric formulas
      Layer 5 — Examples: similar Q-SQL pairs (few-shot)
      Layer 6 — Question: the actual user question

    Each layer is only included if it has content — no empty sections
    in the prompt that could confuse the LLM.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env. Get your free key at console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        logger.info(f"SQLGenerator initialised with model: {MODEL}")

    # ── Main entry point ───────────────────────────────────────────────────────

    def generate(self, context: SchemaContext) -> SQLResult:
        """
        Generate SQL for a user question given its SchemaContext.
        Attempts up to MAX_RETRIES times if extraction fails.
        Returns a SQLResult with the SQL, prompt, and metadata.
        """
        if context.is_empty():
            return SQLResult(
                sql="",
                prompt="",
                success=False,
                error="SchemaContext is empty — no tables retrieved",
            )

        prompt = self._build_prompt(context)
        attempts = 0
        last_error = ""

        while attempts < MAX_RETRIES:
            attempts += 1
            try:
                raw_response = self._call_groq(prompt)
                sql = self._extract_sql(raw_response)
                explanation = self._extract_explanation(raw_response)

                if not sql:
                    last_error = "No SQL block found in LLM response"
                    logger.warning(
                        f"Attempt {attempts}: no SQL extracted. "
                        f"Raw response: {raw_response[:200]}"
                    )
                    continue

                if not self._is_valid_sql(sql):
                    last_error = f"Extracted text does not look like SQL: {sql[:100]}"
                    logger.warning(f"Attempt {attempts}: invalid SQL extracted")
                    continue

                logger.info(f"SQL generated in {attempts} attempt(s): {sql[:80]}...")
                return SQLResult(
                    sql=sql,
                    prompt=prompt,
                    explanation=explanation,
                    success=True,
                    error="",
                    attempts=attempts,
                )

            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempts}: Groq API error: {e}")
                if attempts >= MAX_RETRIES:
                    break

        return SQLResult(
            sql="",
            prompt=prompt,
            success=False,
            error=f"Failed after {attempts} attempts. Last error: {last_error}",
            attempts=attempts,
        )

    def regenerate_with_feedback(
        self, context: SchemaContext, previous_sql: str, error_message: str
    ) -> SQLResult:
        """
        Regenerate SQL after an execution error.
        Called by the fix loop (executor.py) when SQL fails to run.
        Appends the previous SQL and error to the prompt so the LLM
        can correct the specific mistake.
        """
        base_prompt = self._build_prompt(context)
        feedback_block = self._build_feedback_block(previous_sql, error_message)
        prompt = base_prompt + feedback_block

        attempts = 0
        last_error = ""

        while attempts < MAX_RETRIES:
            attempts += 1
            try:
                raw_response = self._call_groq(prompt)
                sql = self._extract_sql(raw_response)
                explanation = self._extract_explanation(raw_response)

                if sql and self._is_valid_sql(sql):
                    logger.info(f"SQL corrected in {attempts} attempt(s)")
                    return SQLResult(
                        sql=sql,
                        prompt=prompt,
                        explanation=explanation,
                        success=True,
                        error="",
                        attempts=attempts,
                    )

            except Exception as e:
                last_error = str(e)
                logger.error(f"Regeneration attempt {attempts}: {e}")

        return SQLResult(
            sql="",
            prompt=prompt,
            success=False,
            error=f"Regeneration failed: {last_error}",
            attempts=attempts,
        )

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_prompt(self, context: SchemaContext) -> str:
        """
        Build the full LLM prompt from SchemaContext.

        Structure follows Vanna's get_sql_prompt() pattern with additions
        for join conditions (our Graph-RAG enhancement).

        Every section is clearly labelled so the LLM understands
        the role of each piece of information.
        """
        sections = []

        # ── Layer 1: System instructions ──────────────────────────────────────
        sections.append(self._system_prompt())

        # ── Layer 2: Schema — DDL statements ──────────────────────────────────
        if context.relevant_ddls:
            ddl_block = "\n\n".join(context.relevant_ddls)
            sections.append(f"""### DATABASE SCHEMA
The following CREATE TABLE statements define the available tables.
Use ONLY these tables and columns in your SQL — do not invent columns.

{ddl_block}""")

        # ── Layer 3: Join conditions — Neo4j graph output ──────────────────────
        if context.join_context:
            sections.append(f"""### {context.join_context}

Use these exact join conditions when joining tables.
Do NOT guess join columns — use only the conditions listed above.""")

        # ── Layer 4: Business documentation and metric definitions ─────────────
        if context.relevant_docs:
            doc_block = "\n".join(f"- {d}" for d in context.relevant_docs)
            sections.append(f"""### BUSINESS DEFINITIONS
{doc_block}""")

        # ── Layer 5: Few-shot examples — similar past Q-SQL pairs ──────────────
        if context.similar_sql:
            examples = []
            for pair in context.similar_sql:
                q = pair.get("question", "")
                sql = pair.get("sql", "")
                if q and sql:
                    examples.append(f"Question: {q}\nSQL:\n```sql\n{sql}\n```")
            if examples:
                example_block = "\n\n".join(examples)
                sections.append(f"""### SIMILAR EXAMPLES
{example_block}""")

        # ── Layer 6: The user question ─────────────────────────────────────────
        sections.append(f"""### QUESTION
{context.question}

Generate a single SQL query that answers this question.
Return the SQL inside a ```sql code block.
After the SQL block, write one sentence explaining what the query does.""")

        return "\n\n".join(sections)

    def _system_prompt(self) -> str:
        """
        System-level instructions for the LLM.
        Establishes role, constraints, and output format.
        Kept concise — long system prompts dilute attention.
        """
        return """You are an expert SQL analyst for a banking microservices platform.
Your job is to generate accurate, executable SQL queries.

STRICT RULES:
1. Use ONLY tables and columns that appear in the DATABASE SCHEMA section.
2. Use ONLY join conditions from the JOIN CONDITIONS section — never guess.
3. Never use SELECT * — always name the specific columns needed.
4. Always use table aliases for clarity (e.g. c for Customer, a for Account).
5. Use ANSI SQL syntax compatible with DuckDB.
6. If a question involves an aggregate (total, average, count), use GROUP BY correctly.
7. Always return a single, complete, executable SQL statement.
8. Do not include explanations inside the SQL — only valid SQL syntax."""

    def _build_feedback_block(self, previous_sql: str, error_message: str) -> str:
        """
        Appended to the base prompt during the fix loop.
        Tells the LLM exactly what went wrong and asks it to correct it.
        """
        return f"""

### CORRECTION REQUIRED
The following SQL query was generated but failed to execute:

```sql
{previous_sql}
```

Error message: {error_message}

Please fix the SQL query. Identify the specific error and correct it.
Return the corrected SQL inside a ```sql code block."""

    # ── Groq API call ──────────────────────────────────────────────────────────

    def _call_groq(self, prompt: str) -> str:
        """
        Single call to Groq API.
        Uses chat completions with system/user split for best results.
        Temperature 0.0 for deterministic SQL generation.
        """
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    # ── SQL extraction ─────────────────────────────────────────────────────────

    def _extract_sql(self, response: str) -> str:
        """
        Extract clean SQL from LLM response.

        Strategy (in order):
          1. Extract from ```sql ... ``` code block (preferred)
          2. Extract from generic ``` ... ``` code block
          3. Look for SELECT/WITH/INSERT at start of a line (fallback)

        Strips all whitespace and trailing semicolons then re-adds one
        semicolon for consistency.

        Mirrors Vanna's extract_sql() logic — battle-tested pattern.
        """
        # Strategy 1: ```sql ... ``` block
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return self._clean_sql(match.group(1))

        # Strategy 2: generic ``` ... ``` block
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if self._looks_like_sql(candidate):
                return self._clean_sql(candidate)

        # Strategy 3: find SELECT/WITH statement directly in text
        match = re.search(
            r"((?:SELECT|WITH|INSERT|UPDATE|DELETE)\s+.+)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return self._clean_sql(match.group(1))

        return ""

    def _extract_explanation(self, response: str) -> str:
        """
        Extract the plain English explanation that follows the SQL block.
        Used in the UI to show users what the query does.
        """
        # Remove SQL code blocks first
        cleaned = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # Take the first non-empty sentence
        sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
        if sentences:
            return sentences[0] + "."
        return ""

    def _clean_sql(self, sql: str) -> str:
        """
        Normalise extracted SQL.
        - Strip leading/trailing whitespace
        - Remove trailing semicolon then add exactly one back
        - Collapse multiple blank lines
        """
        sql = sql.strip()
        sql = sql.rstrip(";").strip()
        sql = re.sub(r"\n{3,}", "\n\n", sql)
        return sql + ";"

    def _looks_like_sql(self, text: str) -> bool:
        """Quick check that extracted text is plausibly SQL."""
        keywords = ["SELECT", "WITH", "FROM", "WHERE", "JOIN", "GROUP", "INSERT"]
        text_upper = text.upper()
        return any(kw in text_upper for kw in keywords)

    def _is_valid_sql(self, sql: str) -> bool:
        """
        Basic SQL validation before returning to caller.
        Checks that:
          - SQL is not empty
          - Contains at least SELECT or WITH
          - Contains FROM (needed for any real query)
          - Does not contain placeholder text like <column_name>
        """
        if not sql or len(sql.strip()) < 10:
            return False

        sql_upper = sql.upper()

        if not any(kw in sql_upper for kw in ["SELECT", "WITH"]):
            return False

        if "FROM" not in sql_upper:
            return False

        # Reject if LLM left placeholder markers
        if "<" in sql and ">" in sql:
            return False

        return True


# ── Seeded Q-SQL pairs ─────────────────────────────────────────────────────────

SEED_QA_PAIRS = [
    {
        "question": "Show me all customers with high risk rating",
        "sql": (
            "SELECT c.customer_id, c.name, c.risk_rating "
            "FROM Customer c "
            "WHERE c.risk_rating = 'HIGH';"
        ),
    },
    {
        "question": "What is the total balance for each account",
        "sql": (
            "SELECT a.account_id, a.account_type, SUM(ab.balance) AS total_balance "
            "FROM Account a "
            "JOIN AccountBalance ab ON ab.account_id = a.account_id "
            "GROUP BY a.account_id, a.account_type;"
        ),
    },
    {
        "question": "List all trades executed for a given order",
        "sql": (
            "SELECT t.trade_id, t.price, t.quantity, o.order_id "
            "FROM Trade t "
            "JOIN Order o ON t.order_id = o.order_id;"
        ),
    },
    {
        "question": "Which advisors are assigned to accounts",
        "sql": (
            "SELECT ad.advisor_id, ad.name, a.account_id, a.account_type "
            "FROM Advisor ad "
            "JOIN Account a ON a.advisor_id = ad.advisor_id;"
        ),
    },
    {
        "question": "Show cash movements with direction for each account",
        "sql": (
            "SELECT cm.movement_id, cm.amount, cm.direction, cm.timestamp, "
            "a.account_id "
            "FROM CashMovement cm "
            "JOIN Account a ON cm.account_id = a.account_id "
            "ORDER BY cm.timestamp DESC;"
        ),
    },
    {
        "question": "Find all orders placed by customers in a specific branch",
        "sql": (
            "SELECT o.order_id, o.side, o.quantity, c.name AS customer_name "
            "FROM Order o "
            "JOIN Account a ON o.account_id = a.account_id "
            "JOIN Customer c ON a.customer_id = c.customer_id "
            "JOIN Branch b ON c.branch_id = b.branch_id "
            "WHERE b.branch_id = 1;"
        ),
    },
    {
        "question": "What instruments have been traded and their security types",
        "sql": (
            "SELECT i.instrument_id, i.name, st.type_name "
            "FROM Instrument i "
            "JOIN SecurityType st ON i.security_type_id = st.security_type_id;"
        ),
    },
    {
        "question": "Count the number of trades per account",
        "sql": (
            "SELECT a.account_id, COUNT(t.trade_id) AS trade_count "
            "FROM Account a "
            "JOIN Trade t ON t.account_id = a.account_id "
            "GROUP BY a.account_id "
            "ORDER BY trade_count DESC;"
        ),
    },
    {
        "question": "Show all transfer requests with source and destination account",
        "sql": (
            "SELECT tr.transfer_id, tr.amount, tr.status, "
            "tr.source_account, tr.dest_account "
            "FROM TransferRequest tr;"
        ),
    },
    {
        "question": "List all users and their associated advisor",
        "sql": (
            "SELECT u.user_id, u.name AS user_name, "
            "ad.name AS advisor_name "
            "FROM User u "
            "JOIN Advisor ad ON u.advisor_id = ad.advisor_id;"
        ),
    },
]

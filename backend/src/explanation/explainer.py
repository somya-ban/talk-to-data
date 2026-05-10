"""
Result explainer — Stage 5 of the query pipeline.

Three responsibilities, all tied to the three pillars:

1. NARRATION (Clarity pillar)
   Calls LLM with the question, the result summary, and the metric used.
   Returns 2-3 plain English sentences. No jargon. No SQL. No column names.
   The user asked a business question — they get a business answer.

2. CHART TYPE SELECTION (Clarity pillar)
   Inspects the DataFrame shape and column types to auto-select the right
   visualisation. The decision follows the same logic production BI tools use:
     - line   → time-series data (date/timestamp column + numeric column)
     - bar    → categorical comparison (one category, one numeric)
     - stat   → single aggregated value (1 row, 1 column)
     - table  → everything else (multi-column, multi-category, decomposition)

3. TRANSPARENCY METADATA (Trust pillar)
   Returns which metric definition was used, its plain English description,
   and the formula — structured so the React layer can render the inline
   "Using: total_balance — Sum of all account balances" line without the
   SQL being the primary surface.

Design reference:
  ThoughtSpot Spotter shows reasoning in plain language, not raw SQL.
  Wren AI shows "AI-generated summaries alongside SQL" with SQL collapsed.
  Julius AI shows generated code fully — but their audience is data scientists.
  Our audience is banking analysts. Primary surface: plain English narration
  + one-line metric attribution. SQL in a collapsed panel for verification.

The narration prompt is carefully constrained:
  - 2 sentences maximum for simple results
  - 3 sentences maximum for complex multi-row results
  - Must mention the specific number / key finding
  - Must reference the metric name naturally (not the SQL formula)
  - Must not use the words "query", "SQL", "database", "table", "column"

ExplainerResult is the contract this module returns to app.py.
app.py serialises it to JSON for the React frontend.
"""

import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from groq import Groq


# ── Chart type enum values ─────────────────────────────────────────────────────
# Used by React to pick the right Recharts component.
CHART_LINE = "line"
CHART_BAR = "bar"
CHART_TABLE = "table"
CHART_STAT = "stat"  # single KPI number — no chart needed


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class ExplainerResult:
    """
    Structured output from the explainer.
    This is serialised to JSON and returned in the /api/query response.
    """

    narration: str  # 2-3 sentence plain English answer
    chart_type: str  # line | bar | table | stat
    chart_data: List[Dict[str, Any]]  # DataFrame as list-of-dicts (JSON-safe)
    columns: List[str]  # column names for the React table/chart
    metric_used: Optional[Dict]  # {name, description, sql_formula} or None
    row_count: int
    x_key: Optional[str]  # x-axis column for line/bar charts
    y_key: Optional[str]  # y-axis column for line/bar charts


# ── Narration prompts ──────────────────────────────────────────────────────────

_NARRATION_SYSTEM = """You are a banking data analyst explaining query results to
a business user who does not read SQL or technical jargon.

Rules:
- Write exactly 2-3 sentences. No more.
- State the key finding with the actual number(s) from the results.
- Reference the metric by its business name (e.g. "total balance"), not its formula.
- Do not use the words: query, SQL, database, table, column, schema, execute, run.
- Do not start with "The query" or "Based on your query".
- Write as a concise analyst summary, not a technical description.
- Use £ for GBP amounts, $ for USD. Format large numbers with commas."""

_NARRATION_USER = """Question asked: {question}

{metric_context}

Result summary ({row_count} rows returned):
{result_summary}

Write the analyst summary now:"""


# ── ResultExplainer ────────────────────────────────────────────────────────────


class ResultExplainer:
    """
    Stage 5 of the query pipeline.

    Usage in app.py:
        explainer = ResultExplainer(api_key=api_key)
        result = explainer.explain(
            question=question,
            df=masked_df,
            metric_used=metric_dict.get_metric("total_balance"),
        )
        # result.narration  → "The total balance across all active accounts..."
        # result.chart_type → "bar"
        # result.chart_data → [{"country": "UK", "total_balance": 4200000}, ...]
    """

    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def explain(
        self,
        question: str,
        df: pd.DataFrame,
        metric_used: Optional[Dict] = None,
    ) -> ExplainerResult:
        """
        Main entry point. Takes the masked result DataFrame and returns
        a fully structured ExplainerResult ready for JSON serialisation.
        """
        if df is None or df.empty:
            return self._empty_result(question)

        chart_type = self._select_chart_type(df)
        x_key, y_key = self._select_axes(df, chart_type)
        chart_data = self._to_chart_data(df)
        narration = self._narrate(question, df, metric_used, chart_type)

        return ExplainerResult(
            narration=narration,
            chart_type=chart_type,
            chart_data=chart_data,
            columns=list(df.columns),
            metric_used=metric_used,
            row_count=len(df),
            x_key=x_key,
            y_key=y_key,
        )

    # ── Chart type selection ───────────────────────────────────────────────────

    def _select_chart_type(self, df: pd.DataFrame) -> str:
        """
        Select visualisation type based on DataFrame shape and column types.

        Decision logic (in priority order):
          1. Single cell (1 row, 1 col) → stat (large KPI number display)
          2. Date/timestamp col + numeric col → line (time-series)
          3. 1 string col + 1 numeric col → bar (categorical comparison)
          4. Everything else → table
        """
        cols = df.columns.tolist()
        n_rows, n_cols = df.shape

        # Single KPI value
        if n_rows == 1 and n_cols == 1:
            return CHART_STAT

        # Find column type categories
        date_cols = [c for c in cols if self._is_date_col(df, c)]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        text_cols = [
            c
            for c in cols
            if not pd.api.types.is_numeric_dtype(df[c]) and not self._is_date_col(df, c)
        ]

        # Time series: has a date column and at least one numeric column
        if date_cols and numeric_cols and n_cols <= 4:
            return CHART_LINE

        # Categorical comparison: exactly one text col and one numeric col
        if len(text_cols) == 1 and len(numeric_cols) == 1 and n_cols == 2:
            return CHART_BAR

        # Multi-category bar (e.g. top N countries by revenue)
        if (
            len(text_cols) >= 1
            and len(numeric_cols) >= 1
            and n_rows <= 20
            and n_cols <= 3
        ):
            return CHART_BAR

        return CHART_TABLE

    def _select_axes(
        self,
        df: pd.DataFrame,
        chart_type: str,
    ) -> tuple:
        """
        Select x_key and y_key for Recharts line/bar charts.
        Returns (None, None) for table and stat chart types.
        """
        if chart_type not in (CHART_LINE, CHART_BAR):
            return None, None

        cols = df.columns.tolist()
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        date_cols = [c for c in cols if self._is_date_col(df, c)]
        text_cols = [c for c in cols if c not in numeric_cols and c not in date_cols]

        if chart_type == CHART_LINE:
            x = date_cols[0] if date_cols else (text_cols[0] if text_cols else cols[0])
            y = numeric_cols[0] if numeric_cols else cols[-1]
            return x, y

        if chart_type == CHART_BAR:
            x = text_cols[0] if text_cols else cols[0]
            y = numeric_cols[0] if numeric_cols else cols[-1]
            return x, y

        return None, None

    def _is_date_col(self, df: pd.DataFrame, col: str) -> bool:
        """Check if a column contains date or datetime values."""
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        col_lower = col.lower()
        return any(
            k in col_lower
            for k in ["date", "time", "month", "year", "week", "day", "period"]
        )

    # ── Data serialisation ─────────────────────────────────────────────────────

    def _to_chart_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to a JSON-serialisable list of dicts.
        Truncates to 500 rows maximum for chart performance.
        Handles Timestamp, Decimal, and other non-JSON types.
        """
        display_df = df.head(500)
        records = []
        for _, row in display_df.iterrows():
            record = {}
            for col in display_df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                elif hasattr(val, "isoformat"):
                    record[col] = val.isoformat()
                elif isinstance(val, (int, float, str, bool)):
                    record[col] = val
                else:
                    record[col] = str(val)
            records.append(record)
        return records

    # ── Narration ──────────────────────────────────────────────────────────────

    def _narrate(
        self,
        question: str,
        df: pd.DataFrame,
        metric_used: Optional[Dict],
        chart_type: str,
    ) -> str:
        """
        Call Groq to generate a 2-3 sentence plain English narration.
        Passes a compressed result summary — not the full DataFrame.
        """
        result_summary = self._summarise_df(df, chart_type)

        metric_context = ""
        if metric_used:
            metric_context = (
                f"This answer uses the confirmed metric '{metric_used['name']}': "
                f"{metric_used['description']}"
            )

        prompt = _NARRATION_USER.format(
            question=question,
            metric_context=metric_context,
            row_count=len(df),
            result_summary=result_summary,
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": _NARRATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # slight warmth — this is natural language, not SQL
                max_tokens=200,  # 2-3 sentences max
            )
            return response.choices[0].message.content.strip()
        except Exception as _e:
            # Never let narration failure break the pipeline
            return self._fallback_narration(df, question)

    def _summarise_df(self, df: pd.DataFrame, chart_type: str) -> str:
        """
        Build a compact text summary of the DataFrame for the narration prompt.
        Avoids sending hundreds of rows to the LLM.
        Strategy:
          - stat:  the single value
          - line/bar with ≤10 rows: all rows
          - line/bar with >10 rows: first 5, last 5, min, max, mean of numeric cols
          - table: column names + first 5 rows + row count
        """
        if chart_type == CHART_STAT:
            val = df.iloc[0, 0]
            col = df.columns[0]
            return f"{col}: {val}"

        lines = []

        if len(df) <= 10:
            lines.append(df.to_string(index=False, max_rows=10))
        else:
            lines.append(f"First 5 rows:\n{df.head(5).to_string(index=False)}")
            lines.append(f"\nLast 5 rows:\n{df.tail(5).to_string(index=False)}")

            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                stats = numeric_cols.agg(["min", "max", "mean"]).round(2)
                lines.append(f"\nNumeric summary:\n{stats.to_string()}")

        if len(df) > 10:
            lines.append(f"\nTotal rows: {len(df)}")

        return "\n".join(lines)

    def _fallback_narration(self, df: pd.DataFrame, question: str) -> str:
        """
        Rule-based narration when the LLM call fails.
        Never surfaces an error to the user.
        """
        n_rows = len(df)
        if n_rows == 1 and len(df.columns) == 1:
            val = df.iloc[0, 0]
            col = df.columns[0]
            return f"The {col.replace('_', ' ')} is {val}."

        return (
            f"The query returned {n_rows} record{'s' if n_rows != 1 else ''}. "
            f"The results are shown below."
        )

    def _empty_result(self, question: str) -> ExplainerResult:
        """Return a structured empty result when the DataFrame is None."""
        return ExplainerResult(
            narration=(
                "No results were found for this question. "
                "This may be because the filters are too specific, "
                "or the data does not contain records matching this criteria. "
                "Try rephrasing or broadening the question."
            ),
            chart_type=CHART_TABLE,
            chart_data=[],
            columns=[],
            metric_used=None,
            row_count=0,
            x_key=None,
            y_key=None,
        )

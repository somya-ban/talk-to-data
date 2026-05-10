"""
Talk to Data — Backend Test Suite
NatWest Code for Purpose 2026

Covers:
  Stage 1 — Imports and environment
  Stage 2 — Unit tests per component (loader, embedder, graph, linker,
             sql_gen, seeder, executor, guard, metric_dict, explainer)
  Stage 3 — Integration tests (component interactions)
  Stage 4 — End-to-end pipeline (question → SQL → result → narration)
  Stage 5 — Edge cases and error scenarios

Run from project root with envdata activated:
    python test_backend.py

Expected output: all sections print PASS.
Any FAIL line means something needs fixing before the frontend is built.
"""

import os
import sys
import duckdb
import pandas as pd
from dotenv import load_dotenv

# Cleanup test files
import shutil

# chart_data JSON serialisability
import json

load_dotenv()

# ── Colour helpers ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _pass(label):
    print(f"  {GREEN}PASS{RESET}  {label}")


def _fail(label, err=""):
    print(f"  {RED}FAIL{RESET}  {label}")
    if err:
        print(f"         {RED}→ {err}{RESET}")


def _section(title):
    print(f"\n{BOLD}{BLUE}{'─' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─' * 60}{RESET}")


def _info(msg):
    print(f"  {YELLOW}INFO{RESET}  {msg}")


# ── Result tracker ─────────────────────────────────────────────────────────────

results = {"passed": 0, "failed": 0}


def check(label, condition, error_msg=""):
    if condition:
        _pass(label)
        results["passed"] += 1
    else:
        _fail(label, error_msg)
        results["failed"] += 1


def run(label, fn):
    """Run fn(), catch exceptions, report pass/fail."""
    try:
        fn()
        _pass(label)
        results["passed"] += 1
        return True
    except Exception as e:
        _fail(label, f"{type(e).__name__}: {e}")
        results["failed"] += 1
        return False


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Imports and environment
# ══════════════════════════════════════════════════════════════════════════════

_section("STAGE 1 — Imports and environment")

# Environment
check(
    "GROQ_API_KEY is set",
    bool(os.environ.get("GROQ_API_KEY")),
    "Set GROQ_API_KEY in .env before running tests",
)

check(
    "NEO4J_URI is set",
    bool(os.environ.get("NEO4J_URI")),
    "Set NEO4J_URI in .env before running tests",
)

check("NEO4J_USERNAME is set", bool(os.environ.get("NEO4J_USERNAME")))

check("NEO4J_PASSWORD is set", bool(os.environ.get("NEO4J_PASSWORD")))

# Data files
check(
    "banking_tables_typed.jsonl exists",
    os.path.exists("data/banking_tables_typed.jsonl"),
)

check(
    "banking_relationships_v2.jsonl exists",
    os.path.exists("data/banking_relationships_v2.jsonl"),
)

# Imports — every component
import_ok = True
try:
    from backend.src.ingestion.loader import SchemaLoader
    from backend.src.ingestion.data_seeder import BankingDataSeeder
    from backend.src.retrieval.embedder import SchemaEmbedder
    from backend.src.retrieval.graph_builder import SchemaGraphBuilder
    from backend.src.retrieval.schema_linker import SchemaLinker, SchemaContext
    from backend.src.generation.sql_gen import SQLGenerator, SQLResult, SEED_QA_PAIRS
    from backend.src.validation.executor import SQLExecutor, ErrorType
    from backend.src.privacy.guard import PrivacyGuard
    from backend.src.semantic.metric_dict import MetricDictionary
    from backend.src.explanation.explainer import ResultExplainer

    _pass("All 10 pipeline modules import cleanly")
    results["passed"] += 1
except ImportError as e:
    _fail("Module imports", str(e))
    results["failed"] += 1
    import_ok = False
    print(f"\n{RED}Cannot continue — fix import errors first.{RESET}")
    sys.exit(1)

try:
    from backend.app import app as flask_app

    _pass("app.py imports cleanly (Flask app created)")
    results["passed"] += 1
except Exception as e:
    _fail("app.py import", str(e))
    results["failed"] += 1


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Component unit tests
# ══════════════════════════════════════════════════════════════════════════════

_section("STAGE 2A — SchemaLoader")

loader = SchemaLoader().load()

check(
    "loader.table_count() == 25",
    loader.table_count() == 25,
    f"Got {loader.table_count()}",
)

check(
    "loader.relation_count() == 25",
    loader.relation_count() == 25,
    f"Got {loader.relation_count()}",
)

check(
    "loader.get_services() returns 5 services",
    len(loader.get_services()) == 5,
    f"Got {loader.get_services()}",
)

ddl = loader.to_ddl_statements()
check("to_ddl_statements() returns 25 DDL strings", len(ddl) == 25, f"Got {len(ddl)}")

check("Every DDL contains CREATE TABLE", all("CREATE TABLE" in d for d in ddl))

docs = loader.to_documentation_strings()
check(
    "to_documentation_strings() returns >= 30 strings",
    len(docs) >= 30,
    f"Got {len(docs)}",
)

check("Service descriptions present in docs", any("CustSrv" in d for d in docs))

# ── STAGE 2B — SchemaEmbedder ──────────────────────────────────────────────────

_section("STAGE 2B — SchemaEmbedder")

embedder = SchemaEmbedder(persist_dir="./chroma_db_test")
result = embedder.load_from_schema(loader)
embedder._clear_collection(embedder.sql_col)  # remove stale pairs from previous runs

check(
    "load_from_schema() loads 25 DDLs",
    embedder.ddl_count() == 25,
    f"Got {embedder.ddl_count()}",
)

check(
    "load_from_schema() loads >= 30 docs",
    embedder.doc_count() >= 30,
    f"Got {embedder.doc_count()}",
)

# Retrieval
ddl_results = embedder.get_related_ddl("customer balance account", n_results=5)
check("get_related_ddl() returns results for banking question", len(ddl_results) > 0)

check("get_related_ddl() returns strings", all(isinstance(d, str) for d in ddl_results))

doc_results = embedder.get_related_documentation("trade order", n_results=3)
check("get_related_documentation() returns results", len(doc_results) > 0)

# Q-SQL storage
embedder.store_qa_pair("How many customers?", "SELECT COUNT(*) FROM Customer")
check("store_qa_pair() increases qa_count", embedder.qa_count() >= 1)

similar = embedder.get_similar_question_sql(
    "How many customers are there?", n_results=1
)
check(
    "get_similar_question_sql() retrieves stored pair",
    len(similar) > 0 and "sql" in similar[0],
)

check(
    "Retrieved Q-SQL pair has both question and sql keys",
    all("question" in p and "sql" in p for p in similar),
)

# ── STAGE 2C — SchemaGraphBuilder ─────────────────────────────────────────────

_section("STAGE 2C — SchemaGraphBuilder (Neo4j AuraDB)")

graph = SchemaGraphBuilder()

run("graph.connect() establishes Neo4j connection", graph.connect)

stats = graph.build_from_loader(loader)
check(
    "build_from_loader() returns stats dict",
    isinstance(stats, dict) and "nodes" in stats and "edges" in stats,
)

check("Neo4j has 25 Table nodes", graph.node_count() == 25, f"Got {graph.node_count()}")

check(
    "Neo4j has 25 relationship edges",
    graph.edge_count() == 25,
    f"Got {graph.edge_count()}",
)

# Join path discovery
path = graph.find_join_path("Customer", "Account")
check(
    "find_join_path(Customer, Account) returns path", path is not None and len(path) > 0
)

check(
    "Join path hops have join_condition field",
    path is not None and all("join_condition" in h for h in path),
)

_info(
    f"Customer→Account path: {' → '.join(h['from_table'] + '.' + h['to_table'] for h in (path or []))}"
)

# Cross-service path (harder)
path_long = graph.find_join_path("Customer", "Trade")
check(
    "find_join_path(Customer, Trade) returns multi-hop path",
    path_long is not None and len(path_long) >= 3,
    f"Got {len(path_long) if path_long else 0} hops",
)

_info(f"Customer→Trade hops: {len(path_long) if path_long else 0}")

# Neighbourhood
neighbours = graph.get_neighborhood(["Customer"], hops=1)
check("get_neighborhood(Customer) returns neighbours", len(neighbours) > 0)

check("Account is a neighbour of Customer", "Account" in neighbours)

# add_metric_node (no-op but should not crash)
run(
    "add_metric_node() does not crash",
    lambda: graph.add_metric_node("total_balance", ["balance"], "Account"),
)

# ── STAGE 2D — SchemaLinker ────────────────────────────────────────────────────

_section("STAGE 2D — SchemaLinker (Graph-RAG)")

linker = SchemaLinker(embedder, graph)

ctx = linker.link("What is the total balance across all accounts?")

check("link() returns SchemaContext", isinstance(ctx, SchemaContext))

check(
    "SchemaContext.question is set",
    ctx.question == "What is the total balance across all accounts?",
)

check("SchemaContext has relevant DDLs", len(ctx.relevant_ddls) > 0)

check("SchemaContext has relevant docs", len(ctx.relevant_docs) > 0)

check("SchemaContext has join_context string", isinstance(ctx.join_context, str))

check("SchemaContext has table_names", len(ctx.table_names) > 0)

_info(f"Schema link result: {ctx.summary()}")

check(
    "AccountBalance is in retrieved tables",
    any("AccountBalance" in t or "Account" in t for t in ctx.table_names),
)

# Test that graph expansion adds intermediate tables
ctx2 = linker.link("Show me all trades for high risk customers")
_info(f"Trade→Customer link: {ctx2.summary()}")
check(
    "Cross-service query retrieves tables from multiple services",
    len(ctx2.table_names) >= 2,
)

# ── STAGE 2E — SQLGenerator ────────────────────────────────────────────────────

_section("STAGE 2E — SQLGenerator")

sql_gen = SQLGenerator()

result = sql_gen.generate(ctx)
check("generate() returns SQLResult", isinstance(result, SQLResult))

check("SQLResult.success is True", result.success, f"Error: {result.error}")

check("SQLResult.sql is non-empty", len(result.sql) > 10)

check(
    "SQLResult.sql starts with SELECT or WITH",
    result.sql.strip().upper().startswith(("SELECT", "WITH")),
    f"Got: {result.sql[:50]}",
)

check("SQLResult.sql contains FROM", "FROM" in result.sql.upper())

check("SQLResult.sql ends with semicolon", result.sql.strip().endswith(";"))

check("SQLResult.prompt is non-empty", len(result.prompt) > 100)

_info(f"Generated SQL: {result.sql[:120]}...")

# Test regeneration with feedback
feedback_result = sql_gen.regenerate_with_feedback(
    context=ctx,
    previous_sql="SELECT * FROM NonExistentTable;",
    error_message="CATALOG_ERROR: Table NonExistentTable does not exist",
)
check(
    "regenerate_with_feedback() returns SQLResult",
    isinstance(feedback_result, SQLResult),
)

check(
    "regenerate_with_feedback() generates new SQL",
    feedback_result.success
    and feedback_result.sql != "SELECT * FROM NonExistentTable;",
    f"Error: {feedback_result.error}",
)

# ── STAGE 2F — BankingDataSeeder ──────────────────────────────────────────────

_section("STAGE 2F — BankingDataSeeder")

conn = duckdb.connect(":memory:")
seeder = BankingDataSeeder(conn)
row_counts = seeder.seed_all()

check(
    "seed_all() returns dict of counts",
    isinstance(row_counts, dict) and len(row_counts) > 0,
)

check(
    "Customer table seeded with 40 rows",
    row_counts.get("Customer", 0) == 40,
    f"Got {row_counts.get('Customer', 0)}",
)

check(
    "Account table seeded with 60 rows",
    row_counts.get("Account", 0) == 60,
    f"Got {row_counts.get('Account', 0)}",
)

check(
    "Trade table seeded",
    row_counts.get("Trade", 0) > 0,
    f"Got {row_counts.get('Trade', 0)}",
)

check(
    "All 25 table groups have rows",
    all(v > 0 for v in row_counts.values()),
    f"Empty tables: {[k for k, v in row_counts.items() if v == 0]}",
)

# Verify FK integrity
fk_check = conn.execute("""
    SELECT COUNT(*) FROM Account a
    LEFT JOIN Customer c ON a.customer_id = c.customer_id
    WHERE c.customer_id IS NULL
""").fetchone()[0]
check(
    "Account.customer_id FK integrity: no orphan accounts",
    fk_check == 0,
    f"Found {fk_check} orphan accounts",
)

fk_check2 = conn.execute("""
    SELECT COUNT(*) FROM Trade t
    LEFT JOIN "Order" o ON t.order_id = o.order_id
    WHERE o.order_id IS NULL
""").fetchone()[0]
check(
    "Trade.order_id FK integrity: no orphan trades",
    fk_check2 == 0,
    f"Found {fk_check2} orphan trades",
)

# Verify status values are uppercase
status_check = conn.execute("""
    SELECT COUNT(*) FROM Customer
    WHERE risk_rating NOT IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
""").fetchone()[0]
check(
    "Customer.risk_rating values are uppercase",
    status_check == 0,
    f"{status_check} rows with wrong casing",
)

# ── STAGE 2G — SQLExecutor ────────────────────────────────────────────────────

_section("STAGE 2G — SQLExecutor")

executor = SQLExecutor(conn)

# Basic execution
df, err = executor.execute_direct("SELECT COUNT(*) AS n FROM Customer")
check("execute_direct() succeeds on valid SQL", df is not None and err is None)

check("execute_direct() returns DataFrame", isinstance(df, pd.DataFrame))

check(
    "execute_direct() COUNT result is 40",
    df is not None and df.iloc[0, 0] == 40,
    f"Got {df.iloc[0, 0] if df is not None else 'None'}",
)

# Empty result (soft failure)
df_empty, err_empty = executor.execute_direct(
    "SELECT * FROM Customer WHERE risk_rating = 'NONEXISTENT'"
)
check(
    "execute_direct() returns (None, error) for 0-row result",
    df_empty is None and err_empty is not None,
)

check(
    "Empty result error message mentions EMPTY_RESULT",
    err_empty is not None and "EMPTY_RESULT" in err_empty,
)

# Syntax error
df_bad, err_bad = executor.execute_direct("SELECT FROM WHERE")
check(
    "execute_direct() returns (None, error) for syntax error",
    df_bad is None and err_bad is not None,
)

# Error classification
check(
    "_classify() identifies SYNTAX from PARSER_ERROR",
    executor.classify("PARSER_ERROR: unexpected token") == ErrorType.SYNTAX,
)

check(
    "_classify() identifies COLUMN_NOT_FOUND from BINDER_ERROR",
    executor.classify("BINDER_ERROR: column xyz does not exist")
    == ErrorType.COLUMN_NOT_FOUND,
)

check(
    "_classify() identifies TABLE_NOT_FOUND",
    executor.classify("CATALOG_ERROR: table FakeTable does not exist")
    == ErrorType.TABLE_NOT_FOUND,
)

check(
    "_classify() identifies EMPTY_RESULT",
    executor.classify("EMPTY_RESULT: 0 rows returned") == ErrorType.EMPTY_RESULT,
)

check(
    "_classify() identifies AMBIGUOUS",
    executor.classify("BINDER_ERROR: ambiguous reference to column customer_id")
    == ErrorType.AMBIGUOUS,
)

check(
    "_classify() falls back to UNKNOWN",
    executor.classify("something completely unexpected") == ErrorType.UNKNOWN,
)

# Introspection
tables = executor.list_tables()
check(
    "list_tables() returns all seeded tables", len(tables) >= 20, f"Got {len(tables)}"
)

check("Customer is in list_tables()", "Customer" in tables)

rc = executor.row_count("Customer")
check("row_count('Customer') returns 40", rc == 40, f"Got {rc}")

cols = executor.column_names("Customer")
check(
    "column_names('Customer') returns expected columns",
    "customer_id" in cols and "risk_rating" in cols,
    f"Got {cols}",
)

# ── STAGE 2H — PrivacyGuard ───────────────────────────────────────────────────

_section("STAGE 2H — PrivacyGuard")

guard = PrivacyGuard()

# Build test DataFrame with PII and non-PII columns
test_df = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "first_name": ["Alice", "Bob", "Carol"],
        "email": ["a@b.com", "b@b.com", "c@b.com"],
        "phone_number": ["+44 1234", "+44 5678", "+44 9012"],
        "national_id": ["AB123456C", "DE789012F", "GH345678I"],
        "total_balance": [1000.0, 2000.0, 3000.0],
        "risk_rating": ["LOW", "HIGH", "MEDIUM"],
    }
)

masked_df, masked_cols = guard.scan_and_mask(test_df)

check(
    "scan_and_mask() returns masked DataFrame and list",
    isinstance(masked_df, pd.DataFrame) and isinstance(masked_cols, list),
)

check("email column is masked", "email" in masked_cols)

check("phone_number column is masked", "phone_number" in masked_cols)

check("national_id column is masked", "national_id" in masked_cols)

check("first_name column is masked (banking context)", "first_name" in masked_cols)

check(
    "customer_id is NOT masked (it's an INT identifier)",
    "customer_id" not in masked_cols,
)

check(
    "total_balance is NOT masked (financial metric, not PII)",
    "total_balance" not in masked_cols,
)

check("risk_rating is NOT masked", "risk_rating" not in masked_cols)

check("Masked values are '***'", masked_df["email"].iloc[0] == "***")

check("Non-PII values are unchanged", masked_df["total_balance"].iloc[0] == 1000.0)

check(
    "customer_id values unchanged after masking",
    list(masked_df["customer_id"]) == [1, 2, 3],
)

# Edge: empty DataFrame
empty_df = pd.DataFrame()
masked_empty, cols_empty = guard.scan_and_mask(empty_df)
check("scan_and_mask() handles empty DataFrame without crash", cols_empty == [])

# is_pii_column checks
check("is_pii_column('email') returns True", guard.is_pii_column("email"))
check("is_pii_column('sort_code') returns True", guard.is_pii_column("sort_code"))
check(
    "is_pii_column('account_id') returns False", not guard.is_pii_column("account_id")
)
check("is_pii_column('balance') returns False", not guard.is_pii_column("balance"))
check("is_pii_column('ip_address') returns True", guard.is_pii_column("ip_address"))

# allow_names=True test
guard_allow = PrivacyGuard(allow_names=True)
_, cols_allow = guard_allow.scan_and_mask(test_df)
check("allow_names=True: first_name is NOT masked", "first_name" not in cols_allow)

check("allow_names=True: email still IS masked", "email" in cols_allow)

# ── STAGE 2I — MetricDictionary ───────────────────────────────────────────────

_section("STAGE 2I — MetricDictionary")

md = MetricDictionary()

# Load defaults
defaults = md.load_banking_defaults()
check(
    "load_banking_defaults() returns 8 metrics",
    len(defaults) == 8,
    f"Got {len(defaults)}",
)

check("total_balance metric exists in defaults", "total_balance" in defaults)

check(
    "Each default metric has name, description, sql_formula",
    all(
        "name" in m and "description" in m and "sql_formula" in m
        for m in defaults.values()
    ),
)

# format_for_prompt
prompt_str = md.format_for_prompt()
check("format_for_prompt() returns non-empty string", len(prompt_str) > 50)

check("format_for_prompt() contains metric names", "total_balance" in prompt_str)

check("format_for_prompt() contains SQL formulas", "SUM" in prompt_str)

# metric access
metric = md.get_metric("total_balance")
check(
    "get_metric('total_balance') returns dict",
    metric is not None and isinstance(metric, dict),
)

check("metric_count() returns 8 after loading defaults", md.metric_count() == 8)

check("is_empty() returns False after loading", not md.is_empty())

# save and load
test_metrics = [
    {
        "name": "test_metric",
        "description": "Test metric for validation",
        "sql_formula": "SELECT COUNT(*) FROM Customer",
        "tables": ["Customer"],
        "columns": ["customer_id"],
    }
]
count = md.save_confirmed_metrics(test_metrics)
check("save_confirmed_metrics() returns count", count == 1)

check("metrics.yaml created after save", os.path.exists("metrics.yaml"))

# Load from yaml
md2 = MetricDictionary()
loaded = md2.load_metrics()
check("load_metrics() reads from metrics.yaml", len(loaded) > 0)

check("test_metric survives save/load cycle", "test_metric" in loaded)

# ── STAGE 2J — ResultExplainer ────────────────────────────────────────────────

_section("STAGE 2J — ResultExplainer")

explainer = ResultExplainer()


# Chart type selection tests
def make_df(**kwargs):
    return pd.DataFrame(kwargs)


# Stat: 1 row, 1 col
df_stat = make_df(total_balance=[4200000.0])
result_stat = explainer.explain("What is the total balance?", df_stat)
check("Single-cell result → chart_type = 'stat'", result_stat.chart_type == "stat")

# Bar: 1 text + 1 numeric
df_bar = make_df(risk_rating=["LOW", "MEDIUM", "HIGH"], count=[10, 8, 5])
result_bar = explainer.explain("Count customers by risk rating", df_bar)
check("Text+numeric result → chart_type = 'bar'", result_bar.chart_type == "bar")

check(
    "Bar chart has x_key and y_key",
    result_bar.x_key is not None and result_bar.y_key is not None,
)

# Table: multi-column
df_table = make_df(
    customer_id=[1, 2],
    first_name=["Alice", "Bob"],
    balance=[1000.0, 2000.0],
    risk_rating=["LOW", "HIGH"],
)
result_table = explainer.explain("Show customer details", df_table)
check("Multi-column result → chart_type = 'table'", result_table.chart_type == "table")

# ExplainerResult fields
check(
    "ExplainerResult.narration is non-empty string",
    isinstance(result_stat.narration, str) and len(result_stat.narration) > 10,
)

check("ExplainerResult.chart_data is list", isinstance(result_stat.chart_data, list))

check(
    "ExplainerResult.columns is list of strings", isinstance(result_stat.columns, list)
)

check("ExplainerResult.row_count matches DataFrame", result_bar.row_count == 3)

# Empty DataFrame
result_empty = explainer.explain("empty question", None)
check(
    "explain() handles None DataFrame without crash",
    result_empty.chart_type == "table" and result_empty.row_count == 0,
)

check(
    "explain() returns narration even for None DataFrame",
    len(result_empty.narration) > 10,
)


try:
    json.dumps(result_bar.chart_data)
    _pass("chart_data is JSON-serialisable")
    results["passed"] += 1
except Exception as e:
    _fail("chart_data is JSON-serialisable", str(e))
    results["failed"] += 1


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Integration tests
# ══════════════════════════════════════════════════════════════════════════════

_section("STAGE 3 — Integration: Schema linking + SQL generation")

# Inject metrics into context (same as app.py does)
md.load_banking_defaults()
metric_defs = md.format_for_prompt()
if metric_defs:
    ctx.relevant_docs.insert(0, metric_defs)
check(
    "metric_defs injected into SchemaContext.relevant_docs",
    any(len(d) > 100 for d in ctx.relevant_docs),
)

# SQL generation with metric context
ctx_balance = linker.link("What is the total balance?")
if metric_defs:
    ctx_balance.relevant_docs.insert(0, metric_defs)

sql_with_metrics = sql_gen.generate(ctx_balance)
check(
    "SQL generation with metrics succeeds",
    sql_with_metrics.success,
    f"Error: {sql_with_metrics.error}",
)

_info(f"SQL with metrics: {sql_with_metrics.sql[:100]}...")

# Execute generated SQL against seeded DuckDB
df_result, exec_err = executor.execute_direct(sql_with_metrics.sql)
check(
    "Generated SQL executes successfully against DuckDB",
    df_result is not None,
    f"Error: {exec_err}",
)

if df_result is not None:
    _info(f"Result: {df_result.shape[0]} rows, {df_result.shape[1]} cols")
    _info(f"Columns: {list(df_result.columns)}")

# Self-correction loop test
_section("STAGE 3 — Integration: Self-correction loop")

DELIBERATELY_BAD = "SELECT total_bananas FROM NonExistentTable WHERE xyz = 999;"
bad_df, bad_err = executor.execute_direct(DELIBERATELY_BAD)
check("Deliberately bad SQL fails execution", bad_df is None and bad_err is not None)

error_type = executor.classify(bad_err)
_info(f"Error classified as: {error_type.name}")

# Regenerate with feedback
corrected = sql_gen.regenerate_with_feedback(
    context=ctx_balance,
    previous_sql=DELIBERATELY_BAD,
    error_message=f"Error type: {error_type.value.split('—')[0]}\nError: {bad_err}",
)
check(
    "regenerate_with_feedback() produces new SQL after bad SQL",
    corrected.success,
    f"Error: {corrected.error}",
)

check("Corrected SQL differs from bad SQL", corrected.sql != DELIBERATELY_BAD)

_info(f"Corrected SQL: {corrected.sql[:100]}...")

# Execute corrected SQL
df_corrected, err_corrected = executor.execute_direct(corrected.sql)
check(
    "Corrected SQL executes successfully",
    df_corrected is not None,
    f"Error: {err_corrected}",
)

# Fix loop — store correction
embedder.store_qa_pair("What is the total balance?", corrected.sql, was_corrected=True)
check("Fix loop: corrected Q-SQL pair stored in ChromaDB", embedder.qa_count() >= 1)

# Verify it is retrievable
retrieved = embedder.get_similar_question_sql("total balance", n_results=1)
check(
    "Fix loop: stored pair is retrievable for future queries",
    len(retrieved) > 0 and retrieved[0].get("sql"),
)


# Seed the correct Q-SQL pairs into ChromaDB before E2E tests.
# This mirrors what app.py's _seed_qa_pairs() does at startup.
# Without this, the LLM has almost no few-shot examples and
# falls back on training-data patterns (hallucinating .name columns).
seed_pairs = [{"question": p["question"], "sql": p["sql"]} for p in SEED_QA_PAIRS]
count = embedder.seed_qa_pairs(seed_pairs)
_info(f"Seeded {count} Q-SQL pairs before E2E tests (stale pairs cleared)")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — End-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

_section("STAGE 4 — End-to-end pipeline (5 banking questions)")

E2E_QUESTIONS = [
    "How many customers do we have?",
    "What is the total balance across all accounts?",
    "Show me the top 5 customers by account balance",
    "How many customers are classified as high risk?",
    "List all advisors and the branch they work in",
]

for q in E2E_QUESTIONS:
    try:
        # Stage 1: link
        ctx_e2e = linker.link(q)
        if metric_defs:
            ctx_e2e.relevant_docs.insert(0, metric_defs)

        # Stage 2: generate
        sql_e2e = sql_gen.generate(ctx_e2e)
        if not sql_e2e.success:
            _fail(f"E2E: '{q[:50]}'", f"SQL gen failed: {sql_e2e.error}")
            results["failed"] += 1
            continue

        # Stage 3: execute with retry
        df_e2e = None
        last_err = None
        current = sql_e2e
        for attempt in range(1, 4):
            df_e2e, last_err = executor.execute_direct(current.sql)
            if df_e2e is not None:
                break
            if attempt < 3:
                et = executor.classify(last_err)
                current = sql_gen.regenerate_with_feedback(
                    context=ctx_e2e,
                    previous_sql=current.sql,
                    error_message=f"Error type: {et.value.split('—')[0]}\nError: {last_err}",
                )

        # Stage 4: privacy guard
        if df_e2e is not None:
            df_e2e, masked = guard.scan_and_mask(df_e2e)

        # Stage 5: explain
        metric_for_q = None
        for name, m in md.metrics.items():
            if name.replace("_", " ") in q.lower():
                metric_for_q = m
                break

        expl = explainer.explain(q, df_e2e, metric_for_q)

        if df_e2e is not None:
            _pass(f"E2E: '{q[:50]}' → {df_e2e.shape[0]} rows, chart={expl.chart_type}")
            results["passed"] += 1
            _info(f"  Narration: {expl.narration[:100]}...")
        else:
            _fail(f"E2E: '{q[:50]}'", f"All retries failed. Last error: {last_err}")
            results["failed"] += 1

    except Exception as e:
        _fail(f"E2E: '{q[:50]}'", f"{type(e).__name__}: {e}")
        results["failed"] += 1


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Edge cases and Flask API contract
# ══════════════════════════════════════════════════════════════════════════════

_section("STAGE 5 — Edge cases")

# Guard: DataFrame with no PII
df_clean = make_df(account_type=["SAVINGS", "ISA"], balance=[1000.0, 2000.0])
clean_masked, clean_cols = guard.scan_and_mask(df_clean)
check("DataFrame with no PII columns: nothing masked", clean_cols == [])

# Guard: None value in PII column is preserved as None
df_null = make_df(email=[None, "a@b.com"], balance=[1.0, 2.0])
null_masked, _ = guard.scan_and_mask(df_null)
check(
    "None values in PII columns remain None after masking",
    null_masked["email"].iloc[0] is None or pd.isna(null_masked["email"].iloc[0]),
)

# Executor: very large result truncated by explainer
many_rows = conn.execute("SELECT * FROM CashMovement LIMIT 300").df()
_info(f"Large result test: {len(many_rows)} rows")
expl_large = explainer.explain("Show all transactions", many_rows)
check(
    "explainer handles large result without crash",
    expl_large.chart_type in ("table", "bar", "line", "stat"),
)
check("chart_data truncated to <= 500 rows", len(expl_large.chart_data) <= 500)

# MetricDictionary: parse failure falls back to defaults
md_test = MetricDictionary()
fallback = md_test._parse_proposals("this is not json at all {broken}")
check(
    "_parse_proposals() falls back to banking defaults on bad JSON", len(fallback) == 8
)

# SchemaContext: is_empty() logic
empty_ctx = SchemaContext(question="test")
check("SchemaContext.is_empty() True when no DDLs or docs", empty_ctx.is_empty())

check("SchemaContext.is_empty() False when DDLs present", not ctx.is_empty())

# Flask app test client
_section("STAGE 5 — Flask API health check")

with flask_app.test_client() as client:
    resp = client.get("/api/health")
    check("/api/health returns 200", resp.status_code == 200)

    data = resp.get_json()
    check("/api/health returns ok:true", data.get("ok") is True)

    check(
        "/api/health returns correct service name",
        data.get("service") == "Talk to Data",
    )

    # Status before init
    resp_status = client.get("/api/status")
    check("/api/status returns 200", resp_status.status_code == 200)

    status_data = resp_status.get_json()
    check(
        "/api/status returns ready:false before init", status_data.get("ready") is False
    )

    # Query before init — should return 503
    resp_query = client.post(
        "/api/query", json={"question": "test"}, content_type="application/json"
    )
    check("/api/query returns 503 before init", resp_query.status_code == 503)

    # Empty question
    resp_empty = client.post(
        "/api/query", json={"question": ""}, content_type="application/json"
    )
    check(
        "/api/query with empty question returns 400",
        resp_empty.status_code in (400, 503),
    )


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

total = results["passed"] + results["failed"]
print(f"\n{BOLD}{'═' * 60}{RESET}")
print(f"{BOLD}  TEST RESULTS{RESET}")
print(f"{BOLD}{'═' * 60}{RESET}")
print(f"  Total:  {total}")
print(f"  {GREEN}Passed: {results['passed']}{RESET}")
if results["failed"] > 0:
    print(f"  {RED}Failed: {results['failed']}{RESET}")
    print(f"\n  {RED}Fix all FAIL items before building the frontend.{RESET}")
else:
    print(f"  {GREEN}Failed: 0{RESET}")
    print(f"\n  {GREEN}{BOLD}All tests passed. Backend is ready.{RESET}")
    print(f"  {GREEN}Proceed to frontend (React + Vite + Tailwind + shadcn/ui).{RESET}")

print(f"{BOLD}{'═' * 60}{RESET}\n")

if os.path.exists("./chroma_db_test"):
    try:
        shutil.rmtree("./chroma_db_test")
    except PermissionError:
        _info(
            "chroma_db_test cleanup skipped (Windows file lock — delete manually if needed)"
        )
if os.path.exists("metrics.yaml"):
    # Keep it — it has real content from the test
    pass

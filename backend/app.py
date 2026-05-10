"""
Talk to Data — Flask REST API
NatWest Code for Purpose 2026 — Theme 1: Seamless Self-Service Intelligence

This file is the orchestrator. It wires every pipeline component together
and exposes a clean REST API that the React frontend consumes.

Architecture:
    React (Vite + Tailwind + shadcn/ui)
        ↕  fetch() / JSON
    Flask API (this file)
        ↕  Python imports
    Pipeline components:
        loader.py → embedder.py → graph_builder.py → schema_linker.py
        metric_dict.py → sql_gen.py → executor.py → guard.py → explainer.py

Endpoint design follows the pattern used by Vanna's vanna-flask and
Wren AI's service layer — clean REST, structured JSON responses,
error messages surfaced as JSON (never as HTML 500 pages).

Pipeline state is stored as module-level globals.
For a single-user hackathon demo this is correct.
For multi-user production you would use a proper session/state store.

CORS is enabled for all origins during development.
The React dev server runs on localhost:5173, Flask on localhost:5000.

API contract:

  POST /api/init
    → initialises all pipeline components
    → seeds DuckDB with synthetic banking data
    → returns component status and row counts

  GET  /api/status
    → returns whether pipeline is ready to accept queries

  POST /api/metrics/propose
    → calls MetricDictionary.propose_metrics() via Groq
    → returns list of proposed metric definitions for user review

  POST /api/metrics/confirm
    body: { metrics: [...] }
    → saves confirmed metrics to metrics.yaml + ChromaDB
    → returns count of confirmed metrics

  GET  /api/metrics
    → returns currently confirmed metrics

  POST /api/query
    body: { question: "..." }
    → runs the full 5-stage pipeline
    → returns structured answer JSON (see QueryResponse below)

  POST /api/fix
    body: { question: "...", corrected_sql: "...", original_sql: "..." }
    → stores corrected Q-SQL pair in ChromaDB (fix loop)
    → returns confirmation

  GET  /api/history
    → returns last 20 query results (for query history panel in UI)

  DELETE /api/history
    → clears query history

Response envelope:
  Every response is JSON with a top-level "ok" boolean.
  On error: { ok: false, error: "message" }
  On success: { ok: true, ...data }
"""

import os
import traceback
import duckdb
from datetime import datetime
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
from dotenv import load_dotenv
from pathlib import Path

# ── Pipeline imports ───────────────────────────────────────────────────────────
from src.ingestion.loader import SchemaLoader
from src.ingestion.data_seeder import BankingDataSeeder
from src.retrieval.embedder import SchemaEmbedder
from src.retrieval.graph_builder import SchemaGraphBuilder
from src.retrieval.schema_linker import SchemaLinker
from src.semantic.metric_dict import MetricDictionary
from src.generation.sql_gen import SQLGenerator, SEED_QA_PAIRS
from src.validation.executor import SQLExecutor
from src.privacy.guard import PrivacyGuard
from src.explanation.explainer import ResultExplainer

load_dotenv()

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # allow React dev server


# ── Pipeline state (module-level, initialised once) ───────────────────────────

_state = {
    "initialised": False,
    "loader": None,
    "embedder": None,
    "graph": None,
    "linker": None,
    "metric_dict": None,
    "sql_gen": None,
    "executor": None,
    "guard": PrivacyGuard(),
    "explainer": None,
    "conn": None,
    "query_history": [],  # list of QueryRecord dicts
}

MAX_HISTORY = 20
MAX_RETRIES = 3


# ── Helper: require initialisation ────────────────────────────────────────────


def _require_init():
    """Return error response if pipeline not ready, else None."""
    if not _state["initialised"]:
        return jsonify(
            {
                "ok": False,
                "error": "Pipeline not initialised. Call POST /api/init first.",
            }
        ), 503
    return None


def _get_api_key():
    """Get Groq API key from request header or environment."""
    return request.headers.get("X-Groq-Api-Key") or os.environ.get("GROQ_API_KEY", "")


# ── /api/init ──────────────────────────────────────────────────────────────────


@app.route("/api/init", methods=["POST"])
def init_pipeline():
    """
    Initialise all pipeline components and seed DuckDB.

    This is called once when the user first opens the app (or after reset).
    The React frontend shows a loading screen while this runs.

    Expected body (optional):
      { "groq_api_key": "..." }
    Can also be passed as X-Groq-Api-Key header or GROQ_API_KEY env var.

    Steps:
      A: Load JSONL schema (25 tables, 25 relationships)
      B: Build Neo4j schema graph
      C: Embed DDL + docs into ChromaDB
      D: Create DuckDB in-memory connection
      E: Seed DuckDB with synthetic banking data (50-100 rows/table)
      F: Initialise remaining components
    """
    try:
        body = request.get_json(silent=True) or {}
        api_key = body.get("groq_api_key") or _get_api_key()

        if not api_key:
            return jsonify(
                {
                    "ok": False,
                    "error": "GROQ_API_KEY is required. Pass in request body or X-Groq-Api-Key header.",
                }
            ), 400

        os.environ["GROQ_API_KEY"] = api_key

        # ── Step A: Load schema ────────────────────────────────────────────────
        loader = SchemaLoader(
            tables_path="data/banking_tables_typed.jsonl",
            relations_path="data/banking_relationships_v2.jsonl",
        ).load()

        # ── Step B: Build graph ────────────────────────────────────────────────
        graph = SchemaGraphBuilder()
        graph.connect()
        graph.build_from_loader(loader)

        # ── Step C: ChromaDB embeddings ────────────────────────────────────────
        embedder = SchemaEmbedder(persist_dir="./chroma_db")
        embedder.load_from_schema(loader)

        # Seed Q-SQL pairs from canonical SEED_QA_PAIRS in sql_gen.py
        _seed_qa_pairs(embedder)

        # Re-embed confirmed metrics into ChromaDB documentation collection.
        # load_from_schema() clears the doc collection — any previously
        # confirmed metrics stored via add_documentation() are wiped.
        # We reload from metrics.yaml and re-embed so semantic retrieval works.
        if Path("metrics.yaml").exists():
            temp_md = MetricDictionary()
            loaded = temp_md.load_metrics()
            for name, m in loaded.items():
                doc_string = (
                    f"Metric: {name}\n"
                    f"Description: {m['description']}\n"
                    f"Formula: {m['sql_formula']}\n"
                    f"Tables: {', '.join(m['tables'])}"
                )
                try:
                    embedder.add_documentation(doc_string)
                except Exception:
                    pass

        # ── Step D + E: DuckDB + synthetic data ────────────────────────────────
        conn = duckdb.connect(":memory:")
        seeder = BankingDataSeeder(conn)
        row_counts = seeder.seed_all()

        # ── Step F: Initialise remaining components ────────────────────────────
        linker = SchemaLinker(embedder, graph)
        metric_dict = MetricDictionary(api_key=api_key)
        sql_gen = SQLGenerator(api_key=api_key)
        executor = SQLExecutor(conn)
        explainer = ResultExplainer(api_key=api_key)

        # Load existing confirmed metrics if metrics.yaml exists
        metric_dict.load_metrics()

        # ── Update state ───────────────────────────────────────────────────────
        _state.update(
            {
                "initialised": True,
                "loader": loader,
                "embedder": embedder,
                "graph": graph,
                "linker": linker,
                "metric_dict": metric_dict,
                "sql_gen": sql_gen,
                "executor": executor,
                "explainer": explainer,
                "conn": conn,
            }
        )

        return jsonify(
            {
                "ok": True,
                "message": "Pipeline initialised successfully.",
                "schema": {
                    "tables": loader.table_count(),
                    "relations": loader.relation_count(),
                    "services": loader.get_services(),
                },
                "graph": {
                    "nodes": graph.node_count(),
                    "edges": graph.edge_count(),
                },
                "embeddings": {
                    "ddl": embedder.ddl_count(),
                    "documentation": embedder.doc_count(),
                    "qa_pairs": embedder.qa_count(),
                },
                "duckdb": {
                    "tables_seeded": len(row_counts),
                    "row_counts": row_counts,
                },
                "metrics_loaded": metric_dict.metric_count(),
            }
        )

    except Exception as e:
        return jsonify(
            {"ok": False, "error": str(e), "detail": traceback.format_exc()}
        ), 500


# ── /api/status ────────────────────────────────────────────────────────────────


@app.route("/api/status", methods=["GET"])
def status():
    """Return pipeline readiness and component summary."""
    if not _state["initialised"]:
        return jsonify({"ok": True, "ready": False})

    md = _state["metric_dict"]
    em = _state["embedder"]

    return jsonify(
        {
            "ok": True,
            "ready": True,
            "metrics_confirmed": md.metric_count() if md else 0,
            "qa_pairs": em.qa_count() if em else 0,
            "metrics_yaml_exists": os.path.exists("metrics.yaml"),
        }
    )


# ── /api/metrics/propose ───────────────────────────────────────────────────────


@app.route("/api/metrics/propose", methods=["POST"])
def propose_metrics():
    """
    Ask the LLM to propose business metric definitions from the banking schema.

    The user reviews these proposals in the UI before any query runs.
    This is the Wren AI AI Studio pattern: AI proposes, human governs.

    Returns a list of proposal objects:
      [{name, description, sql_formula, tables, columns}, ...]
    """
    guard = _require_init()
    if guard:
        return guard

    try:
        loader = _state["loader"]
        metric_dict = _state["metric_dict"]

        # Build schema context from DDL statements
        ddl_statements = loader.to_ddl_statements()
        proposals = metric_dict.propose_from_ddl_list(ddl_statements)

        return jsonify(
            {
                "ok": True,
                "proposals": proposals,
                "count": len(proposals),
            }
        )

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── /api/metrics/confirm ───────────────────────────────────────────────────────


@app.route("/api/metrics/confirm", methods=["POST"])
def confirm_metrics():
    """
    Save user-confirmed metrics to metrics.yaml and ChromaDB.

    Body: { "metrics": [{name, description, sql_formula, tables, columns}, ...] }

    After this call, every query is grounded against these definitions.
    The metric names will appear inline in the narration:
    "Using: total_balance — Sum of all account balances"
    """
    guard = _require_init()
    if guard:
        return guard

    try:
        body = request.get_json(silent=True) or {}
        metrics = body.get("metrics", [])

        if not metrics:
            return jsonify({"ok": False, "error": "No metrics provided."}), 400

        metric_dict = _state["metric_dict"]
        embedder = _state["embedder"]
        graph = _state["graph"]

        count = metric_dict.save_confirmed_metrics(metrics, embedder)

        # Add metric nodes to the schema graph
        loader = _state["loader"]
        tables = list(loader.tables.keys()) if loader else []
        first_table = tables[0].split(".")[-1] if tables else "main_table"

        for m in metrics:
            graph.add_metric_node(
                m.get("name", ""),
                m.get("columns", []),
                first_table,
            )

        return jsonify(
            {
                "ok": True,
                "message": f"{count} metric{'s' if count != 1 else ''} confirmed and saved.",
                "count": count,
            }
        )

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── /api/metrics ───────────────────────────────────────────────────────────────


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return all currently confirmed metrics."""
    guard = _require_init()
    if guard:
        return guard

    md = _state["metric_dict"]
    return jsonify(
        {
            "ok": True,
            "metrics": list(md.metrics.values()),
            "count": md.metric_count(),
        }
    )


# ── /api/query ─────────────────────────────────────────────────────────────────


@app.route("/api/query", methods=["POST"])
def query():
    """
    Main pipeline endpoint. Takes a natural language question and returns
    a fully structured answer.

    Body: { "question": "Which customers have the highest balance?" }

    Pipeline stages:
      1. Schema linker (Graph-RAG): ChromaDB vector search + Neo4j traversal
      2. SQL generation: Groq with schema context + metric definitions + few-shot
      3. Execution + self-correction: DuckDB, up to MAX_RETRIES rounds
      4. Privacy guard: mask PII columns before returning
      5. Explainer: narrate in plain English, select chart type

    Response shape:
      {
        ok: true,
        success: true,
        answer: { narration, chart_type, chart_data, columns, x_key, y_key },
        metric_used: { name, description, sql_formula } | null,
        sql: "SELECT ...",
        was_corrected: false,
        corrections: [],
        privacy: { masked_columns: [] },
        row_count: 5,
        timestamp: "2024-..."
      }
    """
    guard = _require_init()
    if guard:
        return guard

    try:
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()

        if not question:
            return jsonify({"ok": False, "error": "question is required."}), 400

        linker = _state["linker"]
        metric_dict = _state["metric_dict"]
        sql_gen = _state["sql_gen"]
        executor = _state["executor"]
        guard_obj = _state["guard"]
        explainer = _state["explainer"]
        embedder = _state["embedder"]

        # ── Stage 1: Graph-RAG schema linking ──────────────────────────────────
        retrieval = linker.link(question)

        # Inject confirmed metric definitions into SchemaContext docs.
        # Guaranteed to reach the LLM even before ChromaDB retrieval
        # returns metric docs on the very first query.
        metric_defs = metric_dict.format_for_prompt()
        if metric_defs:
            retrieval.relevant_docs.insert(0, metric_defs)

        # ── Stage 2: SQL generation ─────────────────────────────────────────────
        sql_result = sql_gen.generate(retrieval)
        current_sql = sql_result.sql
        was_corrected = False
        correction_history = []

        # ── Stage 3: Execution + self-correction loop ───────────────────────────
        df = None
        error = None

        for attempt in range(1, MAX_RETRIES + 1):
            df, error = executor.execute_direct(current_sql)

            if df is not None:
                break

            error_type = executor.classify(error)
            correction_history.append(
                {
                    "attempt": attempt,
                    "sql": current_sql,
                    "error_type": error_type.value.split("—")[0].strip(),
                    "error_message": str(error)[:300],
                }
            )

            if attempt < MAX_RETRIES:
                full_error = (
                    f"Error type: {error_type.value.split('—')[0].strip()}\n"
                    f"Error: {str(error)}"
                )
                sql_result = sql_gen.regenerate_with_feedback(
                    context=retrieval,
                    previous_sql=current_sql,
                    error_message=full_error,
                )
                current_sql = sql_result.sql if sql_result.success else current_sql
                was_corrected = True

        # ── Stage 4: Privacy guard ─────────────────────────────────────────────
        masked_cols = []
        if df is not None:
            df, masked_cols = guard_obj.scan_and_mask(df)

        # ── Identify which metric was used (for transparency layer) ────────────
        metric_used = _detect_metric_used(question, metric_dict)

        # ── Stage 5: Explainer ─────────────────────────────────────────────────
        explain_result = explainer.explain(
            question=question,
            df=df,
            metric_used=metric_used,
        )

        # ── Store in history ───────────────────────────────────────────────────
        history_entry = {
            "question": question,
            "sql": current_sql,
            "narration": explain_result.narration,
            "chart_type": explain_result.chart_type,
            "row_count": explain_result.row_count,
            "was_corrected": was_corrected,
            "timestamp": datetime.utcnow().isoformat(),
            "success": df is not None,
        }
        _state["query_history"].insert(0, history_entry)
        if len(_state["query_history"]) > MAX_HISTORY:
            _state["query_history"].pop()

        # ── Store successful Q-SQL pair in ChromaDB (fix loop seed) ───────────
        if df is not None and embedder:
            try:
                embedder.store_qa_pair(
                    question, current_sql, was_corrected=was_corrected
                )
            except Exception:
                pass

        # ── Build response ─────────────────────────────────────────────────────
        response = {
            "ok": True,
            "success": df is not None,
            "answer": {
                "narration": explain_result.narration,
                "chart_type": explain_result.chart_type,
                "chart_data": explain_result.chart_data,
                "columns": explain_result.columns,
                "x_key": explain_result.x_key,
                "y_key": explain_result.y_key,
                "row_count": explain_result.row_count,
            },
            "metric_used": metric_used,
            "sql": current_sql,
            "was_corrected": was_corrected,
            "corrections": correction_history,
            "privacy": {"masked_columns": masked_cols},
            "timestamp": datetime.utcnow().isoformat(),
        }

        if df is None:
            response["error"] = (
                "Could not generate a working query after "
                f"{MAX_RETRIES} attempts. Try rephrasing the question."
            )

        return jsonify(response)

    except Exception as e:
        return jsonify(
            {"ok": False, "error": str(e), "detail": traceback.format_exc()}
        ), 500


# ── /api/fix ───────────────────────────────────────────────────────────────────


@app.route("/api/fix", methods=["POST"])
def fix_query():
    """
    Store a user-corrected Q-SQL pair in ChromaDB.

    This is the fix loop (Stage 5 → ChromaDB → Stage 1).
    Every correction makes future similar questions more accurate.

    Body:
      {
        "question":      "original question",
        "corrected_sql": "SELECT ... (user corrected this)",
        "original_sql":  "SELECT ... (what the system generated)"
      }

    Before storing, we validate the corrected SQL executes without error.
    We never store broken SQL — that would corrupt the few-shot examples.
    """
    guard = _require_init()
    if guard:
        return guard

    try:
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()
        corrected_sql = body.get("corrected_sql", "").strip()

        if not question or not corrected_sql:
            return jsonify(
                {"ok": False, "error": "Both question and corrected_sql are required."}
            ), 400

        executor = _state["executor"]
        embedder = _state["embedder"]

        # Validate the corrected SQL before storing
        df, error = executor.execute_direct(corrected_sql)
        if df is None:
            return jsonify(
                {
                    "ok": False,
                    "error": f"Corrected SQL failed validation: {error}",
                    "sql": corrected_sql,
                }
            ), 400

        # Store in ChromaDB
        embedder.store_qa_pair(question, corrected_sql, was_corrected=True)

        return jsonify(
            {
                "ok": True,
                "message": "Correction saved. Future similar questions will use this as a reference.",
                "rows": len(df),
            }
        )

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── /api/history ───────────────────────────────────────────────────────────────


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the query history (most recent first)."""
    n = min(int(request.args.get("n", 10)), MAX_HISTORY)
    return jsonify(
        {
            "ok": True,
            "history": _state["query_history"][:n],
            "total": len(_state["query_history"]),
        }
    )


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear query history."""
    _state["query_history"].clear()
    return jsonify({"ok": True, "message": "History cleared."})


# ── /api/reset ─────────────────────────────────────────────────────────────────


@app.route("/api/reset", methods=["POST"])
def reset_pipeline():
    """
    Reset the pipeline state.
    Call this to re-initialise with a fresh DuckDB and clear metric definitions.
    Does NOT delete metrics.yaml (you need to do that manually if desired).
    """
    _state.update(
        {
            "initialised": False,
            "loader": None,
            "embedder": None,
            "graph": None,
            "linker": None,
            "metric_dict": None,
            "sql_gen": None,
            "executor": None,
            "explainer": None,
            "conn": None,
            "query_history": [],
        }
    )
    return jsonify(
        {"ok": True, "message": "Pipeline reset. Call /api/init to reinitialise."}
    )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _detect_metric_used(question: str, metric_dict: MetricDictionary) -> dict | None:
    """
    Detect which confirmed metric is most relevant to the question.
    Used for the transparency layer: "Using: total_balance — Sum of all balances"

    Simple approach: check if any metric name or its keywords appear in the question.
    This does not need to be perfect — it is a transparency hint, not a routing decision.
    """
    if not metric_dict or metric_dict.is_empty():
        return None

    q_lower = question.lower()
    for name, metric in metric_dict.metrics.items():
        # Check metric name (replacing underscores with spaces)
        if name.replace("_", " ") in q_lower:
            return metric
        # Check keywords from the description
        desc_words = [
            w for w in metric.get("description", "").lower().split() if len(w) > 4
        ]
        if any(w in q_lower for w in desc_words[:5]):
            return metric

    return None


def _seed_qa_pairs(embedder: SchemaEmbedder):
    """
    Seed the ChromaDB Q-SQL collection with 10 banking question-SQL pairs.
    These serve as few-shot examples on the very first query before the user
    has generated any corrections. defined in sql_gen.py — single source of truth.
    These are verified against the actual JSONL schema and match
    the columns DuckDB tables are built from in data_seeder.py.
    """
    for pair in SEED_QA_PAIRS:
        try:
            embedder.store_qa_pair(pair["question"], pair["sql"])
        except Exception:
            pass


# ── /api/query/stream ──────────────────────────────────────────────────────────


@app.route("/api/query/stream", methods=["POST"])
def query_stream():
    """
    Streaming variant of /api/query.

    Emits Server-Sent Events as each pipeline stage completes, giving the
    frontend real-time visibility into what the backend is doing.

    Event sequence:
      stage_active   → stage just started
      stage_complete → stage finished, with real details
      done           → full response payload (same shape as /api/query)
      error          → unrecoverable error

    Each event is a JSON line in SSE format:
      event: stage_complete
      data: {"stage": "linking", "tables": ["Customer", "Account"]}
    """
    guard = _require_init()
    if guard:
        return guard

    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if not question:
        return jsonify({"ok": False, "error": "question is required."}), 400

    def emit(event_type: str, payload: dict) -> str:
        """Format a single SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        """Run the pipeline stage-by-stage, yielding SSE events between stages."""
        try:
            linker = _state["linker"]
            metric_dict = _state["metric_dict"]
            sql_gen = _state["sql_gen"]
            executor = _state["executor"]
            guard_obj = _state["guard"]
            explainer = _state["explainer"]
            embedder = _state["embedder"]

            # ── Stage 1: linking ──────────────────────────────────────────────
            yield emit("stage_active", {"stage": "linking"})

            retrieval = linker.link(question)
            metric_defs = metric_dict.format_for_prompt()
            if metric_defs:
                retrieval.relevant_docs.insert(0, metric_defs)

            # Extract resolved table names from the linker's DDL output
            import re as _re

            tables_resolved = []
            for ddl in retrieval.relevant_ddls:
                m = _re.search(
                    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"']?(\w+)[\"']?",
                    ddl,
                    _re.IGNORECASE,
                )
                if m:
                    tables_resolved.append(m.group(1))

            yield emit(
                "stage_complete",
                {
                    "stage": "linking",
                    "tables": tables_resolved,
                    "table_count": len(tables_resolved),
                },
            )

            # ── Stage 2: generating ───────────────────────────────────────────
            yield emit("stage_active", {"stage": "generating"})

            sql_result = sql_gen.generate(retrieval)
            current_sql = sql_result.sql
            was_corrected = False
            correction_history = []

            yield emit(
                "stage_complete",
                {
                    "stage": "generating",
                    "attempts": sql_result.attempts,
                    "success": sql_result.success,
                },
            )

            # ── Stage 3: executing (with self-correction loop) ────────────────
            yield emit("stage_active", {"stage": "executing"})

            df = None
            error = None

            for attempt in range(1, MAX_RETRIES + 1):
                df, error = executor.execute_direct(current_sql)

                if df is not None:
                    break

                error_type = executor.classify(error)
                correction_history.append(
                    {
                        "attempt": attempt,
                        "sql": current_sql,
                        "error_type": error_type.value.split("—")[0].strip(),
                        "error_message": str(error)[:300],
                    }
                )

                if attempt < MAX_RETRIES:
                    full_error = (
                        f"Error type: {error_type.value.split('—')[0].strip()}\n"
                        f"Error: {str(error)}"
                    )
                    sql_result = sql_gen.regenerate_with_feedback(
                        context=retrieval,
                        previous_sql=current_sql,
                        error_message=full_error,
                    )
                    current_sql = sql_result.sql if sql_result.success else current_sql
                    was_corrected = True

            yield emit(
                "stage_complete",
                {
                    "stage": "executing",
                    "row_count": len(df) if df is not None else 0,
                    "was_corrected": was_corrected,
                    "correction_count": len(correction_history),
                },
            )

            # ── Stage 4: narrating ────────────────────────────────────────────
            yield emit("stage_active", {"stage": "narrating"})

            # Privacy guard runs before narration so masked values flow through
            masked_cols = []
            if df is not None:
                df, masked_cols = guard_obj.scan_and_mask(df)

            metric_used = _detect_metric_used(question, metric_dict)

            explain_result = explainer.explain(
                question=question,
                df=df,
                metric_used=metric_used,
            )

            yield emit(
                "stage_complete",
                {
                    "stage": "narrating",
                    "metric_used": metric_used["name"] if metric_used else None,
                    "chart_type": explain_result.chart_type,
                },
            )

            # ── Persist to history + Q-SQL store (same as /api/query) ─────────
            history_entry = {
                "question": question,
                "sql": current_sql,
                "narration": explain_result.narration,
                "chart_type": explain_result.chart_type,
                "row_count": explain_result.row_count,
                "was_corrected": was_corrected,
                "timestamp": datetime.utcnow().isoformat(),
                "success": df is not None,
            }
            _state["query_history"].insert(0, history_entry)
            if len(_state["query_history"]) > MAX_HISTORY:
                _state["query_history"].pop()

            if df is not None and embedder:
                try:
                    embedder.store_qa_pair(
                        question, current_sql, was_corrected=was_corrected
                    )
                except Exception:
                    pass

            # ── Final payload — identical shape to /api/query response ────────
            final_response = {
                "ok": True,
                "success": df is not None,
                "answer": {
                    "narration": explain_result.narration,
                    "chart_type": explain_result.chart_type,
                    "chart_data": explain_result.chart_data,
                    "columns": explain_result.columns,
                    "x_key": explain_result.x_key,
                    "y_key": explain_result.y_key,
                    "row_count": explain_result.row_count,
                },
                "metric_used": metric_used,
                "sql": current_sql,
                "was_corrected": was_corrected,
                "corrections": correction_history,
                "privacy": {"masked_columns": masked_cols},
                "timestamp": datetime.utcnow().isoformat(),
            }

            if df is None:
                final_response["error"] = (
                    "Could not generate a working query after "
                    f"{MAX_RETRIES} attempts. Try rephrasing the question."
                )

            yield emit("done", final_response)

        except Exception as e:
            yield emit(
                "error",
                {
                    "ok": False,
                    "error": str(e),
                    "detail": traceback.format_exc(),
                },
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables proxy buffering for nginx
        },
    )


# ── Health check ───────────────────────────────────────────────────────────────


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "Talk to Data", "version": "1.0.0"})


# ── Error handlers ─────────────────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"ok": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"ok": False, "error": "Internal server error."}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Talk to Data")
    print("Flask API starting on http://localhost:5000")
    print("React frontend: http://localhost:5173")
    print("=" * 60)
    print()
    print("Endpoints:")
    print("  POST /api/init             → initialise pipeline")
    print("  GET  /api/status           → check readiness")
    print("  POST /api/metrics/propose  → LLM proposes metrics")
    print("  POST /api/metrics/confirm  → save confirmed metrics")
    print("  GET  /api/metrics          → get confirmed metrics")
    print("  POST /api/query            → ask a question")
    print("  POST /api/fix              → store corrected SQL")
    print("  GET  /api/history          → query history")
    print()
    app.run(debug=True, port=5000, threaded=False)

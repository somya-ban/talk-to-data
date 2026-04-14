"""
Flask API entry point for the Graph-RAG Text-to-SQL backend.

Endpoints:
  - GET  /          : basic info
  - GET  /health    : readiness check
  - GET  /schema    : optional debug info
  - POST /query     : main Text-to-SQL pipeline
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.generation.sql_gen import SQLGenerator, SEED_QA_PAIRS
from src.ingestion.loader import SchemaLoader
from src.retrieval.embedder import SchemaEmbedder
from src.retrieval.graph_builder import SchemaGraphBuilder
from src.retrieval.schema_linker import SchemaLinker

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


def _project_paths() -> Tuple[Path, Path]:
    backend_dir = Path(__file__).resolve().parents[1]
    data_dir = backend_dir / "data"
    return backend_dir, data_dir


def _init_pipeline() -> Dict[str, Any]:
    backend_dir, data_dir = _project_paths()

    loader = SchemaLoader(
        tables_path=str(data_dir / "banking_tables_typed.jsonl"),
        relations_path=str(data_dir / "banking_relationships_v2.jsonl"),
    ).load()

    embedder = SchemaEmbedder(persist_dir=str(backend_dir / "chroma_db"))
    embedder.load_from_schema(loader)
    embedder.seed_qa_pairs(SEED_QA_PAIRS)

    graph = SchemaGraphBuilder()
    graph.connect()
    graph.build_from_loader(loader)

    linker = SchemaLinker(embedder=embedder, graph=graph)
    generator = SQLGenerator()

    return {
        "loader": loader,
        "embedder": embedder,
        "graph": graph,
        "linker": linker,
        "generator": generator,
    }


def _execute_sql(sql: str) -> pd.DataFrame:
    db_path = os.getenv("DUCKDB_PATH", ":memory:")
    con = duckdb.connect(database=db_path)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def _explain(df: pd.DataFrame, question: str) -> str:
    if df is None:
        return "No result was produced."
    if df.empty:
        return "No rows matched the question."
    cols = ", ".join(df.columns.tolist())
    return f"Returned {len(df)} rows for '{question}' with columns: {cols}."

def _should_execute_sql() -> bool:
    sql_only = os.getenv("SQL_ONLY", "true").strip().lower()
    if sql_only in {"1", "true", "yes", "y", "on"}:
        return False
    value = os.getenv("EXECUTE_SQL", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def create_app() -> Flask:
    _configure_logging()
    logger = logging.getLogger("backend.app")

    app = Flask(__name__)
    CORS(
        app,
        resources={r"/api/*": {"origins": "http://localhost:3000"}},
        supports_credentials=True
    )

    try:
        pipeline = _init_pipeline()
        app.config["pipeline"] = pipeline
        _log_event(logger, "startup_ready", tables=pipeline["loader"].table_count())
    except Exception as exc:
        app.config["pipeline_error"] = str(exc)
        _log_event(logger, "startup_failed", error=str(exc))

    @app.route("/", methods=["GET"])
    def root() -> Tuple[Any, int]:
        return jsonify({"status": "ok", "service": "talk-to-data-backend"}), 200

    @app.route("/health", methods=["GET"])
    def health() -> Tuple[Any, int]:
        err = app.config.get("pipeline_error")
        if err:
            return jsonify({"status": "error", "error": err}), 500
        return jsonify({"status": "ok"}), 200

    @app.route("/schema", methods=["GET"])
    def schema() -> Tuple[Any, int]:
        err = app.config.get("pipeline_error")
        if err:
            return jsonify({"status": "error", "error": err}), 500
        loader: SchemaLoader = app.config["pipeline"]["loader"]
        return (
            jsonify(
                {
                    "tables": loader.all_table_names(),
                    "table_count": loader.table_count(),
                    "relation_count": loader.relation_count(),
                }
            ),
            200,
        )

    @app.route("/query", methods=["POST", "OPTIONS", "GET"])
    @app.route("/api/query", methods=["POST", "OPTIONS", "GET"])
    def query():
        if request.method == "GET":
            question = (request.args.get("query") or "").strip()
            if not question:
                return (
                    jsonify(
                        {"error": "Use POST with JSON body or GET ?query=..."}
                    ),
                    400,
                )
            payload = {"query": question}
        else:
            payload = request.get_json(silent=True) or {}
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        err = app.config.get("pipeline_error")
        if err:
            return jsonify({"error": err}), 500

        question = str(payload.get("query", "")).strip()

        if not question:
            return jsonify({"error": "Missing 'query' in request body."}), 400

        pipeline = app.config["pipeline"]
        linker: SchemaLinker = pipeline["linker"]
        generator: SQLGenerator = pipeline["generator"]

        start = time.time()
        try:
            _log_event(logger, "query_received", query=question)

            schema_context = linker.link(question)
            sql_result = generator.generate(schema_context)

            if not sql_result.success:
                _log_event(
                    logger,
                    "sql_generation_failed",
                    query=question,
                    error=sql_result.error,
                )
                return jsonify({"error": sql_result.error}), 500

            sql = sql_result.sql
            data = []
            explanation = sql_result.explanation or "SQL generated successfully."

            if _should_execute_sql():
                df = _execute_sql(sql)
                data = df.to_dict(orient="records")
                explanation = sql_result.explanation or _explain(df, question)
            duration_ms = int((time.time() - start) * 1000)
            _log_event(
                logger,
                "query_complete",
                query=question,
                duration_ms=duration_ms,
                rows=len(data),
            )

            return (
                jsonify(
                    {
                        "query": question,
                        "sql": sql,
                        "explanation": explanation,
                        "data": data,
                    }
                ),
                200,
            )

        except Exception as exc:
            _log_event(logger, "query_failed", query=question, error=str(exc))
            return jsonify({"error": str(exc)}), 500

    @app.teardown_appcontext
    def _shutdown(exception: Exception | None) -> None:
        graph: SchemaGraphBuilder | None = None
        pipeline = app.config.get("pipeline")
        if pipeline:
            graph = pipeline.get("graph")
        if graph:
            graph.close()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

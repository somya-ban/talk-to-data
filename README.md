# Talk to Data — Seamless Self-Service Intelligence

**Natural language → SQL → Insights.** A Graph-RAG powered text-to-SQL system for banking microservice data, built for NatWest Code for Purpose 2026.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20llama--3.3--70b-orange.svg)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/vector--store-ChromaDB-blueviolet)](https://trychroma.com)
[![Neo4j](https://img.shields.io/badge/graph--db-Neo4j%20AuraDB-brightgreen)](https://neo4j.com/cloud/aura)
[![DuckDB](https://img.shields.io/badge/sql--engine-DuckDB-yellow)](https://duckdb.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## NatWest Code for Purpose 2026 — Theme 1

> **Talk to Data** — Seamless Self-Service Intelligence
>
> Business stakeholders and analysts often rely on data teams to generate reports, causing bottlenecks. This solution lets any user ask questions in plain English and get instant, accurate SQL-backed answers — no technical knowledge required.

---

## Pipeline

![Talk to Data Pipeline](docs/pipeline.svg)

---

## What It Does

Ask a question in plain English and get back:

**1. Relevant Schema** — Graph-RAG retrieves only the tables and columns needed, not the entire schema

**2. Join Paths** — Neo4j graph traversal finds exactly how tables connect across microservices

**3. Generated SQL** — Groq LLM writes accurate, executable SQL using the retrieved context

**4. Query Results** — DuckDB executes the SQL and returns a pandas DataFrame

**5. Plain English Explanation** — Every query comes with a one-sentence description of what it does

---

## Why Graph-RAG?

Standard text-to-SQL gives the LLM the full schema and hopes it writes the right JOINs. That fails on cross-service joins.

Our banking schema spans **5 microservices**. When a user asks *"show trades for high-risk customers"*, the path is:

```
Customer → Account → Order → Trade
```

ChromaDB alone retrieves `Customer` and `Trade` — it misses `Account` and `Order` because they are not mentioned in the question. Without them, the LLM either hallucinates JOIN columns or produces a broken query.

Graph-RAG solves this:

| Component | Role |
|---|---|
| **ChromaDB** | Finds semantically relevant tables via vector similarity |
| **Neo4j** | Traverses the schema graph to find missing intermediate tables |
| **SchemaLinker** | Merges both results into one context with exact JOIN conditions |
| **Groq LLM** | Generates SQL using the complete, JOIN-path-aware context |

---

## Architecture

### Five-Stage Pipeline

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   SCHEMA LINKER                     │
│                                                     │
│  ChromaDB (semantic)    Neo4j AuraDB (structural)   │
│  ├── DDL collection     ├── 25 Table nodes          │
│  ├── Doc collection     ├── 25 Relationship edges   │
│  └── Q-SQL collection   └── FK-derived JOIN paths   │
│                                                     │
│  Output: SchemaContext (DDLs + Docs + JoinPaths)    │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  SQL GENERATOR                      │
│                                                     │
│  Model: llama-3.3-70b-versatile (Groq, free tier)  │
│  Prompt layers:                                     │
│    1. System instructions + strict rules            │
│    2. DDL statements (table structure)              │
│    3. JOIN conditions (from Neo4j)                  │
│    4. Business definitions (metric dictionary)      │
│    5. Few-shot Q-SQL examples (from ChromaDB)       │
│    6. User question                                 │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                    EXECUTOR                         │
│                                                     │
│  DuckDB in-memory SQL execution                     │
│  Self-correction loop (up to 3 retries)             │
│  Error → regenerate SQL with feedback               │
│  Correct SQL stored back to ChromaDB                │
└─────────────────────────────────────────────────────┘
     │
     ▼
  DataFrame + Chart + Explanation
```

### Schema Data

**25 tables** across 5 banking microservices, fully validated:

| Service | Tables |
|---|---|
| `CustSrv` | Customer, CustomerAddress, CustomerType, Branch, Company |
| `CoreSrv` | Account, AccountBalance, CashMovement, TransferRequest |
| `WealthSrv` | Advisor, ProductWrapper, WrapProvider, ProductWrapperType |
| `TradeSrv` | Order, Trade, Instrument, SecurityType, SecuritySubType, AggregateOrder, Batch |
| `AuthSrv` | User, UserGroup, UserGroupMembership, Application, Session |

**25 relationships** — all FK-backed, join conditions validated, stored as Neo4j graph edges.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Vector Store | ChromaDB 0.5.3 | Three collections: DDL, docs, Q-SQL pairs |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Local, no API cost, 384-dim vectors |
| Graph DB | Neo4j AuraDB Free | Shortest-path traversal for multi-hop JOINs |
| LLM | Groq llama-3.3-70b-versatile | Free tier, 6000 RPM, deterministic at temp=0 |
| SQL Engine | DuckDB 0.10.3 | In-memory, zero-config, pandas-native |
| Backend | Flask | REST API for frontend |
| Frontend | React | Interactive query interface |
| Schema Source | JSONL (custom format) | 25 tables, 25 relationships, FK-validated |

---

## Project Structure

```
talk-to-data/
├── data/
│   ├── banking_tables_typed.jsonl      # 25 validated table definitions
│   └── banking_relationships_v2.jsonl  # 25 FK-backed relationship edges
├── src/
│   ├── ingestion/
│   │   └── loader.py                   # JSONL → DDL + documentation strings
│   ├── retrieval/
│   │   ├── embedder.py                 # ChromaDB vector store (3 collections)
│   │   ├── graph_builder.py            # Neo4j graph population + path queries
│   │   └── schema_linker.py            # Graph-RAG: ChromaDB + Neo4j → SchemaContext
│   ├── generation/
│   │   └── sql_gen.py                  # Groq LLM SQL generation + retry loop
│   ├── semantic/
│   │   └── metric_dict.py              # User-confirmed metric definitions
│   ├── validation/
│   │   └── executor.py                 # DuckDB execution + self-correction loop
│   ├── privacy/
│   │   └── guard.py                    # PII column masking
│   └── explanation/
│       └── explainer.py                # Plain English query narration
├── docs/
│   └── pipeline.svg                    # Architecture diagram
├── .env.example                        # Credential template (no secrets)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) — takes 2 minutes
- A free [Neo4j AuraDB instance](https://console.neo4j.io) — takes 5 minutes

### Installation

```bash
git clone <your-repo-url>
cd talk-to-data

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=your_instance_id
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_instance_id
```

### Verify Each Component

```bash
# 1. Test Neo4j connection
python test_neo4j.py

# 2. Test the full Graph-RAG pipeline
python test_schema_linker.py

# 3. Test SQL generation
python test_sql_gen.py
```

### Run the Application

```bash
python app.py
```

Navigate to `http://localhost:5000` and start asking questions.

---

## Example Queries

| Question | Generated SQL |
|---|---|
| Show all trades for high-risk customers | 4-table JOIN: Customer → Account → Order → Trade |
| Total balance across all accounts | SUM(balance) on AccountBalance |
| List advisors and their assigned branch | Advisor JOIN Branch on branch_id |
| Cash movements for a specific account | CashMovement filtered by account_id |
| Count trades per instrument | GROUP BY on Trade JOIN Instrument |

---

## Key Design Decisions

### 1. Never Give the LLM the Full Schema

Following Vanna's core insight: giving an LLM all 25 table DDLs produces worse SQL than giving it only the 6-8 relevant ones. ChromaDB retrieves the relevant subset; Neo4j adds missing intermediate tables.

### 2. Graph-RAG Over Pure RAG

Pure vector similarity retrieves tables mentioned in the question. It misses tables that are *structurally necessary* but *semantically invisible*. Neo4j's shortest-path traversal closes this gap — every join path is explicitly provided to the LLM.

### 3. Semantic Layer for Metrics

Banking metrics (`total_balance`, `net_cash_flow`) have precise definitions. Users confirm these before the first query. Confirmed definitions are stored in ChromaDB's documentation collection so the LLM uses the exact formula — not its own guess.

### 4. Self-Correction Loop

When DuckDB execution fails, the error message is appended to the prompt and the LLM regenerates. Corrected SQL is stored back into ChromaDB as a Q-SQL pair — the system improves with every correction.

---

## Architecture Decisions — Research Basis

This system synthesises lessons from:

- **Vanna AI** — Three training types (DDL, documentation, Q-SQL pairs), ChromaDB collections pattern
- **LinkedIn SQL Bot** — Schema linking: table pruning before LLM context construction
- **DIN-SQL** — Decomposed SQL generation for complex multi-table queries
- **CHESS** — Evidence-based prompting with explicit join conditions
- **Spider 2.0** — Cross-service schema validation methodology

---

## Team

Built for **NatWest Code for Purpose 2026** — Theme 1: Talk to Data.

Final-year CSE students, Thapar Institute of Engineering and Technology.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

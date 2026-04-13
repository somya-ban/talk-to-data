"""
Verification test for sql_gen.py
Run: python test_sql_gen.py
"""

from src.ingestion.loader import SchemaLoader
from src.retrieval.embedder import SchemaEmbedder
from src.retrieval.graph_builder import SchemaGraphBuilder
from src.retrieval.schema_linker import SchemaLinker
from src.generation.sql_gen import SQLGenerator, SEED_QA_PAIRS

print("=== SQL GENERATOR TEST ===\n")

print("[1/6] Loading schema...")
loader = SchemaLoader(
    tables_path="data/banking_tables_typed.jsonl",
    relations_path="data/banking_relationships_v2.jsonl",
).load()

print("[2/6] Initialising ChromaDB...")
embedder = SchemaEmbedder()
embedder.load_from_schema(loader)

print("[3/6] Seeding Q-SQL pairs...")
seeded = embedder.seed_qa_pairs(SEED_QA_PAIRS)
print(f"      Seeded {seeded} Q-SQL pairs into ChromaDB")

print("[4/6] Connecting Neo4j...")
graph = SchemaGraphBuilder()
graph.connect()
graph.build_from_loader(loader)

print("[5/6] Initialising SchemaLinker and SQLGenerator...")
linker = SchemaLinker(embedder=embedder, graph=graph)
generator = SQLGenerator()

print("\n[6/6] Testing SQL generation...\n")

questions = [
    "Show me all trades for high-risk customers",
    "What is the total balance across all accounts",
    "List all advisors and their assigned branch",
]

for q in questions:
    print(f"Q: {q}")
    ctx = linker.link(q)
    result = generator.generate(ctx)

    if result.success:
        print(f"SQL ({result.attempts} attempt(s)):")
        print(f"  {result.sql}")
        if result.explanation:
            print(f"Explanation: {result.explanation}")
    else:
        print(f"FAILED: {result.error}")
    print()

graph.close()
print("=== DONE ===")

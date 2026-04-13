"""
Verification test for schema_linker.py
Run: python test_schema_linker.py
"""

from src.ingestion.loader import SchemaLoader
from src.retrieval.embedder import SchemaEmbedder
from src.retrieval.graph_builder import SchemaGraphBuilder
from src.retrieval.schema_linker import SchemaLinker

print("=== SCHEMA LINKER TEST ===\n")

# Initialise all three components
print("[1/5] Loading schema...")
loader = SchemaLoader(
    tables_path="data/banking_tables_typed.jsonl",
    relations_path="data/banking_relationships_v2.jsonl",
).load()
print(f"      Tables: {loader.table_count()}, Relations: {loader.relation_count()}")

print("\n[2/5] Initialising ChromaDB embedder...")
embedder = SchemaEmbedder()
result = embedder.load_from_schema(loader)
print(f"      DDL: {result['ddl_loaded']}, Docs: {result['docs_loaded']}")

print("\n[3/5] Connecting to Neo4j graph...")
graph = SchemaGraphBuilder()
graph.connect()
stats = graph.build_from_loader(loader)
print(f"      Nodes: {stats['nodes']}, Edges: {stats['edges']}")

print("\n[4/5] Initialising SchemaLinker...")
linker = SchemaLinker(embedder=embedder, graph=graph)
print("      Ready")

print("\n[5/5] Testing three questions...\n")

questions = [
    "Show me all trades for high-risk customers",
    "What is the total balance across all accounts",
    "List all advisors and their assigned branch",
]

for q in questions:
    print(f"  Q: {q}")
    ctx = linker.link(q)
    print(f"     Tables found:  {ctx.table_names}")
    print(f"     DDLs:          {len(ctx.relevant_ddls)}")
    print(f"     Docs:          {len(ctx.relevant_docs)}")
    print(f"     SQL examples:  {len(ctx.similar_sql)}")
    print(f"     Join hops:     {len(ctx.join_paths)}")
    if ctx.join_context:
        print(f"     Join context:\n{ctx.join_context}")
    print()

graph.close()
print("=== DONE ===")

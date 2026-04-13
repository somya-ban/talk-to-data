import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# import os

# print("URI:", os.getenv("NEO4J_URI"))

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"URI:      {repr(URI)}")
print(f"USERNAME: {repr(USERNAME)}")
print(f"PASSWORD: {'SET' if PASSWORD else 'MISSING'}")
print()

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
try:
    driver.verify_connectivity()
    print("Connection: OK")
    with driver.session() as session:
        count = session.run("MATCH (n) RETURN COUNT(n) AS c").single()["c"]
        print(f"Node count: {count}")
finally:
    driver.close()
    print("Done.")

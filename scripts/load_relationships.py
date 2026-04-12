import json
from neo4j_connection import get_session

def create_relationship(tx, rel):

    source = rel["source"].split(".")[-1]
    target = rel["target"].split(".")[-1]
    relation = rel["relation"]

    query = f"""
    MATCH (a:{source})
    MATCH (b:{target})
    MERGE (a)-[:{relation}]->(b)
    """

    tx.run(query)


def load_relationships():

    with open("../data/banking_relationships.jsonl") as f:
        relationships = [json.loads(line) for line in f]

    with get_session() as session:
        for rel in relationships:
            session.write_transaction(
                create_relationship,
                rel
            )

if __name__ == "__main__":
    load_relationships()
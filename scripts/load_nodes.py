import json
from neo4j_connection import get_session

def create_node(tx, table):

    label = table["table"].split(".")[-1]

    query = f"""
    MERGE (n:{label} {{
        table_name:$table,
        primary_key:$pk,
        description:$desc
    }})
    """

    tx.run(
        query,
        table=table["table"],
        pk=table["primary_key"],
        desc=table["description"]
    )


def load_nodes():

    with open("../data/banking_tables.jsonl") as f:
        tables = [json.loads(line) for line in f]

    with get_session() as session:
        for table in tables:
            session.write_transaction(
                create_node,
                table
            )

if __name__ == "__main__":
    load_nodes()
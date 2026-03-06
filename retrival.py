import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def get_graph_context(user_query, client, driver):
    """
    1. Extracts the core entity from the user query using LLM.
    2. Searches Neo4j for that entity and its relationships.
    """
    # Step A: Identify the subject of the question
    extract_res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "Extract the single most important noun or entity from the user question. Return ONLY the word."},
            {"role": "user", "content": user_query}
        ]
    )
    entity_name = extract_res.choices[0].message.content.strip()

    # Step B: Query Neo4j (using the 'id' property seen in your screenshot)
    # We look for the node and its immediate neighbors to build context
    cypher = """
    MATCH (n)
    WHERE n.id CONTAINS $entity OR n.name CONTAINS $entity
    MATCH (n)-[r]-(neighbor)
    RETURN n.id AS source, type(r) AS rel, neighbor.id AS target
    LIMIT 15
    """

    with driver.session() as session:
        results = session.run(cypher, {"entity": entity_name})
        relations = [f"({row['source']}) -[{row['rel']}]-> ({row['target']})" for row in results]
        return "\n".join(relations)


def ask_llm_with_context(user_query, context, client):
    """Sends the retrieved graph data and the user query to the LLM."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a data assistant. Use the following graph relationships to answer the question accurately."},
            {"role": "user", "content": f"Graph Context:\n{context}\n\nQuestion: {user_query}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def main():
    # Initialize Clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    try:
        query = input("Ask your graph a question: ")

        # 1. Get data from Neo4j
        print("Searching graph...")
        context = get_graph_context(query, client, driver)

        if not context:
            print("No matching relationships found in the graph.")
            return

        # 2. Get answer from LLM
        print("Generating answer...")
        answer = ask_llm_with_context(query, context, client)

        print(f"\nResult:\n{answer}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
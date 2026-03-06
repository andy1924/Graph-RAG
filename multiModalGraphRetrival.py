import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()


class MultimodalGraphRetriever:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def extract_search_term(self, question):
        """Extracts the core entity to find a starting point in the graph."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "Extract the single most important noun from the question for a graph search. Return only the word."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()

    def get_multimodal_context(self, question):
        """
        Traverses the graph to find the entity and its multimodal neighbors
        (Text, Tables, and Images) using the PRECEDES relationship.
        """
        entity = self.extract_search_term(question)

        # Cypher query for Relationship-Aware Traversal
        cypher = """
        MATCH (n)
        WHERE n.id CONTAINS $entity OR n.content CONTAINS $entity
        MATCH (n)-[r:PRECEDES|MENTIONS]-(neighbor)
        RETURN 
            n.id AS source_id, 
            labels(n)[0] AS source_type,
            n.content AS source_content,
            type(r) AS relationship,
            neighbor.id AS target_id,
            labels(neighbor)[0] AS target_type,
            neighbor.content AS target_content
        LIMIT 15
        """

        with self.driver.session() as session:
            results = session.run(cypher, {"entity": entity})
            context_blocks = []

            for record in results:
                block = (
                    f"[{record['source_type']}] {record['source_content']}\n"
                    f"  --({record['relationship']})--> \n"
                    f"[{record['target_type']}] {record['target_content']}"
                )
                context_blocks.append(block)

            return "\n\n".join(context_blocks)

    def close(self):
        self.driver.close()


def main():
    retriever = MultimodalGraphRetriever()
    user_query = "What is positional encoding connected to?"

    print(f"Retrieving context for: {user_query}...")
    context = retriever.get_multimodal_context(user_query)

    if context:
        print("\n--- Retrieved Multimodal Context ---")
        print(context)
        # Next Step: Pass 'context' and 'user_query' to your LLM for the final answer
    else:
        print("No relevant graph nodes found.")

    retriever.close()


if __name__ == "__main__":
    main()
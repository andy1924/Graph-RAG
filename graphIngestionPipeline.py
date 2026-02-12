import os
import json
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

# 1. Setup paths
input_path = "data/raw"
output_path = "data/preprocessed"
os.makedirs(output_path, exist_ok=True) # Ensures the folder exists

loader = DirectoryLoader(
    path=input_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

if len(documents) == 0:
    print(f"File location is empty. Add data first.")
else:
    # Initializing NEO4J (Optional if you're just transforming, but kept for your workflow)
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )

    llm = ChatOpenAI(model='gpt-4o-mini')
    llmTransformer = LLMGraphTransformer(llm=llm)

    # 2. Transform Documents
    graphDocuments = llmTransformer.convert_to_graph_documents(documents)

    # 3. Save to data/preprocessed
    # We convert the objects to dictionaries so they can be saved as JSON
    output_file = os.path.join(output_path, "graph_data.json")
    
    serializable_data = [
        {
            "nodes": [
                {"id": n.id, "type": n.type, "properties": n.properties} 
                for n in doc.nodes
            ],
            "relationships": [
                {
                    "source": r.source.id, 
                    "target": r.target.id, 
                    "type": r.type, 
                    "properties": r.properties
                } 
                for r in doc.relationships
            ],
            "source": doc.source.metadata # Keeps track of which file this came from
        }
        for doc in graphDocuments
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Successfully saved {len(graphDocuments)} graph documents to {output_file}")
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()    #Make sure you have added .env file
loader = DirectoryLoader(
        path="data/raw",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
documents = loader.load()
if len(documents) == 0:
    print(f"File location is empty. Add data first.")
#Initializing NEO4J credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url= NEO4J_URI,
    username= NEO4J_USERNAME,
    password= NEO4J_PASSWORD
)

llm = ChatOpenAI(model='gpt-4o-mini') #Initializing Open AI's API for LLM orchestration

llmTransformer = LLMGraphTransformer(llm = llm)

graphDocuments = llmTransformer.convert_to_graph_documents(documents)

print(graphDocuments)
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma


# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")


# PDF location
PDF_PATH = "data/Introduction_to_ML_Notes.pdf"


# 1. Load PDF
print("\nLoading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"PDF loaded successfully!")
print(f"Number of pages: {len(documents)}")


# 2. Split PDF into chunks
print("\nSplitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks created: {len(chunks)}")


# 3. Create Cohere embeddings
print("\nCreating Cohere embeddings...")

embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-v4.0"
)


# 4. Store embeddings in ChromaDB
print("\nCreating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="ml_notes"
)


print("\n====================================")
print("RAG INGESTION COMPLETED")
print("====================================")
print(f"Pages  : {len(documents)}")
print(f"Chunks : {len(chunks)}")
print("Database: ./chroma_db")
print("====================================")
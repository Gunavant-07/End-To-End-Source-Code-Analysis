from src.helper import repo_ingestion, download_huggingface_embeddings, load_repo_files, text_splitter, get_vector_db_path
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import sys

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 

# url = "https://github.com/Gunavant-07/Medical-Chat-Boat.git"

#repo_ingestion(url)


# Get repository path from app.py
repo_path = sys.argv[1]

# Get repository name
repo_name = sys.argv[2]

print(f"Processing repository: {repo_name}")
print(f"Repository path: {repo_path}")


# Load repository files
documents = load_repo_files(repo_path)

print(f"Documents loaded: {len(documents)}")

# Split documents into chunks
text_chunks = text_splitter(documents)

print(f"Text chunks created: {len(text_chunks)}")

# Load Hugging Face embeddings
embeddings = download_huggingface_embeddings()


# Get repository-specific vector DB path
persist_directory = get_vector_db_path(repo_name)
print(f"Vector database path: {persist_directory}")



# Create Chroma vector database
chromadb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Vector database created successfully.")
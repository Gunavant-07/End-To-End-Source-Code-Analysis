from git import Repo
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
import re

REPO_DATA_DIR = r"E:\GenAi\Project\clone_repo\repo_data"

os.makedirs(REPO_DATA_DIR, exist_ok=True)

REPO_BASE_DIR = os.path.join(REPO_DATA_DIR, "repositories")
VECTOR_DB_BASE_DIR = os.path.join(REPO_DATA_DIR, "vector_db")

os.makedirs(REPO_BASE_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)

#clone any github repo to local directory
def get_repo_name(repo_url):

    repo_name = repo_url.rstrip("/").split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        repo_name
    )

    return repo_name

def repo_ingestion(repo_url):

    repo_name = get_repo_name(repo_url)

    repo_path = os.path.join(
        REPO_BASE_DIR,
        repo_name
    )

    os.makedirs(
        REPO_BASE_DIR,
        exist_ok=True
    )

    if os.path.exists(repo_path):

        print(
            f"Repository already exists: {repo_path}"
        )

        return repo_path, repo_name

    print(
        f"Cloning repository to: {repo_path}"
    )

    Repo.clone_from(
        repo_url,
        repo_path
    )

    return repo_path, repo_name

# Get vector DB path
# ==============================

def get_vector_db_path(repo_name):

    db_path = os.path.join(
        VECTOR_DB_BASE_DIR,
        repo_name
    )

    os.makedirs(
        db_path,
        exist_ok=True
    )

    return db_path

#Loading the repo files and parsing them into documents
def load_repo_files(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                       glob="**/*",
                                       suffixes=[".py"],
                                       parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    
    documents = loader.load()
    return documents

# create a function to split the documents into smaller chunks
def text_splitter(documents):
    document_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=500, chunk_overlap=20)
    text = document_splitter.split_documents(documents)
    return text

#loading the embeddings from huggingface
def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

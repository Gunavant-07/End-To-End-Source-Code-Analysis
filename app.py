from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from src.helper import repo_ingestion, download_huggingface_embeddings, get_vector_db_path
import subprocess

app = Flask(__name__)

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  



llm = ChatOpenAI(
    model="gpt-5.6-luna",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4,
)

embeddings = download_huggingface_embeddings()


Chromadb = None
qa = None
memory = None
current_repo = None

# Create QA chain for repository

def create_qa_chain(repo_name):

    global Chromadb
    global qa
    global memory
    global current_repo

    # Get repository-specific vector DB
    persist_directory = get_vector_db_path(repo_name)

    print("Loading vector DB:")
    print(persist_directory)

    Chromadb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Create memory
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    # Create retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=Chromadb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8
            }
        ),
        memory=memory
    )

    current_repo = repo_name

    print(f"QA chain loaded for: {repo_name}")




@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def git_repo():
    
    if request.method == 'POST':
        user_input = request.form['question']
        repo_path, repo_name = repo_ingestion(user_input)
        subprocess.run(
            [
                "python",
                "store_index.py",
                repo_path,
                repo_name
            ],
            check=True
        )
        create_qa_chain(repo_name)

    return jsonify({
        "response": f"Repository {repo_name} processed successfully.",
        "repo_name": repo_name
    })

@app.route('/get', methods=['POST'])
def chat():

    msg = request.form['msg']
    print("User question:", msg)

    if msg.lower() == "clear":
        memory.clear()
        return "Chat history cleared."
    if qa is None:
        return "Please load a GitHub repository first."
    
    result = qa.invoke({
        "question": msg
    })

    answer = result["answer"]

    print("Answer:", answer)

    return str(answer)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7008, debug=True)
# End-To-End-Source-Code-Analysis

An AI-powered application that analyzes GitHub repositories using **RAG (Retrieval-Augmented Generation)**. Users can provide a GitHub repository URL and ask questions about its source code.

## Features

* Clone GitHub repositories
* Parse and split source code
* Generate embeddings using Hugging Face
* Store embeddings in ChromaDB
* Repository-specific vector databases
* Ask questions about the source code using an LLM
* Flask-based web interface

## Tech Stack

* Python
* Flask
* LangChain
* Hugging Face Embeddings
* ChromaDB
* GitPython
* OpenAI-compatible LLM

## Workflow

```text
GitHub URL
    ↓
Clone Repository
    ↓
Load & Split Source Code
    ↓
Generate Embeddings
    ↓
Store in ChromaDB
    ↓
Ask Questions
    ↓
Retrieve Relevant Code
    ↓
LLM Generates Answer
```

## Storage Structure

```text
repo_data/
├── repositories/
│   ├── repo1/
│   └── repo2/
│
└── vector_db/
    ├── repo1/
    └── repo2/
```

Each repository has its own vector database.

## Installation

```bash
conda create -n scodeanalysis python=3.11
conda activate scodeanalysis
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:7008
```

Enter a GitHub repository URL and start asking questions about the code.

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

Do not commit `.env` or API keys to GitHub.

## License

MIT License

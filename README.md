# Local-document-rag-chatbot
Local RAG chatbot using Mistral GGUF, ChromaDB, LangChain and Streamlit for querying PDF and DOCX documents without external APIs.
# Local Document RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot that enables users to query PDF and Word documents using a locally hosted LLM.

The application ingests documents into a ChromaDB vector store, retrieves relevant context using semantic search, and generates answers using a local Mistral GGUF model through `llama-cpp-python`.

## Features

- Fully local deployment
- No OpenAI or external LLM API required
- PDF document support
- DOCX document support
- ChromaDB vector database
- Streamlit chat interface
- Local Mistral GGUF inference
- Source-aware responses
- CPU-compatible setup

---

## Architecture

```text
PDF / DOCX Documents
          │
          ▼
  Document Loader
          │
          ▼
   Text Chunking
          │
          ▼
 Sentence Embeddings
          │
          ▼
     ChromaDB
          │
          ▼
      Retriever
    (Top-K Chunks)
          │
          ▼
      Mistral LLM
      (GGUF Model)
          │
          ▼
 Generated Response
      + Sources
```

---

## Project Structure

```text
local-document-rag-chatbot/
│
├── docs/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── app.py
├── ingest.py
├── rag_chain.py
├── download_model.py
├── tunnel.py
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

## Tech Stack

### LLM

- Mistral 7B Instruct (GGUF)
- llama-cpp-python

### Retrieval

- LangChain
- ChromaDB

### Embeddings

- Sentence Transformers

### UI

- Streamlit

### Document Processing

- PyPDF
- python-docx

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git

cd <repo-name>
```

### 2. Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / Mac

```bash
python -m venv venv

source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file from `.env.example`.

Example:

```env
MODEL_PATH=./models/mistral-7b-instruct-v0.1.Q5_K_S.gguf

DOCS_DIR=./docs

CHROMA_DIR=./chroma_db

TOP_K=5

MAX_TOKENS=512
```

---

## Download a Model

Place a compatible GGUF model inside the `models` folder.

Supported options:

### Mistral 7B Instruct

Recommended for best quality.

### TinyLlama

Smaller and faster CPU inference.

### Phi-2

Balanced size and performance.

Alternatively run:

```bash
python download_model.py
```

---

## Adding Documents

Place your documents inside:

```text
docs/
```

Example:

```text
docs/
├── Employee Handbook.pdf
├── Company Policy.docx
└── Training Guide.pdf
```

---

## Create the Vector Database

Run ingestion:

```bash
python ingest.py
```

This process:

1. Loads documents
2. Splits text into chunks
3. Generates embeddings
4. Stores vectors in ChromaDB

---

## Start the Chat Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Example Questions

```text
What is the company leave policy?

Summarize the onboarding document.

Which documents mention safety procedures?

What are the approval steps described in the SOP?
```

---

## How It Works

1. User uploads documents to the `docs` folder.
2. Documents are indexed into ChromaDB.
3. User asks a question in the Streamlit UI.
4. Relevant chunks are retrieved using semantic similarity search.
5. Retrieved context is sent to the Mistral model.
6. The model generates a grounded response.
7. Source documents are displayed alongside the answer.

---

## Performance Notes

- Works entirely on local hardware.
- CPU inference may take 20-120 seconds depending on hardware.
- Large models require more RAM.
- Retrieval quality depends on document quality and chunking strategy.

---

## Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Multi-document upload from UI
- Chat history memory
- Response streaming
- Source highlighting
- User authentication
- GPU acceleration
- Reranking models

---

## Security

This project is designed to run locally.

No document data is sent to external APIs unless users modify the implementation.

---

## License

MIT License

Feel free to use, modify, and distribute.

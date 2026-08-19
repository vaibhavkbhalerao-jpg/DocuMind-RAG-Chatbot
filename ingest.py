"""
Document ingestion pipeline.
Loads PDFs and .docx files from the docs/ directory,
splits them into chunks, embeds them, and saves to ChromaDB.

Uses a fully offline TF-IDF based embedding — no internet required.

Usage:
    python ingest.py
    python ingest.py --docs-dir /path/to/other/folder
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(docs_dir: str) -> list:
    """Load all PDFs and .docx files from docs_dir."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"ERROR: docs directory not found: {docs_path.resolve()}")
        sys.exit(1)

    documents = []
    files_found = list(docs_path.glob("**/*.pdf")) + list(docs_path.glob("**/*.docx"))

    if not files_found:
        print(f"No PDF or .docx files found in {docs_path.resolve()}")
        print("Add your documents to the docs/ folder and run again.")
        sys.exit(0)

    for file_path in files_found:
        print(f"  Loading: {file_path.name}")
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))
            else:
                continue
            documents.extend(loader.load())
        except Exception as e:
            print(f"  WARNING: Could not load {file_path.name}: {e}")

    print(f"\nLoaded {len(documents)} pages/sections from {len(files_found)} file(s).")
    return documents


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def embed_and_store(chunks: list, chroma_dir: str):
    """
    Embed chunks using TF-IDF vectors (fully offline, no downloads).
    Saves the index to disk as a pickle file.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    print(f"Embedding {len(chunks)} chunks using TF-IDF (fully offline)...")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"source": chunk.metadata.get("source", "unknown")} for chunk in chunks]

    # Fit TF-IDF on all chunks
    vectorizer = TfidfVectorizer(max_features=5000)
    vectors = vectorizer.fit_transform(texts).toarray()

    # Save everything to disk
    os.makedirs(chroma_dir, exist_ok=True)
    index_path = os.path.join(chroma_dir, "index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "vectors": vectors,
            "texts": texts,
            "metadatas": metadatas,
        }, f)

    print(f"Done. {len(chunks)} chunks stored in '{chroma_dir}/index.pkl'.")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store.")
    parser.add_argument("--docs-dir", default=DOCS_DIR, help="Directory containing PDFs/.docx")
    parser.add_argument("--chroma-dir", default=CHROMA_DIR, help="Output directory for index")
    args = parser.parse_args()

    print(f"=== Document Ingestion ===")
    print(f"Source: {Path(args.docs_dir).resolve()}")
    print(f"Index:  {Path(args.chroma_dir).resolve()}\n")

    print("Step 1: Loading documents...")
    documents = load_documents(args.docs_dir)

    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)

    print("\nStep 3: Embedding and storing...")
    embed_and_store(chunks, args.chroma_dir)

    print("\nIngestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()

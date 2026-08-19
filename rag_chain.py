"""
Core RAG logic: retrieves relevant document chunks and queries the local LLM.
Uses TF-IDF for retrieval and ctransformers for LLM inference — fully offline.

Usage (from other modules):
    from rag_chain import RAGChain
    chain = RAGChain()
    answer, sources = chain.ask_with_sources("What is our vacation policy?")
"""

import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/mistral-7b-instruct-v0.1.Q5_K_S.gguf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# Mistral Instruct v0.1 prompt format
PROMPT_TEMPLATE = """<s>[INST] You are a helpful assistant that answers questions based on the provided context from internal company documents.

Use ONLY the information in the context below to answer the question. If the context does not contain enough information to answer, say "I don't have enough information in the documents to answer this."

Context:
{context}

Question: {question} [/INST]"""


class RAGChain:
    """Loads the TF-IDF index and LLM once, then answers questions."""

    def __init__(self):
        self._check_prerequisites()

        print("Loading document index...")
        index_path = os.path.join(CHROMA_DIR, "index.pkl")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.vectors = data["vectors"]
        self.texts = data["texts"]
        self.metadatas = data["metadatas"]
        print(f"Loaded {len(self.texts)} chunks from index.")

        print(f"Loading LLM from {MODEL_PATH} ...")
        print("(First load may take 30-60 seconds on CPU)")
        from ctransformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type="mistral",
            max_new_tokens=MAX_TOKENS,
            temperature=0.1,
            context_length=4096,
            local_files_only=True,
        )

        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        print("RAG chain ready.")

    def _check_prerequisites(self):
        """Fail fast with helpful error messages if setup is incomplete."""
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Copy your .gguf file to the models/ directory and update .env"
            )
        index_path = os.path.join(CHROMA_DIR, "index.pkl")
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run: python ingest.py\n"
                "Make sure you have documents in the docs/ folder first."
            )

    def _retrieve(self, question: str) -> tuple[list, list]:
        """Find top-K most relevant chunks using TF-IDF cosine similarity."""
        import numpy as np

        query_vec = self.vectorizer.transform([question]).toarray()

        # Cosine similarity between query and all chunks
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = self.vectors / norms
        query_norm = query_vec / (np.linalg.norm(query_vec) or 1)
        scores = normed @ query_norm.T
        scores = scores.flatten()

        top_indices = scores.argsort()[::-1][:TOP_K]
        docs = [self.texts[i] for i in top_indices]
        sources = [self.metadatas[i] for i in top_indices]
        return docs, sources

    def _format_context(self, docs: list, sources: list) -> str:
        """Format retrieved chunks into a single context string."""
        return "\n\n---\n\n".join(
            f"[From: {Path(sources[i].get('source', 'unknown')).name}]\n{docs[i]}"
            for i in range(len(docs))
        )

    def ask(self, question: str) -> str:
        """Ask a question. Returns the LLM's answer as a string."""
        docs, sources = self._retrieve(question)
        context = self._format_context(docs, sources)
        prompt_text = self.prompt.format(context=context, question=question)
        return self.llm(prompt_text)

    def ask_with_sources(self, question: str) -> tuple[str, list[str]]:
        """Ask a question. Returns (answer, list_of_source_filenames)."""
        docs, sources = self._retrieve(question)
        context = self._format_context(docs, sources)
        prompt_text = self.prompt.format(context=context, question=question)
        answer = self.llm(prompt_text)
        source_names = list({Path(s.get("source", "unknown")).name for s in sources})
        return answer, source_names

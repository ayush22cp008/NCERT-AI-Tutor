"""
ingest.py — NCERT AI Tutor
----------------------------
Step 1: Load all NCERT PDF files from the data/ncert_pdfs/ folder.
Step 2: Split them into overlapping chunks for better retrieval.
Step 3: Generate embeddings using a local sentence-transformers model.
Step 4: Store everything in a persistent ChromaDB vector database.

Run this script ONCE (or whenever you add new PDFs) before launching the app.
Usage:
    python ingest.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Reconfigure stdout for UTF-8 encoding to support emojis on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# --- Configuration ---
PDF_DIR = "data/ncert_pdfs"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdfs(pdf_dir: str):
    """Load all PDFs from the given directory."""
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{pdf_dir}'. Please add NCERT textbooks.")
        return []

    print(f"📚 Found {len(pdf_files)} PDF file(s): {pdf_files}")

    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        print(f"   Loading: {pdf_file}...")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

    print(f"✅ Total pages loaded: {len(documents)}")
    return documents


def split_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


def create_vector_store(chunks):
    """Embed chunks and store in ChromaDB."""
    print(f"🧠 Loading embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"💾 Creating/Updating ChromaDB at: '{CHROMA_DB_DIR}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"✅ Vector store created with {vector_store._collection.count()} entries.")
    return vector_store


if __name__ == "__main__":
    print("=" * 50)
    print("  NCERT AI Tutor — Ingestion Pipeline")
    print("=" * 50)

    # Ensure the PDF directory exists
    os.makedirs(PDF_DIR, exist_ok=True)

    documents = load_pdfs(PDF_DIR)
    if not documents:
        exit(1)

    chunks = split_documents(documents)
    create_vector_store(chunks)

    print("\n🎉 Ingestion complete! You can now run: streamlit run app.py")

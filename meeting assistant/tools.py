"""
tools.py
========
All LangChain tools for the Meeting Transcript Analysis System.
Covers: Session Manager, Document Loaders, Text Cleaning, Embedding, ChromaDB.
"""

import os
import re
import uuid
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Embedding client (reused across tools)
# ---------------------------------------------------------------------------
def _get_embedding_function():
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        model="text-embedding-ada-002",
    )

# ---------------------------------------------------------------------------
# ChromaDB client (persistent, file-backed)
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

# ---------------------------------------------------------------------------
# Session state file (lightweight JSON store)
# ---------------------------------------------------------------------------
SESSION_STORE_PATH = os.getenv("SESSION_STORE_PATH", "./sessions.json")

def _load_session_store() -> dict:
    if Path(SESSION_STORE_PATH).exists():
        with open(SESSION_STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_session_store(store: dict):
    with open(SESSION_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)

# ===========================================================================
# TOOL 1 — Session Manager
# ===========================================================================

@tool
def session_manager_tool(action: str, session_id: Optional[str] = None, transcript_hash: Optional[str] = None) -> dict:
    """
    Manages session lifecycle and transcript deduplication.

    Actions:
      - create_session       : Creates a new session, returns session_id + collection_name.
      - load_session         : Loads an existing session by session_id.
      - store_transcript_hash: Stores transcript hash against a session_id.
      - detect_duplicate     : Checks if transcript_hash already exists; returns session_id if found.
      - get_all_sessions     : Returns all sessions.

    Args:
        action          : One of the actions above.
        session_id      : (optional) Session identifier.
        transcript_hash : (optional) MD5/SHA hash of transcript content.

    Returns a dict with keys relevant to the action performed.
    """
    store = _load_session_store()

    if action == "create_session":
        new_id = str(uuid.uuid4())
        collection_name = f"session_{new_id.replace('-', '_')}"
        store[new_id] = {
            "session_id": new_id,
            "collection_name": collection_name,
            "transcript_hash": None,
            "status": "active",
        }
        _save_session_store(store)
        return {"status": "created", "session_id": new_id, "collection_name": collection_name}

    elif action == "load_session":
        if session_id and session_id in store:
            return {"status": "loaded", **store[session_id]}
        return {"status": "not_found", "session_id": session_id}

    elif action == "store_transcript_hash":
        if session_id and session_id in store:
            store[session_id]["transcript_hash"] = transcript_hash
            _save_session_store(store)
            return {"status": "stored", "session_id": session_id, "transcript_hash": transcript_hash}
        return {"status": "error", "message": "Session not found"}

    elif action == "detect_duplicate":
        for sid, data in store.items():
            if data.get("transcript_hash") == transcript_hash and transcript_hash is not None:
                return {
                    "status": "duplicate_found",
                    "session_id": sid,
                    "collection_name": data["collection_name"],
                }
        return {"status": "no_duplicate"}

    elif action == "get_all_sessions":
        return {"status": "ok", "sessions": store}

    return {"status": "error", "message": f"Unknown action: {action}"}


# ===========================================================================
# TOOL 2 — Document Loaders
# ===========================================================================

@tool
def load_pdf_tool(file_bytes: bytes, filename: str = "upload.pdf") -> str:
    """
    Extracts text from a PDF file given its raw bytes.

    Args:
        file_bytes : Raw bytes of the PDF file.
        filename   : Original filename (used for temp file naming).

    Returns the extracted plain text.
    """
    try:
        import pdfplumber
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        os.unlink(tmp_path)
        return "\n".join(text_parts)
    except Exception as e:
        return f"ERROR loading PDF: {e}"


@tool
def load_docx_tool(file_bytes: bytes, filename: str = "upload.docx") -> str:
    """
    Extracts text from a DOCX file given its raw bytes.

    Args:
        file_bytes : Raw bytes of the DOCX file.
        filename   : Original filename.

    Returns the extracted plain text.
    """
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        return f"ERROR loading DOCX: {e}"


@tool
def load_txt_tool(file_bytes: bytes, filename: str = "upload.txt") -> str:
    """
    Decodes text from a TXT file given its raw bytes.

    Args:
        file_bytes : Raw bytes of the TXT file.
        filename   : Original filename.

    Returns the decoded plain text.
    """
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        return f"ERROR loading TXT: {e}"


# ===========================================================================
# TOOL 3 — Text Cleaning Utility
# ===========================================================================

@tool
def clean_text_tool(raw_text: str) -> str:
    """
    Cleans and normalizes raw transcript text.

    Steps:
      - Strips leading/trailing whitespace per line.
      - Collapses multiple blank lines into one.
      - Normalises unicode whitespace.
      - Removes null bytes and control characters.

    Args:
        raw_text : The raw transcript string.

    Returns cleaned text.
    """
    if not raw_text or raw_text.startswith("ERROR"):
        return raw_text
    # Remove null bytes and non-printable control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_text)
    # Normalise unicode spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    # Collapse 3+ consecutive blank lines → 1 blank line
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


# ===========================================================================
# TOOL 4 — Hashing Utility
# ===========================================================================

@tool
def hash_transcript_tool(text: str) -> str:
    """
    Generates a SHA-256 content hash for a transcript string.

    Args:
        text : The cleaned transcript text.

    Returns a hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ===========================================================================
# TOOL 5 — Embedding + ChromaDB Store/Retrieve
# ===========================================================================

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Simple sliding-window chunker."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


@tool
def store_embeddings_tool(text: str, collection_name: str) -> dict:
    """
    Chunks the transcript text, generates embeddings, and stores them in ChromaDB
    under the given collection namespace.

    Args:
        text            : Cleaned transcript text.
        collection_name : ChromaDB collection name (session-scoped).

    Returns a dict with status and number of chunks stored.
    """
    try:
        ef = _get_embedding_function()
        client = _get_chroma_client()

        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        chunks = _chunk_text(text)
        if not chunks:
            return {"status": "error", "message": "No chunks generated from text."}

        embeddings = ef.embed_documents(chunks)
        ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
        )
        return {"status": "stored", "chunks": len(chunks), "collection": collection_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def retrieve_context_tool(query: str, collection_name: str, top_k: int = 5) -> str:
    """
    Retrieves the most relevant transcript chunks from ChromaDB for a given query.

    Args:
        query           : The user's question.
        collection_name : ChromaDB collection name (session-scoped).
        top_k           : Number of top results to retrieve.

    Returns a concatenated string of the top-k relevant chunks.
    """
    try:
        ef = _get_embedding_function()
        client = _get_chroma_client()

        collection = client.get_or_create_collection(name=collection_name)
        query_embedding = ef.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant context found in the transcript."
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        return f"ERROR retrieving context: {e}"

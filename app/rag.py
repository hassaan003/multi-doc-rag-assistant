from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredEPubLoader)

# The main project folder, found from this file's own location
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # reads GOOGLE_API_KEY from .env

# Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
FETCH_K = 20  # how many chunks to fetch from the vector database for context
MAX_HISTORY = 5  # remember the last 5 question-answer pairs
CHROMA_DIR = str(BASE_DIR / "chroma_db")
LLM_MODEL = "gemini-3.1-flash-lite"

# Local embeddings (run on your CPU, no API limits)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# LLM for writing answers
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

# Chat memory: session_id -> list of (question, answer)
histories: dict[str, list[tuple[str, str]]] = {}

REWRITE_PROMPT = """Below is a chat history and a new question from the user.
Rewrite the new question so it can be understood on its own, without the chat history.
Replace words like "it", "that", "this" with what they refer to.
Do NOT answer the question. Return ONLY the rewritten question.

Chat history:
{history}

New question: {question}

Standalone question:"""

PROMPT = """Answer the question using the context below.
Use only the information in the context. Do not add facts from outside.
If the context has related information but not a full answer, say what it does say and what is missing.
Only say "I could not find this in the uploaded documents." if the context has nothing relevant.
Use the chat history only to understand what the user wants (for example "explain simply" or "make it shorter").

Chat history:
{history}

Context:
{context}

Question: {question}"""

def get_text(reply) -> str:
    """Gemini may return a list of blocks. Join all the text parts into one string."""
    content = reply.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)

def format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)


def rewrite_question(question: str, history: list[tuple[str, str]]) -> str:
    """Turn a follow-up like 'explain that more simply' into a full question."""
    if not history:
        return question
    reply = llm.invoke(
        REWRITE_PROMPT.format(history=format_history(history), question=question)
    )
    return get_text(reply).strip()


def ingest_file(path: str, filename: str, session_id: str = "default") -> dict:
    """Read a file, cut it into chunks, and save them in the session's vector database."""
    ext = path.lower()
    
    if ext.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif ext.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    elif ext.endswith(".csv"):
        loader = CSVLoader(path)
    elif ext.endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(path)
    elif ext.endswith((".ppt", ".pptx")):
        loader = UnstructuredPowerPointLoader(path)
    elif ext.endswith(".html"):
        loader = UnstructuredHTMLLoader(path)
    elif ext.endswith(".md"):
        loader = UnstructuredMarkdownLoader(path)
    elif ext.endswith(".epub"):
        loader = UnstructuredEPubLoader(path)
    else:
        raise ValueError("Unsupported file type")

    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text found in this file.")

    for chunk in chunks:
        chunk.metadata["source"] = filename

    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    
    # Store documents in THIS session's specific collection
    db = get_session_db(session_id)
    db.add_documents(chunks, ids=ids)
    return {"pages": len(pages), "chunks": len(chunks)}

def delete_session_data(session_id: str):
    """Wipe chat history and delete this session's collection from ChromaDB."""
    histories.pop(session_id, None)
    try:
        client = db._client
        client.delete_collection(name=f"session_{session_id}")
    except Exception:
        pass

def ask(question: str, session_id: str = "default") -> dict:
    history = histories.get(session_id, [])
    standalone = rewrite_question(question, history)
    
    # Search ONLY within this session's collection
    db = get_session_db(session_id)
    results = db.max_marginal_relevance_search(standalone, k=TOP_K, fetch_k=FETCH_K)
    if not results:
        return {
            "answer": "No documents uploaded yet.",
            "standalone_question": standalone,
            "sources": [],
        }

    """Read a file, cut it into chunks, and save them in the vector database."""
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.endswith(".txt"):
        loader = TextLoader(path)
    elif path.endswith(".csv"):
        loader = CSVLoader(path)
    elif path.endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(path)
    elif path.endswith((".ppt", ".pptx")):
        loader = UnstructuredPowerPointLoader(path)
    elif path.endswith(".html"):
        loader = UnstructuredHTMLLoader(path)
    elif path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(path)
    elif path.endswith(".epub"):
        loader = UnstructuredEPubLoader(path)
    else:
        raise ValueError("Unsupported file type")

    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text found in this file.")

    for chunk in chunks:
        chunk.metadata["source"] = filename

    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    db.add_documents(chunks, ids=ids)
    return {"pages": len(pages), "chunks": len(chunks)}


def ask(question: str, session_id: str = "default") -> dict:
    """Rewrite the question, find the best chunks in this session, answer, and remember chat."""
    history = histories.get(session_id, [])

    standalone = rewrite_question(question, history)
    
    # Search ONLY within this session's vector collection
    db = get_session_db(session_id)
    results = db.max_marginal_relevance_search(standalone, k=TOP_K, fetch_k=FETCH_K)
    
    if not results:
        return {
            "answer": "No documents uploaded yet in this session.",
            "standalone_question": standalone,
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in results)
    reply = llm.invoke(
        PROMPT.format(
            history=format_history(history), context=context, question=standalone
        )
    )
    
    answer = get_text(reply)

    history.append((question, answer))
    histories[session_id] = history[-MAX_HISTORY:]

    sources = [
        {
            "file": doc.metadata.get("source"),
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content[:500],
        }
        for doc in results
    ]
    return {"answer": answer, "standalone_question": standalone, "sources": sources}


def reset_session(session_id: str) -> None:
    histories.pop(session_id, None)

def get_session_db(session_id: str):
    """Get a Chroma collection unique to this session ID for strict data isolation."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=f"session_{session_id}"
    )
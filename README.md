# Multi-Doc RAG Assistant

A production-grade, multi-format Retrieval-Augmented Generation (RAG) web application built with **FastAPI**, **LangChain**, **ChromaDB**, and **Google Gemini**. 

Designed for strict data isolation and multi-turn conversational accuracy, this assistant allows users to upload multiple document formats, ask context-aware follow-up questions, inspect source citations, and completely purge session state and vector indices on demand.

---

## Key Features

* **Multi-Format Document Ingestion:** Dynamically routes and parses `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`, `.md`, `.html`, and `.epub` files using modular LangChain document loaders.
* **Session-Isolated Vector Storage:** Each user session writes to an isolated ChromaDB collection (`session_<id>`), preventing cross-session data bleeding.
* **Cascading Session Teardown:** Deleting a session executes a full cleanup lifecycle—wiping ephemeral chat history, dropping the session's ChromaDB collection, and unlinking uploaded source files from disk.
* **Conversational Memory & Query Rewriting:** Maintains per-session chat history and uses an LLM pre-processing pass to rewrite context-dependent follow-up questions (e.g., *"explain that more simply"*) into standalone search queries before retrieval.
* **Max Marginal Relevance (MMR) Retrieval:** Balances semantic similarity with result diversity (`fetch_k=20`, `k=6`) to avoid redundant chunks and maximize context coverage.
* **Zero-Cost Local Embeddings:** Runs `all-MiniLM-L6-v2` via HuggingFace locally on CPU, eliminating embedding API rate limits and latency overhead.
* **Source Grounding & Citations:** Strict prompt grounding ensures answers rely solely on uploaded context, returning expandable source snippets with filenames and page numbers.

---

## Tech Stack

* **Backend:** FastAPI, Uvicorn, Python
* **AI & Orchestration:** LangChain, Google Gemini (`gemini-3.1-flash-lite` / `gemini-1.5-flash`)
* **Embeddings & Vector Store:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`), ChromaDB
* **Document Parsers:** `pypdf`, `unstructured`, `python-docx`, `python-pptx`
* **Frontend:** Vanilla HTML5, CSS3 (Responsive Dark/Light Mode), JavaScript (Fetch API, Drag-and-Drop)

---

## Project Structure

```text
multi-doc-rag-assistant/
├── app/
│   ├── main.py          # FastAPI routes, file validation, and session lifecycle hooks
│   └── rag.py           # Document loaders, MMR retrieval, query rewriting, and ChromaDB management
├── static/
│   └── index.html       # Single-page chat UI with drag-and-drop uploads and source viewer
├── uploads/             # Transient local storage for active session files (gitignored)
├── chroma_db/           # Local persistent ChromaDB storage (gitignored)
├── .env                 # Environment configuration for GOOGLE_API_KEY (gitignored)
└── requirements.txt     # Python dependencies

Getting Started1. Clone the RepositoryBashgit clone [https://github.com/hassaan003/multi-doc-rag-assistant.git](https://github.com/hassaan003/multi-doc-rag-assistant.git)
cd multi-doc-rag-assistant
2. Set Up a Virtual EnvironmentBashpython -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
3. Install DependenciesBashpip install fastapi uvicorn python-dotenv pydantic
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-google-genai langchain-chroma
pip install pypdf unstructured python-docx python-pptx markdown epub sentence-transformers
4. Configure Environment VariablesCreate a .env file in the root directory and add your Google AI Studio API key:Code snippetGOOGLE_API_KEY=your_google_api_key_here
5. Run the ApplicationBashuvicorn app.main:app --reload
Chat Interface: Open http://127.0.0.1:8000/Interactive API Docs (Swagger UI): Open http://127.0.0.1:8000/docsArchitecture FlowIngestion (POST /upload): Validates file extension $\rightarrow$ saves file locally $\rightarrow$ extracts text via format-specific loader $\rightarrow$ splits text (chunk_size=1000, chunk_overlap=200) $\rightarrow$ embeds locally and stores in session_<id> Chroma collection.Retrieval & Generation (POST /ask): Rewrites follow-up query using recent session history $\rightarrow$ performs MMR vector search on session_<id> $\rightarrow$ injects retrieved chunks into grounded system prompt $\rightarrow$ generates response via Gemini and returns answer + source metadata.Cleanup (DELETE /sessions/{session_id}): Clears in-memory chat history $\rightarrow$ drops session_<id> collection from ChromaDB $\rightarrow$ deletes transient files from uploads/

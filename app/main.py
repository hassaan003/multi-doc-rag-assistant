import shutil
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from pydantic import BaseModel

from app import rag

app = FastAPI(title="Document RAG Chatbot")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...), session_id: str = Query("default")):
    allowed_extensions = (".pdf", ".txt", ".csv", ".doc", ".docx", ".ppt", ".pptx", ".html", ".md", ".epub")
    
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format."
        )

    filename = Path(file.filename).name
    path = UPLOAD_DIR / filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Pass the session_id so embeddings go into this session's isolated collection
        info = rag.ingest_file(str(path), filename, session_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return {"filename": filename, **info}


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    return rag.ask(request.question, request.session_id)


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    # Wipe Chroma collection, chat history, and local files
    rag.delete_session_data(session_id)
    
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            try:
                file_path.unlink()
            except Exception:
                pass

    return {"status": "cleared"}
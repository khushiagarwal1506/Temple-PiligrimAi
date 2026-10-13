import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure root backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.api.chat import router as chat_router
from app.api.maps import router as maps_router
from app.api.auth import router as auth_router
from app.api.chat_history import router as chat_history_router

app = FastAPI(
    title="PilgrimAI Local Offline RAG API",
    description="100% Local Offline RAG pipeline using Qdrant (on-disk), BM25, RRF, Cross-Encoder, and Ollama Qwen2.5.",
    version="3.0.0"
)

import logging

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """FastAPI startup lifecycle hook - eager coordinator and RAG initialization."""
    try:
        from app.db.chat_history import init_chat_indexes
        init_chat_indexes()
        from app.api.chat import get_coordinator
        get_coordinator()
        logger.info("[STARTUP SUCCESS] MasterAgentCoordinator, RAG services, and DB indexes initialized successfully.")
    except Exception as e:
        logger.error(f"[STARTUP FAILED] Error initializing agent coordinator at startup: {e}", exc_info=True)
        raise e

# Enable CORS for local testing / frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat_router)
app.include_router(maps_router)
app.include_router(auth_router)
app.include_router(chat_history_router)


# Mount Static Web Assets
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", tags=["UI"])
async def root():
    """Serve web chat UI interface at root URL."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {
        "status": "online",
        "service": "PilgrimAI Local Offline RAG API",
        "endpoints": {
            "chat": "POST /chat",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}

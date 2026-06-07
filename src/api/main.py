"""
src/api/main.py
────────────────
Unified FastAPI application — serves both:
  • Frontend  → GET  /          (Jinja2 HTML dashboard)
  • Backend   → POST /api/v1/analyse  (AI pipeline)

No Flask. No separate process. One server, one port.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import settings
from src.api.routes import router

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "src" / "templates"
STATIC_DIR    = BASE_DIR / "src" / "static"


def create_app() -> FastAPI:
    application = FastAPI(
        title="AI Interview Performance & Tone Analyzer",
        description=(
            "Unified FastAPI server — serves the interactive dashboard UI and "
            "drives the sequential AI pipeline: Speaker Diarization → "
            "Transcription → Acoustic Analysis → Vector Search → Gemini LLM."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Static files ──────────────────────────────────────────────────────────
    application.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    # ── Templates ─────────────────────────────────────────────────────────────
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # ── CORS ──────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routes ────────────────────────────────────────────────────────────
    application.include_router(router)

    # ── Frontend route ────────────────────────────────────────────────────────
    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "app_title": "AI Interview Analyzer"},
        )

    @application.on_event("startup")
    async def _startup():
        logger.info(
            "Unified server started → http://localhost:%d  |  Docs → /docs",
            settings.app_port,
        )

    return application


app: FastAPI = create_app()

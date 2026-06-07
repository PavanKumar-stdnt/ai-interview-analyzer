# AI Interview Performance & Tone Analyzer (Unified FastAPI)

A production-grade AI system that analyses mock interview recordings.
**Single unified FastAPI server** — no Flask, no separate processes.

---

## Architecture

```
Browser
   │
   ▼
FastAPI  :8000   (serves BOTH frontend dashboard + AI API)
   │
   ├── GET  /                    → Jinja2 HTML dashboard
   ├── POST /api/v1/analyse      → Sequential AI pipeline
   └── GET  /api/v1/health       → Health check
         │
         ├─ 1. Pyannote Diarization   (GPU → VRAM flush)
         ├─ 2. Faster-Whisper         (GPU → VRAM flush)
         ├─ 3. Librosa Acoustics      (CPU only)
         ├─ 4. Qdrant Vector Search   (local disk)
         └─ 5. Gemini 2.5 Flash LLM   (API → structured JSON)
```

---

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Configure
cp .env.template .env      # fill in GEMINI_API_KEY and HF_TOKEN

# 3. Run
python run.py
```

Open **http://localhost:8000**

---

## URLs

| URL | Purpose |
|-----|---------|
| http://localhost:8000 | Dashboard |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/api/v1/health | Health check |

---

## Mock Mode (no model downloads)

In `src/api/routes.py` set:
```python
USE_MOCK_PIPELINE: bool = True
```

---

## Project Structure

```
ai-interview-analyzer/
├── config/settings.py           # Pydantic Settings (single APP_PORT)
├── src/
│   ├── api/
│   │   ├── main.py              # Unified FastAPI: frontend + backend
│   │   ├── routes.py            # /api/v1/* endpoints
│   │   └── schemas.py           # Pydantic schemas
│   ├── templates/
│   │   └── index.html           # Jinja2 dashboard template
│   ├── static/                  # CSS / JS static assets
│   ├── database/qdrant_setup.py
│   ├── pipelines/
│   │   ├── diarization.py
│   │   ├── transcription.py
│   │   └── acoustics.py
│   └── services/gemini_service.py
├── .env.template
├── requirements.txt
└── run.py                       # Seeds DB + starts single server
```

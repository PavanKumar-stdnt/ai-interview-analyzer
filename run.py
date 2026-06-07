"""
run.py
───────
Orchestration entry point.

1. Seeds the Qdrant 'interview_rubrics' collection (idempotent).
2. Launches a SINGLE unified FastAPI server (uvicorn) that serves
   both the HTML dashboard and the AI API.

Usage:
    python run.py
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_seed() -> None:
    try:
        from src.database.qdrant_setup import seed_rubrics_collection
        logger.info("Seeding Qdrant database …")
        seed_rubrics_collection()
        logger.info("✅  Database ready.")
    except Exception as exc:
        logger.error("Seed failed: %s — continuing anyway.", exc)


def start_server() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", str(settings.app_port),
        "--log-level", "info",
    ]
    logger.info("Starting unified FastAPI server on port %d …", settings.app_port)
    return subprocess.Popen(cmd, cwd=str(ROOT))


def main() -> None:
    run_seed()

    proc = start_server()

    def _shutdown(sig, frame):
        logger.info("Shutting down …")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║  AI Interview Analyzer — Unified FastAPI Server      ║\n"
        "║                                                      ║\n"
        "║  Dashboard  →  http://localhost:%-5s               ║\n"
        "║  API Docs   →  http://localhost:%-5s/docs          ║\n"
        "║  Health     →  http://localhost:%-5s/api/v1/health ║\n"
        "╚══════════════════════════════════════════════════════╝",
        settings.app_port, settings.app_port, settings.app_port,
    )

    # Block until process exits
    proc.wait()


if __name__ == "__main__":
    main()

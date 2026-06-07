"""
src/api/routes.py
──────────────────
FastAPI router — all API endpoints under /api/v1/

Pipeline execution order (strictly sequential to prevent OOM):
  1. Save uploaded file
  2. Diarization  → clear VRAM
  3. Transcription → clear VRAM
  4. Acoustics (CPU)
  5. Qdrant vector search
  6. Gemini LLM evaluation
  7. Return structured JSON

──────────────────────────────────────────────────────────
MOCK MODE  →  set USE_MOCK_PIPELINE = True below
──────────────────────────────────────────────────────────
"""

from __future__ import annotations

import gc
import logging
import shutil
import uuid
from pathlib import Path
import subprocess

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from config.settings import settings
from src.api.schemas import AnalysisResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Analysis"])

# ── Toggle to test without downloading any AI models ──────────────────────────
USE_MOCK_PIPELINE: bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flush_vram(label: str) -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        logger.info("[VRAM] Cleared after %s.", label)
    except Exception:
        pass


def _save_upload(file: UploadFile) -> Path:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("[Upload] Saved → %s (%d bytes)", dest, dest.stat().st_size)
    return dest


# ── Mock pipeline ──────────────────────────────────────────────────────────────

def _run_mock_pipeline(audio_path: Path) -> dict:
    logger.warning("[MOCK] Mock pipeline active — no real models used.")
    return {
        "status": "success",
        "filename": audio_path.name,
        "duration_seconds": 95.4,
        "speaker_count": 2,
        "transcript": (
            "Good morning. My name is Alex Chen and I am excited to be here. "
            "A Python list is a mutable ordered collection, meaning you can add, "
            "remove, or change elements after creation. A tuple is immutable — "
            "once defined its contents cannot change. Tuples are faster and more "
            "memory-efficient. Because tuples are hashable they can be used as "
            "dictionary keys, which lists cannot."
        ),
        "acoustics": {
            "volume_label": "Normal", "rms_mean": 0.071,
            "pitch_mean_hz": 148.3, "pitch_label": "Normal",
            "speech_rate_wpm": 132.0, "pacing_label": "Normal",
            "filler_count": 2, "filler_words": ["um", "like"],
            "duration_seconds": 95.4,
        },
        "matched_rubric_question": "What is the difference between a Python List and a Tuple?",
        "evaluation": {
            "overall_score": 82,
            "technical_accuracy":   {"topic": "Technical Accuracy",   "score": 9, "feedback": "Correctly identified mutability, immutability, and hashability with accurate use-cases."},
            "communication_clarity":{"topic": "Communication Clarity", "score": 8, "feedback": "Clear structure with good real-world examples. Minor filler words detected."},
            "confidence_level":     {"topic": "Confidence Level",      "score": 7, "feedback": "Steady pace and normal pitch. Could project slightly more authority."},
            "structure_and_depth":  {"topic": "Structure & Depth",     "score": 8, "feedback": "Logical flow covering key concepts. A code snippet would strengthen the answer."},
            "strengths": [
                "Accurate technical knowledge of mutability vs immutability",
                "Good use of concrete real-world examples",
                "Appropriate pacing at 132 WPM",
                "Correctly explained hashability for dict keys",
            ],
            "improvement_areas": [
                "Reduce filler words — 2 detected (um, like)",
                "Add a brief code example to illustrate differences",
                "Expand on performance benchmarks with numbers",
            ],
            "recommended_topics": [
                "Python data structures deep-dive",
                "Memory management in CPython",
                "Time complexity: list vs tuple operations",
            ],
            "tone_summary": (
                "The candidate speaks with measured confidence and a clear educational tone. "
                "Delivery is steady with minimal hesitation. The overall impression is of a "
                "well-prepared engineer who understands Python fundamentals."
            ),
            "hiring_recommendation": "Yes",
        },
    }


# ── Real pipeline ──────────────────────────────────────────────────────────────

def _run_real_pipeline(audio_path: Path) -> dict:
    
    # ── Convert audio to 16kHz mono WAV for pyannote/whisper ──
    #wav_path = str(Path(audio_path).with_suffix(".wav"))
    #wav_path = str(audio_path.with_name(f"{audio_path.stem}_converted.wav"))

    #subprocess.run([
        #"ffmpeg",
        #"-y",
        #"-i", str(audio_path),
       # "-ac", "1",
      #  "-ar", "16000",
     #   wav_path
    #], check=True)

    #audio_path = Path(wav_path)

    
    
    from src.pipelines.diarization import DiarizationPipeline
    from src.pipelines.transcription import TranscriptionPipeline
    from src.pipelines.acoustics import analyse_audio
    from src.database.qdrant_setup import search_rubrics
    from src.services.gemini_service import GeminiEvaluationService

    # Step 1 — Diarization
    logger.info("[Pipeline] 1/6 Diarization")
    dia = DiarizationPipeline().run(audio_path)
    _flush_vram("diarization")
    speaker_count = dia.speaker_count or 1

    # Step 2 — Transcription
    logger.info("[Pipeline] 2/6 Transcription")
    txn = TranscriptionPipeline().run(audio_path)
    _flush_vram("transcription")
    if txn.error:
        raise RuntimeError(f"Transcription failed: {txn.error}")
    transcript = txn.transcript

    # Step 3 — Acoustics (CPU)
    logger.info("[Pipeline] 3/6 Acoustics")
    ac = analyse_audio(audio_path, transcript=transcript)
    acoustics_dict = {
        "volume_label": ac.volume_label, "rms_mean": ac.rms_mean,
        "pitch_mean_hz": ac.pitch_mean_hz, "pitch_label": ac.pitch_label,
        "speech_rate_wpm": ac.speech_rate_wpm, "pacing_label": ac.pacing_label,
        "filler_count": ac.filler_count, "filler_words": ac.filler_words or [],
        "duration_seconds": ac.duration_seconds,
    }

    # Step 4 — Vector Search (transcript chunked → sentence-based sliding window)
    logger.info("[Pipeline] 4/6 Vector Search — sentence-chunked transcript")
    rubrics = search_rubrics(transcript, top_k=1)
    rubric = rubrics[0] if rubrics else {}
    matched_question = rubric.get("question", "General Interview Question")

    # Step 5 — Gemini LLM
    logger.info("[Pipeline] 5/6 Gemini Evaluation")
    evaluation = GeminiEvaluationService().evaluate(
        transcript=transcript, rubric=rubric,
        acoustics=acoustics_dict, speaker_count=speaker_count,
    )

    # Step 6 — Assemble
    logger.info("[Pipeline] 6/6 Assembling response")
    return {
        "status": "success",
        "filename": audio_path.name,
        "duration_seconds": ac.duration_seconds,
        "speaker_count": speaker_count,
        "transcript": transcript,
        "acoustics": acoustics_dict,
        "matched_rubric_question": matched_question,
        "evaluation": evaluation.model_dump(),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/analyse",
    response_model=AnalysisResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyse an interview audio recording",
)
async def analyse_interview(
    file: UploadFile = File(..., description="Audio file (wav / mp3 / flac)")
):
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided.")

    # Basic MIME guard
    allowed = {"audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/mp4", "audio/x-wav"}
    if file.content_type and file.content_type not in allowed:
        logger.warning("[Upload] Received content-type: %s", file.content_type)

    audio_path: Path | None = None
    try:
        audio_path = _save_upload(file)
        result = _run_mock_pipeline(audio_path) if USE_MOCK_PIPELINE else _run_real_pipeline(audio_path)
        return JSONResponse(content=result)

    except Exception as exc:
        logger.error("[Route] Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
                logger.info("[Cleanup] Removed temp file: %s", audio_path)
            except Exception:
                pass


@router.get("/health", summary="Health check")
async def health():
    return {"status": "healthy", "mock_mode": USE_MOCK_PIPELINE, "server": "unified-fastapi"}

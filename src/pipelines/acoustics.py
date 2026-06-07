"""
src/pipelines/acoustics.py
───────────────────────────
CPU-bound acoustic feature extraction using Librosa.

Computes:
  - Volume level  → classified as "Low" / "Normal" / "High"
  - Pitch mean    → proxy for vocal confidence (Hz)
  - Speech rate   → words-per-minute approximation
  - Filler counts → "um", "uh", "like", "you know" occurrences

No GPU resources are touched in this module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Volume thresholds (RMS in linear scale) ────────────────────────────────────
_RMS_LOW_THRESHOLD = 0.02
_RMS_HIGH_THRESHOLD = 0.12


@dataclass
class AcousticsResult:
    volume_label: str = "Normal"         # "Low" | "Normal" | "High"
    rms_mean: float = 0.0
    pitch_mean_hz: float = 0.0
    pitch_label: str = "Normal"          # "Uncertain" | "Normal" | "Confident"
    speech_rate_wpm: float = 0.0
    pacing_label: str = "Normal"         # "Slow" | "Normal" | "Fast"
    filler_count: int = 0
    filler_words: list[str] | None = None
    duration_seconds: float = 0.0
    error: str | None = None


# ── Filler word patterns ───────────────────────────────────────────────────────
_FILLER_PATTERN = re.compile(
    r"\b(um+|uh+|like|you know|basically|literally|right\?|so+|actually)\b",
    re.IGNORECASE,
)


def analyse_audio(audio_path: str | Path, transcript: str = "") -> AcousticsResult:
    """
    Perform CPU-bound acoustic analysis on *audio_path*.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file (wav / mp3 / flac).
    transcript : str, optional
        Transcribed text used for filler-word counting and WPM estimation.

    Returns
    -------
    AcousticsResult
        Populated dataclass with labelled acoustic features.
    """
    import librosa  # lazy import keeps startup fast

    try:
        logger.info("[Acoustics] Loading audio (CPU): %s", audio_path)
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # ── Volume (RMS) ──────────────────────────────────────────────────────
        rms_frames = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms_frames))

        if rms_mean < _RMS_LOW_THRESHOLD:
            volume_label = "Low"
        elif rms_mean > _RMS_HIGH_THRESHOLD:
            volume_label = "High"
        else:
            volume_label = "Normal"

        # ── Pitch (Yin fundamental frequency) ─────────────────────────────────
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        # Filter out unvoiced frames (yin returns fmin for unvoiced)
        voiced_f0 = f0[f0 > 85]
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # Classify pitch as a confidence proxy
        # Lower/monotone pitch → "Uncertain"; higher variance/pitch → "Confident"
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        if pitch_std < 15:
            pitch_label = "Uncertain"
        elif pitch_std > 40:
            pitch_label = "Confident"
        else:
            pitch_label = "Normal"

        # ── Speech rate (WPM) ─────────────────────────────────────────────────
        word_count = len(transcript.split()) if transcript.strip() else 0
        wpm = (word_count / duration * 60) if duration > 0 else 0.0

        if wpm < 100:
            pacing_label = "Slow"
        elif wpm > 160:
            pacing_label = "Fast"
        else:
            pacing_label = "Normal"

        # ── Filler words ──────────────────────────────────────────────────────
        filler_matches = _FILLER_PATTERN.findall(transcript)
        filler_count = len(filler_matches)

        logger.info(
            "[Acoustics] volume=%s rms=%.4f pitch=%.1fHz(%s) wpm=%.0f(%s) fillers=%d",
            volume_label,
            rms_mean,
            pitch_mean,
            pitch_label,
            wpm,
            pacing_label,
            filler_count,
        )

        return AcousticsResult(
            volume_label=volume_label,
            rms_mean=round(rms_mean, 4),
            pitch_mean_hz=round(pitch_mean, 2),
            pitch_label=pitch_label,
            speech_rate_wpm=round(wpm, 1),
            pacing_label=pacing_label,
            filler_count=filler_count,
            filler_words=list(set(filler_matches)),
            duration_seconds=round(duration, 2),
        )

    except Exception as exc:
        logger.error("[Acoustics] Failed: %s", exc, exc_info=True)
        return AcousticsResult(error=str(exc))

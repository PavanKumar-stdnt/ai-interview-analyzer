"""
src/api/schemas.py
────────────────────
Pydantic request/response schemas for the FastAPI layer.
Kept separate from service-level schemas to maintain clean layer boundaries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentSchema(BaseModel):
    start: float
    end: float
    text: str | None = None
    speaker: str | None = None


class AcousticsSchema(BaseModel):
    volume_label: str
    rms_mean: float
    pitch_mean_hz: float
    pitch_label: str
    speech_rate_wpm: float
    pacing_label: str
    filler_count: int
    filler_words: list[str]
    duration_seconds: float


class TopicScoreSchema(BaseModel):
    topic: str
    score: int = Field(ge=0, le=10)
    feedback: str


class AnalysisResponse(BaseModel):
    """Top-level response envelope returned by POST /api/v1/analyse."""

    status: str = "success"
    filename: str
    duration_seconds: float
    speaker_count: int
    transcript: str
    acoustics: AcousticsSchema
    matched_rubric_question: str
    evaluation: dict  # serialised InterviewEvaluation

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "filename": "interview.wav",
                "duration_seconds": 120.5,
                "speaker_count": 2,
                "transcript": "Good morning …",
                "acoustics": {},
                "matched_rubric_question": "What is a Vector Database?",
                "evaluation": {},
            }
        }


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str

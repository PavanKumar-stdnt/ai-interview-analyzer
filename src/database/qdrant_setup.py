"""
src/database/qdrant_setup.py
────────────────────────────
Qdrant database connection helpers and a one-shot seed function that
populates the "interview_rubrics" collection with 3 sample rubrics.

Chunking Strategy — Sentence-Based with Sliding Window
───────────────────────────────────────────────────────
Interview transcripts can easily exceed the 512-token limit of
all-MiniLM-L6-v2. To prevent silent truncation and improve rubric
matching accuracy, the search_rubrics() function:

  1. Splits the transcript into overlapping sentence windows
     (4 sentences per chunk, 1 sentence overlap).
  2. Embeds each chunk independently.
  3. Searches Qdrant with every chunk vector.
  4. Returns the single highest-scoring rubric across all chunks.

This ensures that even a 30-minute interview (2000+ tokens) is fully
covered and the most relevant segment drives the rubric match.

Rubric seeding does NOT chunk — rubric ideal_answers are short
(< 150 tokens) and are best embedded as single coherent units.

Usage (standalone seed):
    python -m src.database.qdrant_setup
"""

from __future__ import annotations

import logging
import re
import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "interview_rubrics"
VECTOR_DIM       = 384      # all-MiniLM-L6-v2 output dimension
RUBRIC_SCORE_KEY = "rubric"

# ── Chunking parameters (sentence-based sliding window) ───────────────────────
CHUNK_SENTENCES = 4         # sentences per chunk window
CHUNK_OVERLAP   = 1         # sentences shared between consecutive windows
MIN_CHUNK_WORDS = 8         # discard chunks shorter than this (noise filter)

# ── Sample rubric payloads ─────────────────────────────────────────────────────
SEED_RUBRICS: list[dict] = [
    {
        "question": "What is the difference between a Python List and a Tuple?",
        "ideal_answer": (
            "Lists are mutable ordered sequences; elements can be added, removed, or changed "
            "after creation. Tuples are immutable; once created their contents cannot change. "
            "Tuples are generally faster and use less memory. Lists are preferred for homogeneous "
            "data that may grow; tuples for fixed heterogeneous records or as dictionary keys."
        ),
        "key_concepts": ["mutability", "immutability", "memory efficiency", "hashability", "use cases"],
        "difficulty": "beginner",
        "category": "Python Fundamentals",
    },
    {
        "question": "What is a Vector Database and why is it useful for AI applications?",
        "ideal_answer": (
            "A vector database stores high-dimensional numerical vectors (embeddings) and provides "
            "efficient approximate nearest-neighbour (ANN) search. It is useful for AI applications "
            "such as semantic search, recommendation systems, and RAG pipelines because it allows "
            "retrieval of semantically similar content rather than exact keyword matches. Examples "
            "include Qdrant, Pinecone, Weaviate, and Chroma."
        ),
        "key_concepts": ["embeddings", "ANN search", "semantic similarity", "RAG", "scalability"],
        "difficulty": "intermediate",
        "category": "AI Infrastructure",
    },
    {
        "question": "Explain the principles of REST API design.",
        "ideal_answer": (
            "REST (Representational State Transfer) is an architectural style for distributed "
            "hypermedia systems. Core principles: (1) Stateless — each request must contain all "
            "information needed; (2) Client-Server separation; (3) Uniform Interface — resources "
            "identified by URIs, manipulated via standard HTTP verbs (GET, POST, PUT, DELETE); "
            "(4) Layered System; (5) Cacheable responses. Good REST APIs use nouns for resources, "
            "proper HTTP status codes, versioning, and consistent JSON payloads."
        ),
        "key_concepts": ["stateless", "HTTP verbs", "resources", "URIs", "status codes", "versioning"],
        "difficulty": "intermediate",
        "category": "Software Engineering",
    },
]


# ── Client factory ─────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """Return a persistent local QdrantClient instance."""
    return QdrantClient(path=settings.qdrant_db_path)


# ── Seed helper ────────────────────────────────────────────────────────────────

def seed_rubrics_collection() -> None:
    """
    Idempotent seed function.
    - Creates the 'interview_rubrics' collection if it does not exist.
    - Embeds and upserts the 3 sample rubrics.
    - Safe to call multiple times (uses fixed deterministic UUIDs).
    """
    logger.info("Initialising Qdrant seed routine …")

    client = get_qdrant_client()

    # ── Ensure collection exists ──────────────────────────────────────────────
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'.", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists — skipping creation.", COLLECTION_NAME)

    # ── Embed rubrics ─────────────────────────────────────────────────────────
    logger.info("Loading embedding model for seed …")
    embedder = SentenceTransformer(settings.embedding_model)

    texts = [r["ideal_answer"] for r in SEED_RUBRICS]
    vectors = embedder.encode(texts, show_progress_bar=False).tolist()

    # ── Upsert with deterministic UUIDs so the function is idempotent ─────────
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, rubric["question"])),
            vector=vector,
            payload=rubric,
        )
        for rubric, vector in zip(SEED_RUBRICS, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Upserted %d rubric(s) into '%s'.", len(points), COLLECTION_NAME)

    # Cleanup embedder
    del embedder
    logger.info("Seed completed successfully.")


# ── Chunking utility ───────────────────────────────────────────────────────────

def _sentence_chunks(text: str) -> list[str]:
    """
    Split *text* into overlapping sentence-window chunks.

    Strategy: Sentence-Based Sliding Window
    ────────────────────────────────────────
    Why sentence-based?
      - Interview transcripts are spoken language → natural sentence units.
      - Preserves complete thoughts; never cuts mid-idea.
      - Overlap retains cross-boundary context so no information is lost
        at chunk edges.

    Why sliding window (overlap)?
      - A key concept may span the boundary between two chunks.
      - Without overlap, boundary sentences are under-represented in search.
      - 1-sentence overlap is enough for short rubric matching without
        creating redundant duplicate embeddings.

    Parameters
    ----------
    text : str
        Raw transcript or any text to be chunked.

    Returns
    -------
    list[str]
        List of chunk strings, each covering CHUNK_SENTENCES sentences
        with CHUNK_OVERLAP sentences shared with the adjacent chunk.

    Example
    -------
    Sentences: [S1, S2, S3, S4, S5, S6, S7]
    CHUNK_SENTENCES=4, CHUNK_OVERLAP=1, step=3

    Chunk 1 → S1 S2 S3 S4
    Chunk 2 → S4 S5 S6 S7   ← S4 repeated (overlap)
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text.strip())

    # Clean and filter noise sentences
    sentences = [s.strip() for s in raw if len(s.split()) >= MIN_CHUNK_WORDS]

    # If too short for windowing just return the whole text as one chunk
    if len(sentences) <= CHUNK_SENTENCES:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    step = CHUNK_SENTENCES - CHUNK_OVERLAP      # = 3 with defaults

    for i in range(0, len(sentences), step):
        window = sentences[i : i + CHUNK_SENTENCES]
        if window:
            chunks.append(" ".join(window))

    logger.debug(
        "[Chunking] %d sentence(s) → %d chunk(s) "
        "(window=%d, overlap=%d, step=%d)",
        len(sentences), len(chunks),
        CHUNK_SENTENCES, CHUNK_OVERLAP, step,
    )
    return chunks


# ── Retrieval helper ───────────────────────────────────────────────────────────

def search_rubrics(query_text: str, top_k: int = 1) -> list[dict]:
    """
    Chunk the transcript, embed each chunk, search Qdrant with every
    chunk vector, and return the single best-matching rubric payload.

    Why chunk before searching?
    ───────────────────────────
    all-MiniLM-L6-v2 has a hard 512-token limit. A real interview
    transcript can be 800-3000+ tokens. Without chunking, the model
    silently truncates the input — the second half of a 10-minute
    interview is never searched. Chunking ensures every spoken sentence
    contributes to rubric matching.

    Flow
    ────
    transcript (full text)
        │
        ▼  _sentence_chunks()
        │
    [chunk_1, chunk_2, ..., chunk_n]   ← each ≤ ~120 tokens
        │
        ▼  embedder.encode() per chunk
        │
    [vec_1, vec_2, ..., vec_n]
        │
        ▼  client.search() per vector
        │
    all_hits (flattened)
        │
        ▼  max(score)
        │
    best_rubric_payload

    Parameters
    ----------
    query_text : str
        Full transcript text from the interview.
    top_k : int
        Number of rubric candidates to retrieve per chunk (default 1).

    Returns
    -------
    list[dict]
        Single-element list containing the best rubric payload,
        or empty list if collection is empty.
    """
    client  = get_qdrant_client()
    embedder = SentenceTransformer(settings.embedding_model)

    # ── Chunk the transcript ──────────────────────────────────────────────────
    chunks = _sentence_chunks(query_text)

    # Fallback: if transcript is too short to chunk, use it directly
    if not chunks:
        chunks = [query_text.strip()]

    logger.info("[VectorSearch] Searching %d chunk(s) against rubrics.", len(chunks))

    # ── Embed all chunks at once (single encoder pass = faster) ───────────────
    vectors = embedder.encode(chunks, show_progress_bar=False).tolist()
    del embedder  # free memory immediately after encoding

    # ── Search Qdrant with each chunk vector ──────────────────────────────────
    all_hits = []
    for idx, vector in enumerate(vectors):
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        for hit in hits:
            logger.debug(
                "[VectorSearch] Chunk %d/%d → rubric='%s' score=%.4f",
                idx + 1, len(vectors),
                hit.payload.get("question", "?")[:50],
                hit.score,
            )
        all_hits.extend(hits)

    if not all_hits:
        logger.warning("[VectorSearch] No rubric matches found.")
        return []

    # ── Return the single highest-scoring rubric across all chunks ────────────
    best = max(all_hits, key=lambda h: h.score)
    logger.info(
        "[VectorSearch] Best match → '%s' (score=%.4f)",
        best.payload.get("question", "?")[:60],
        best.score,
    )
    return [best.payload]


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    seed_rubrics_collection()
    print("✅  Database seeded successfully.")

from __future__ import annotations


class CvRagError(Exception):
    """Base exception for all cv-rag errors."""


class IngestError(CvRagError):
    """Raised when the ingest pipeline encounters a non-recoverable error."""


class RetrievalError(CvRagError):
    """Raised when retrieval fails or finds no relevant sources."""


class GenerationError(CvRagError):
    """Raised when LLM generation fails (subprocess error, empty output)."""


class CitationValidationError(CvRagError):
    """Raised when answer citation validation fails after repair."""

    def __init__(self, reason: str, draft: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.draft = draft

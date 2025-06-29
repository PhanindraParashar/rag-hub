"""
Pydantic schemas for BM25‑based keyword extraction.
Place this file in the `schemas` package
(e.g. `project_root/schemas/keywords.py`).
Compatible with both **Pydantic v1** and **v2**.
"""

from __future__ import annotations
from typing import Set, Optional, List

from pydantic import BaseModel, Field, validator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------

class KeywordExtractionConfig(BaseModel):
    """Validated hyper‑parameters for BM25 keyword extraction.

    The defaults give balanced results for most English prose; adjust per
    project by instantiating with keyword arguments or loading from JSON.
    """

    page_splitter: str = Field("\n\n------\n\n", min_length=1)
    score_threshold: float = Field(
        1.2,
        ge=0.0,
        description="Keep tokens whose BM25 score ≥ this value",
    )
    max_keywords: Optional[int] = Field(
        12, gt=0, description="Hard cap per page; None = unlimited"
    )
    lemmatize: bool = Field(
        True, description="Turn spaCy lemmatisation on/off if model present"
    )
    min_chars: int = Field(
        3, ge=1, description="Discard tokens shorter than this many characters"
    )
    k1: float = Field(1.5, gt=0)
    b: float = Field(0.75, gt=0, lt=1)
    stopwords: Set[str] = Field(default_factory=lambda: set(ENGLISH_STOP_WORDS))
    extra_stopwords: Set[str] = Field(
        default_factory=set, description="Custom domain‑specific stop‑words"
    )

    # -------------------------------------------------------------------
    # Validators – designed to work under both Pydantic 1.x and 2.x.
    # In Pydantic v2 you would normally use @field_validator; here we stick to
    # @validator for back‑compatibility.
    # -------------------------------------------------------------------

    @validator("extra_stopwords", pre=True, always=True)
    def _normalise_extras(
        cls, v: Set[str] | List[str] | None  # type: ignore[arg-type]
    ) -> Set[str]:
        """Ensure custom stop‑words are lower‑cased for reliable matching."""
        if v is None:
            return set()
        return {w.lower() for w in (set(v) if not isinstance(v, set) else v)}


# ---------------------------------------------------------------------------
# Page‑level result schema (optional helper)
# ---------------------------------------------------------------------------

class PageKeywords(BaseModel):
    """Keywords detected for a single page."""

    page_number: int
    keywords: List[str]

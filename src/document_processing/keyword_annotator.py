"""
BM25KeywordAnnotator
====================
Split long text into pages, compute **BM25** scores, and attach a dynamic set
of high‑scoring keywords to each page's metadata. Designed for RAG pipelines.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from typing import List, Dict, Union, Optional

import numpy as np
from langchain_core.documents import Document

from schemas.keywords import KeywordExtractionConfig, PageKeywords

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
except Exception:
    _NLP = None
    warnings.warn(
        "spaCy or its English model not found – lemmatisation disabled. "
        "Run `pip install spacy && python -m spacy download en_core_web_sm` "
        "to enable it."
    )


class BM25KeywordAnnotator:
    """Annotate each page of a document with BM25‑ranked keywords."""

    TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

    def __init__(self, cfg: KeywordExtractionConfig | dict | None = None):
        """Create a new annotator.

        Parameters
        ----------
        cfg : KeywordExtractionConfig | dict | None
            – Pass a fully‑specified ``KeywordExtractionConfig``.
            – Pass a *dict* with fields to override defaults.
            – Pass *None* to use defaults unchanged.
        """

        self.cfg = KeywordExtractionConfig.model_validate(cfg or {})
        self._stopwords = self.cfg.stopwords.union(self.cfg.extra_stopwords)
        self._use_lemma = self.cfg.lemmatize and _NLP is not None

    # ---------------------------------------------------------------------
    def __call__(self, text: str, document_name: str) -> List[Document]:
        """Process *one* document and return a list[Document] (one per page)."""

        pages, tokenised = self._split_and_tokenise(text)
        idf = self._compute_idf(tokenised)
        avgdl = np.mean([len(toks) for toks in tokenised]) or 1.0

        docs: list[Document] = []
        for i, (page_text, tokens) in enumerate(zip(pages, tokenised), start=1):
            info = self._page_keywords(tokens, idf, len(tokens), avgdl, i)
            docs.append(
                Document(
                    page_content=page_text.strip(),
                    metadata={
                        "source": document_name,
                        "page_number": i,
                        "keywords": info.keywords,
                    },
                )
            )
        return docs

    # --------------------- helpers --------------------------------------
    def _split_and_tokenise(self, text: str):
        pages = text.split(self.cfg.page_splitter)

        if self._use_lemma:
            lemmas: List[List[str]] = []
            for doc in _NLP.pipe(pages, batch_size=32):
                lemmas.append(
                    [
                        t.lemma_.lower()
                        for t in doc
                        if t.lemma_
                        and t.lemma_ not in ("", "-PRON-")
                        and len(t.lemma_) >= self.cfg.min_chars
                        and t.lemma_.isalpha()
                        and t.lemma_.lower() not in self._stopwords
                    ]
                )
            return pages, lemmas

        tokenised = [
            [
                tok
                for tok in self.TOKEN_RE.findall(page.lower())
                if len(tok) >= self.cfg.min_chars and tok not in self._stopwords
            ]
            for page in pages
        ]
        return pages, tokenised

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_idf(tokenised_pages: List[List[str]]):
        N = len(tokenised_pages)
        df = Counter()
        for tokens in tokenised_pages:
            df.update(set(tokens))
        return {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def _page_keywords(
        self,
        tokens: List[str],
        idf: dict[str, float],
        dl: int,
        avgdl: float,
        page_number: int,
    ) -> PageKeywords:
        """Compute keyword list for one page."""
        tf = Counter(tokens)
        k1, b = self.cfg.k1, self.cfg.b
        scores: dict[str, float] = {}

        for term, freq in tf.items():
            if term not in idf:
                continue
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score = idf[term] * freq * (k1 + 1) / denom
            if score >= self.cfg.score_threshold:
                scores[term] = score

        if not scores:
            # guarantee at least one token (fallback = highest tf)
            fallback = max(tf, key=tf.get, default=None)
            scores = {fallback: 0.0} if fallback else {}

        ordered = sorted(scores, key=scores.get, reverse=True)
        if self.cfg.max_keywords:
            ordered = ordered[: self.cfg.max_keywords]

        return PageKeywords(page_number=page_number, keywords=ordered)

class TFIDFKeywordAnnotator:
    """Annotate each page with TF‑IDF‑ranked keywords (per‑document IDF)."""

    TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

    def __init__(self, cfg: KeywordExtractionConfig | dict | None = None):
        self.cfg = KeywordExtractionConfig.model_validate(cfg or {})
        self._stopwords = self.cfg.stopwords.union(self.cfg.extra_stopwords)
        self._use_lemma = self.cfg.lemmatize and _NLP is not None

    # ------------------------------------------------------------------
    def __call__(self, text: str, document_name: str) -> List[Document]:
        pages, tokenised = self._split_and_tokenise(text)
        idf = self._compute_idf(tokenised)

        docs: list[Document] = []
        for i, (page_text, tokens) in enumerate(zip(pages, tokenised), 1):
            info = self._page_keywords(tokens, idf, i)
            docs.append(
                Document(
                    page_content=page_text.strip(),
                    metadata={
                        "source": document_name,
                        "page_number": i,
                        "keywords": info.keywords,
                    },
                )
            )
        return docs

    # -------------------- tokenisation -----------------------------
    def _split_and_tokenise(self, text: str):
        pages = text.split(self.cfg.page_splitter)
        if self._use_lemma:
            lemmas: List[List[str]] = []
            for doc in _NLP.pipe(pages, batch_size=32):
                lemmas.append(
                    [
                        t.lemma_.lower()
                        for t in doc
                        if t.lemma_
                        and t.lemma_ not in ("", "-PRON-")
                        and len(t.lemma_) >= self.cfg.min_chars
                        and t.lemma_.isalpha()
                        and t.lemma_.lower() not in self._stopwords
                    ]
                )
            return pages, lemmas

        tokenised = [
            [
                tok
                for tok in self.TOKEN_RE.findall(pg.lower())
                if len(tok) >= self.cfg.min_chars and tok not in self._stopwords
            ]
            for pg in pages
        ]
        return pages, tokenised

    # ------------------------- TF‑IDF core ---------------------------
    @staticmethod
    def _compute_idf(tokenised_pages: List[List[str]]) -> Dict[str, float]:
        """Inverse document freq using ln(N / df)."""
        N = len(tokenised_pages)
        df = Counter()
        for tokens in tokenised_pages:
            df.update(set(tokens))
        return {t: math.log(N / f) if f else 0.0 for t, f in df.items()}

    def _page_keywords(
        self, tokens: List[str], idf: Dict[str, float], page_number: int
    ) -> PageKeywords:
        tf = Counter(tokens)
        scores: Dict[str, float] = {}
        for term, freq in tf.items():
            if term not in idf:
                continue
            score = freq * idf[term]
            if score >= self.cfg.score_threshold:
                scores[term] = score

        if not scores:
            # fallback: at least one token
            fallback = max(tf, key=tf.get, default=None)
            scores = {fallback: 0.0} if fallback else {}

        ordered = sorted(scores, key=scores.get, reverse=True)
        if self.cfg.max_keywords:
            ordered = ordered[: self.cfg.max_keywords]

        return PageKeywords(page_number=page_number, keywords=ordered)


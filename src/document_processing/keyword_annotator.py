"""
Keyword Annotators & Query Processor
====================================
This module provides classes to extract keywords from text using BM25 and
TF-IDF algorithms and to process search queries consistently.

- `BM25KeywordAnnotator`: Splits text into pages, computes BM25 scores,
  and attaches high-scoring keywords to each page's metadata.
- `TFIDFKeywordAnnotator`: Does the same using the TF-IDF algorithm.
- `QueryProcessor`: Processes raw search query strings using the exact same
  tokenization/lemmatization pipeline as the annotators.

All processing classes inherit from a `BaseTextProcessor` to ensure
consistent text handling for indexing and searching, suitable for RAG pipelines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Union

import numpy as np
from langchain_core.documents import Document

from schemas.keywords import KeywordExtractionConfig, PageKeywords

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
except Exception:
    _NLP = None
    print(
        "spaCy or its English model not found – lemmatisation will be disabled. "
        "Run `pip install spacy && python -m spacy download en_core_web_sm` "
        "to enable it."
    )

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
# --------------------------------------------------------------------------
# Base Class for Centralized Text Processing
# --------------------------------------------------------------------------
class BaseTextProcessor:
    """
    A base class to centralize text processing logic (tokenization,
    lemmatization, stopword removal) to ensure consistency across
    document indexing and query processing.
    """
    TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

    def __init__(self, cfg: KeywordExtractionConfig | dict | None = None):
        """
        Initializes the processor with a configuration.

        Parameters
        ----------
        cfg : KeywordExtractionConfig | dict | None
            - A fully-specified `KeywordExtractionConfig`.
            - A dict with fields to override defaults.
            - None to use default settings.
        """
        self.cfg = KeywordExtractionConfig.model_validate(cfg or {})
        self._stopwords = self.cfg.stopwords.union(self.cfg.extra_stopwords)
        self._use_lemma = self.cfg.lemmatize and _NLP is not None

    def _process_text_to_tokens(self, text: str) -> List[str]:
        """
        Applies the full processing pipeline (lemmatization or simple
        tokenization) to a single string of text.
        """
        if self._use_lemma:
            # Process with spaCy for lemmatization
            doc = _NLP(text)
            return [
                t.lemma_.lower()
                for t in doc
                if t.lemma_
                and t.lemma_ not in ("", "-PRON-") # Filter out pronouns and empty lemmas
                and len(t.lemma_) >= self.cfg.min_chars
                and t.lemma_.isalpha() # Ensure tokens are alphabetic
                and t.lemma_.lower() not in self._stopwords
            ]

        # Fallback to simple regex tokenization if lemmatization is disabled
        return [
            tok
            for tok in self.TOKEN_RE.findall(text.lower())
            if len(tok) >= self.cfg.min_chars and tok not in self._stopwords
        ]

    def _split_and_tokenise_pages(self, text: str) -> Tuple[List[str], List[List[str]]]:
        """Splits a document into pages and tokenises each one."""
        pages = text.split(self.cfg.page_splitter)
        tokenised_pages = [self._process_text_to_tokens(page) for page in pages]
        return pages, tokenised_pages

# --------------------------------------------------------------------------
# Query Processor for Consistent Search Term Handling
# --------------------------------------------------------------------------
class QueryProcessor(BaseTextProcessor):
    """
    Processes a raw search query string using the same pipeline as the
    document annotators to ensure tokens match for searching.
    """
    def process(self, query: str) -> List[str]:
        """
        Processes a raw query string into a list of cleaned, ready-to-search tokens.

        Parameters
        ----------
        query : str
            The raw search query from the user.

        Returns
        -------
        List[str]
            A list of processed tokens.
        """
        return self._process_text_to_tokens(query)

# --------------------------------------------------------------------------
# Keyword Annotator Implementations
# --------------------------------------------------------------------------
class BM25KeywordAnnotator(BaseTextProcessor):
    """
    Annotate each page of a document with BM25-ranked keywords.
    Inherits text processing from BaseTextProcessor.
    """
    def __call__(self, text: str, document_name: str) -> List[Document]:
        """Process *one* document and return a list[Document] (one per page)."""
        pages, tokenised_pages = self._split_and_tokenise_pages(text)
        idf = self._compute_idf(tokenised_pages)
        # Calculate avgdl, avoiding division by zero for empty documents
        avgdl = np.mean([len(toks) for toks in tokenised_pages if toks]) or 1.0

        docs: list[Document] = []
        for i, (page_text, tokens) in enumerate(zip(pages, tokenised_pages), start=1):
            if not tokens:  # Do not process pages that yield no tokens
                continue
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

    @staticmethod
    def _compute_idf(tokenised_pages: List[List[str]]) -> Dict[str, float]:
        N = len(tokenised_pages)
        # Correctly count document frequency: a term counts once per page
        df = Counter(token for tokens in tokenised_pages for token in set(tokens))
        return {t: math.log((N - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def _page_keywords(
        self,
        tokens: List[str],
        idf: dict[str, float],
        dl: int,
        avgdl: float,
        page_number: int,
    ) -> PageKeywords:
        """Compute keyword list for one page using BM25 scoring."""
        tf = Counter(tokens)
        k1, b = self.cfg.k1, self.cfg.b
        scores: dict[str, float] = {}

        for term, freq in tf.items():
            if term not in idf:
                continue
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score = (idf[term] * freq * (k1 + 1)) / denom
            if score >= self.cfg.score_threshold:
                scores[term] = score

        if not scores and tokens:
            # Guarantee at least one token (fallback = highest tf)
            fallback = max(tf, key=tf.get)
            scores = {fallback: 0.0}

        ordered = sorted(scores, key=scores.get, reverse=True)
        if self.cfg.max_keywords is not None:
            ordered = ordered[:self.cfg.max_keywords]

        return PageKeywords(page_number=page_number, keywords=ordered)


class TFIDFKeywordAnnotator(BaseTextProcessor):
    """
    Annotate each page with TF-IDF-ranked keywords (per-document IDF).
    Inherits text processing from BaseTextProcessor.
    """
    def __call__(self, text: str, document_name: str) -> List[Document]:
        """Process one document and return a list[Document] (one per page)."""
        pages, tokenised_pages = self._split_and_tokenise_pages(text)
        idf = self._compute_idf(tokenised_pages)

        docs: list[Document] = []
        for i, (page_text, tokens) in enumerate(zip(pages, tokenised_pages), 1):
            if not tokens: # Do not process pages that yield no tokens
                continue
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

    @staticmethod
    def _compute_idf(tokenised_pages: List[List[str]]) -> Dict[str, float]:
        """Inverse document freq using ln(N / df)."""
        N = len(tokenised_pages)
        df = Counter(token for tokens in tokenised_pages for token in set(tokens))
        return {t: math.log(N / f) if f > 0 else 0.0 for t, f in df.items()}

    def _page_keywords(
        self, tokens: List[str], idf: Dict[str, float], page_number: int
    ) -> PageKeywords:
        """Compute keyword list for one page using TF-IDF scoring."""
        tf = Counter(tokens)
        scores: Dict[str, float] = {}
        for term, freq in tf.items():
            if term not in idf:
                continue
            score = freq * idf[term]
            if score >= self.cfg.score_threshold:
                scores[term] = score

        if not scores and tokens:
            # Fallback: guarantee at least one token
            fallback = max(tf, key=tf.get)
            scores = {fallback: 0.0}

        ordered = sorted(scores, key=scores.get, reverse=True)
        if self.cfg.max_keywords is not None:
            ordered = ordered[:self.cfg.max_keywords]

        return PageKeywords(page_number=page_number, keywords=ordered)

class KeyBertAnnotator(BaseTextProcessor):
    def __init__(
        self,
        *,
        use_ngrams: bool = False,
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        top_n: int = 15,
        diversity: float = 0.9,
        STOPWORDS: frozenset = ENGLISH_STOP_WORDS,
    ) -> None:
        super().__init__()

        self._model = KeyBERT()
        self._vectorizer = KeyphraseCountVectorizer(stop_words=self._stopwords)

        self._use_ngrams = use_ngrams
        self._ngram_range = keyphrase_ngram_range
        self._top_n = top_n
        self._diversity = diversity

        self.STOPWORDS = STOPWORDS
        self.kw_model = KeyBERT()
        self.vectorizer = KeyphraseCountVectorizer()
        self._lemmatised_stops = [self._process_text_to_tokens(i) for i in list(self.STOPWORDS)]

    # ------------------------------------------------------------------ #
    def _extract_keywords(self, text: str) -> List[str]:
        if self._use_ngrams:
            pairs = self._model.extract_keywords(
                text,
                vectorizer=self._vectorizer,
                keyphrase_ngram_range=self._ngram_range,
                use_mmr=True,
                diversity=self._diversity,
                top_n=self._top_n,
            )
        else:
            pairs = self._model.extract_keywords(
                text, vectorizer=self._vectorizer, top_n=self._top_n
            )

        processed_kw = [self._process_text_to_tokens(keyword[0]) for keyword in pairs]
        final_keywords = []
        for i in processed_kw:
            final_keywords.extend(i)
        final_keywords = list(set(final_keywords))
        return [kw for kw in final_keywords if kw not in self._lemmatised_stops]

    # ------------------------------------------------------------------ #
    def __call__(
        self, text_or_docs: Union[str, List[Document]], document_name: str
    ) -> List[Document]:
        # -------------------------------------------------------------- #
        # Case 1: Already split into page-Documents
        # -------------------------------------------------------------- #
        if isinstance(text_or_docs, list):
            out: List[Document] = []
            for page_doc in text_or_docs:
                txt = page_doc.page_content.strip()
                if not txt:
                    continue
                kw = self._extract_keywords(txt)
                meta = dict(page_doc.metadata)
                meta.setdefault("source", document_name)
                meta["keywords"] = kw
                out.append(Document(page_content=txt, metadata=meta))
            return out

        # -------------------------------------------------------------- #
        # Case 2: Raw text string (do our own page split)
        # -------------------------------------------------------------- #
        pages, _ = self._split_and_tokenise_pages(text_or_docs)

        docs: List[Document] = []
        for i, page_text in enumerate(pages, start=1):
            stripped = page_text.strip()
            if not stripped:
                continue
            kw = self._extract_keywords(stripped)
            docs.append(
                Document(
                    page_content=stripped,
                    metadata={
                        "source": document_name,
                        "page_number": i,
                        "keywords": kw,
                    },
                )
            )
        return docs
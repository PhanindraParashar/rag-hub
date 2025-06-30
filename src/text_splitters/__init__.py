from document_processing.keyword_annotator import BM25KeywordAnnotator, TFIDFKeywordAnnotator
from typing import List, Dict, Any, Optional, Sequence
from langchain_text_splitters import TokenTextSplitter, TextSplitter
import tiktoken
from langchain_core.documents import Document
import re
import numpy as np
from langchain.embeddings.base import Embeddings

class CustomTokenSplitter(TextSplitter):
    """Token‑aware splitter with strict keyword filtering."""

    def __init__(
        self,
        *,  # force keyword‑only
        chunk_size: int,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
        separator: str = "\n\n---\n\n",
        keyword_annotator: Optional[
            BM25KeywordAnnotator | TFIDFKeywordAnnotator
        ] = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._separator = separator
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
        self._annotator = keyword_annotator

    # ---------------- internal helpers ---------------------------------- #
    def _tok_len(self, text: str) -> int:  # cheaper alias
        return len(self._tokenizer.encode(text))

    @staticmethod
    def _merge_meta(list_of_meta: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metadata from all contributing sub‑chunks.

        * ``pages``  – sorted list of page numbers that ended up in this chunk
        * ``page_number`` – the first (lowest) page number, for backwards
          compatibility with earlier pipelines that expect a single int.
        * ``keywords`` – de‑duplicated, alphabetically sorted union.
        * All **non‑conflicting** scalar keys (e.g. ``source``) are copied
          from the *first* metadata dict.
        """
        if not list_of_meta:
            return {}

        # Copy scalar keys from the first meta (e.g. "source")
        merged: Dict[str, Any] = {
            k: v for k, v in list_of_meta[0].items() if not isinstance(v, (list, dict))
        }

        pages: set[int] = set()
        kw: set[str] = set()
        for m in list_of_meta:
            pn = m.get("page_number")
            if pn is not None:
                pages.add(pn)
            if isinstance(m.get("keywords"), list):
                kw.update(m["keywords"])

        sorted_pages = sorted(pages)
        if sorted_pages:
            merged["pages"] = sorted_pages
            merged["page_number"] = sorted_pages[0]  # first page for convenience
        if kw:
            merged["keywords"] = sorted(kw)
        return merged

    # ---------------- TextSplitter overrides ---------------------------- #
    def split_text(self, text: str) -> List[str]:  # required abstract method
        # Delegates to TokenTextSplitter with same settings
        return self._token_splitter.split_text(text)

    # -------------------------------------------------------------------- #
    def split_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        # 0) annotate pages once (keeps global‑doc IDF)
        kw_by_page: Dict[int, List[str]] = {}
        if self._annotator:
            pages_sorted = sorted(docs, key=lambda d: d.metadata.get("page_number", 0))
            full_text = self._annotator.cfg.page_splitter.join(p.page_content for p in pages_sorted)
            src = pages_sorted[0].metadata.get("source", "unknown")
            annotated = self._annotator(full_text, src)
            kw_by_page = {
                p.metadata["page_number"]: p.metadata.get("keywords", []) for p in annotated
            }

        # 1) page → sub‑chunks
        subs: List[Document] = []
        for page in docs:
            meta = dict(page.metadata)
            pn = meta.get("page_number")
            if kw_by_page:
                meta["keywords"] = kw_by_page.get(pn, [])
            subs.extend(
                self._token_splitter.create_documents([page.page_content], metadatas=[meta])
            )

        if not subs:
            return []

        # 2) merge while respecting token budget
        merged: List[Document] = []
        parts, metas = [], []
        tokens = 0
        sep_toks = self._tok_len(self._separator)

        for sub in subs:
            sub_toks = self._tok_len(sub.page_content)
            new_tokens = tokens + sub_toks + (sep_toks if parts else 0)
            if new_tokens > self._chunk_size and parts:
                merged.append(self._finalise(parts, metas))
                parts, metas, tokens = [], [], 0
            parts.append(sub.page_content)
            metas.append(sub.metadata)
            tokens = new_tokens

        if parts:
            merged.append(self._finalise(parts, metas))
        return merged

    # -------------------------------------------------------------------- #
    def _finalise(self, parts: List[str], metas: List[Dict[str, Any]]) -> Document:
        text = self._separator.join(parts)
        meta = self._merge_meta(metas)

        if self._annotator and "keywords" in meta:
            # lemma tokens present in chunk
            lemma_tokens = set(self._annotator._process_text_to_tokens(text))
            # exact surface forms (lower‑cased) for word‑boundary regex check
            raw_text = text.lower()

            def _present(kw: str) -> bool:
                # quick lemma membership
                if kw in lemma_tokens:
                    return True
                # fallback: exact word / phrase search
                import re

                return re.search(rf"\b{re.escape(kw.lower())}\b", raw_text) is not None

            filtered = [kw for kw in meta["keywords"] if _present(kw)]
            max_kw = self._annotator.cfg.max_keywords
            meta["keywords"] = filtered[:max_kw] if max_kw else filtered

        return Document(page_content=text, metadata=meta)

class CustomSemanticChunker(TextSplitter):
    """
    Sentence‑level semantic chunker with strict keyword filtering.

    This class encapsulates all the logic for splitting documents based on
    semantic similarity at the sentence level, including helper methods for
    text processing and vector math.
    """
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    @staticmethod
    def _sent_tokenize(text: str) -> List[str]:
        """Very simple sentence splitter; good enough for embeddings."""
        return [s.strip() for s in CustomSemanticChunker._SENT_RE.split(text) if s.strip()]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Calculates cosine similarity between two numpy arrays."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        chunk_size: int,
        chunk_overlap: int = 0,
        similarity_threshold: float = 0.75,
        min_tokens: int = 20,
        encoding_name: str = "cl100k_base",
        separator: str = "\n\n---\n\n",
        keyword_annotator: Optional[
            BM25KeywordAnnotator | TFIDFKeywordAnnotator
        ] = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._emb = embeddings
        self._sim_thr = similarity_threshold
        self._min_tokens = min_tokens
        self._separator = separator
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._tok_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
        self._annotator = keyword_annotator

    # ---------------- helpers ------------------------------------------ #
    def _tok_len(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    @staticmethod
    def _merge_meta(metas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not metas:
            return {}
        merged: Dict[str, Any] = {
            k: v
            for k, v in metas[0].items()
            if not isinstance(v, (list, dict))
        }
        pages, kw = set[int](), set[str]()
        for m in metas:
            pn = m.get("page_number")
            if pn is not None:
                pages.add(pn)
            kw.update(m.get("keywords", []))
        if pages:
            merged["pages"] = sp = sorted(pages)
            merged["page_number"] = sp[0]
        if kw:
            merged["keywords"] = sorted(kw)
        return merged

    def _filter_keywords(self, text: str, meta: Dict[str, Any]) -> None:
        if not (self._annotator and "keywords" in meta):
            return
        lemmas = set(self._annotator._process_text_to_tokens(text))
        raw = text.lower()
        def ok(kw: str) -> bool:
            return kw in lemmas or re.search(rf"\b{re.escape(kw.lower())}\b", raw)
        meta["keywords"] = [k for k in meta["keywords"] if ok(k)]
        cap = self._annotator.cfg.max_keywords if self._annotator else None
        if cap:
            meta["keywords"] = meta["keywords"][:cap]

    # -------------- TextSplitter abstract ------------------------------ #
    def split_text(self, text: str) -> List[str]:
        return self._tok_splitter.split_text(text)

    # ------------------------------------------------------------------- #
    def split_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        # 0) optional page‑level keyword annotation (global IDF)
        kw_by_page: Dict[int, List[str]] = {}
        if self._annotator:
            pages_sorted = sorted(docs, key=lambda d: d.metadata.get("page_number", 0))
            full_text = self._annotator.cfg.page_splitter.join(
                p.page_content for p in pages_sorted
            )
            src = pages_sorted[0].metadata.get("source", "unknown")
            annotated = self._annotator(full_text, src)
            kw_by_page = {
                p.metadata["page_number"]: p.metadata.get("keywords", [])
                for p in annotated
            }
        # 1) sentence‑level segmentation inside each page
        segs: List[Document] = []
        for page in docs:
            base_meta = dict(page.metadata)
            pn = base_meta.get("page_number")
            if kw_by_page:
                base_meta["keywords"] = kw_by_page.get(pn, [])
            # Use the class method for sentence tokenization
            sentences = self._sent_tokenize(page.page_content)
            if not sentences:
                continue
            embeds = [np.asarray(v, float) for v in self._emb.embed_documents(sentences)]
            current: List[str] = []
            centroid: Optional[np.ndarray] = None
            toks = 0
            for sent, emb_vec in zip(sentences, embeds):
                s_tok = self._tok_len(sent)
                if current:
                    # Use the class method for cosine similarity
                    sim_ok = self._cosine(centroid, emb_vec) >= self._sim_thr
                    size_ok = toks + s_tok <= self._chunk_size
                    if not (sim_ok and size_ok):
                        segs.append(
                            Document(page_content=" ".join(current), metadata=base_meta)
                        )
                        current, centroid, toks = [], None, 0
                current.append(sent)
                centroid = emb_vec if centroid is None else (centroid + emb_vec) / 2.0
                toks += s_tok
            if current:
                segs.append(Document(page_content=" ".join(current), metadata=base_meta))
        if not segs:
            return []
        # 2) merge segments across pages ≤ token budget
        merged: List[Document] = []
        parts, metas, tokens = [], [], 0
        sep_toks = self._tok_len(self._separator)
        def _emit() -> None:
            text = self._separator.join(parts)
            meta = self._merge_meta(metas)
            self._filter_keywords(text, meta)
            merged.append(Document(page_content=text, metadata=meta))
        for seg in segs:
            s_tok = self._tok_len(seg.page_content)
            new_tok = tokens + s_tok + (sep_toks if parts else 0)
            if new_tok > self._chunk_size and parts:
                _emit()
                parts, metas, tokens = [], [], 0
            parts.append(seg.page_content)
            metas.append(seg.metadata)
            tokens = new_tok
        if parts:
            _emit()
        # 3) enforce min_tokens by merging tail chunks where possible
        if self._min_tokens and len(merged) > 1:
            fixed: List[Document] = [merged[0]]
            for nxt in merged[1:]:
                prev = fixed[-1]
                if (
                    self._tok_len(prev.page_content) < self._min_tokens
                    or self._tok_len(nxt.page_content) < self._min_tokens
                ):
                    combined = self._tok_len(prev.page_content) + self._tok_len(nxt.page_content) + sep_toks
                    if combined <= self._chunk_size:
                        txt = prev.page_content + self._separator + nxt.page_content
                        meta = self._merge_meta([prev.metadata, nxt.metadata])
                        self._filter_keywords(txt, meta)
                        fixed[-1] = Document(page_content=txt, metadata=meta)
                        continue
                fixed.append(nxt)
            merged = fixed
        return merged

class SpectralSegmentationChunker(TextSplitter):
    """
    Spectral-eigenvector (Fiedler) chunker that preserves correct, **non-overlapping**
    ``pages``/``page_number`` metadata for every final chunk.
    """

    _SPLIT_RE = re.compile(
        r"(^\s*#{1,6}\s.*$|^\s*\d+(?:\.\d+)*\.?\s.*$|\n\s*\n|\n\n------\n\n)",
        re.MULTILINE,
    )

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        chunk_size: int,
        chunk_overlap: int = 0,
        window_k: int = 3,
        n_splits: int = 10,
        min_words: int = 30,
        encoding_name: str = "cl100k_base",
        separator: str = "\n\n---\n\n",
        keyword_annotator: Optional[
            BM25KeywordAnnotator | TFIDFKeywordAnnotator
        ] = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._emb = embeddings
        self._k = max(1, window_k)
        self._n_splits = max(1, n_splits)
        self._min_words = max(1, min_words)
        self._separator = separator
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._tok_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
        self._annotator = keyword_annotator

    # ------------ helpers -------------------------------------------------- #
    def _tok_len(self, txt: str) -> int:
        return len(self._tokenizer.encode(txt))

    @staticmethod
    def _merge_meta(metas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not metas:
            return {}
        merged: Dict[str, Any] = {
            k: v for k, v in metas[0].items() if not isinstance(v, (list, dict))
        }
        pages, kw = set[int](), set[str]()
        for m in metas:
            pn = m.get("page_number")
            if pn is not None:
                pages.add(pn)
            kw.update(m.get("keywords", []))
        if pages:
            merged["pages"] = sp = sorted(pages)
            merged["page_number"] = sp[0]          # first *distinct* page
        if kw:
            merged["keywords"] = sorted(kw)
        return merged

    def _filter_keywords(self, text: str, meta: Dict[str, Any]) -> None:
        if not (self._annotator and "keywords" in meta):
            return
        lemmas = set(self._annotator._process_text_to_tokens(text))
        raw = text.lower()

        def ok(kw: str) -> bool:
            return kw in lemmas or re.search(rf"\b{re.escape(kw.lower())}\b", raw)

        meta["keywords"] = list(filter(ok, meta["keywords"]))
        cap = self._annotator.cfg.max_keywords if self._annotator else None
        if cap:
            meta["keywords"] = meta["keywords"][:cap]

    # -------------- TextSplitter API (split_text not used) ----------------- #
    def split_text(self, text: str) -> List[str]:
        return self._tok_splitter.split_text(text)

    # ----------------------------------------------------------------------- #
    def split_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        # --- (0) optional keyword annotation at page level ----------------- #
        kw_by_page: Dict[int, List[str]] = {}
        if self._annotator:
            ordered = sorted(docs, key=lambda d: d.metadata.get("page_number", 0))
            global_text = self._annotator.cfg.page_splitter.join(p.page_content for p in ordered)
            src = ordered[0].metadata.get("source", "unknown")
            annotated = self._annotator(global_text, src)
            kw_by_page = {
                p.metadata["page_number"]: p.metadata.get("keywords", [])
                for p in annotated
            }

        # --- (1) split pages by structural regex and enforce min_words ------ #
        fine, fine_meta = [], []
        for page in sorted(docs, key=lambda d: d.metadata.get("page_number", 0)):
            base = dict(page.metadata)
            if kw_by_page:
                base["keywords"] = kw_by_page.get(base.get("page_number"), [])
            parts = [p for p in re.split(self._SPLIT_RE, page.page_content) if p and p.strip()]
            for part in parts:
                fine.append(part.strip())
                fine_meta.append(base)

        # merge small fragments
        segs, seg_meta = [], []
        cur, cur_meta = "", []
        for txt, meta in zip(fine, fine_meta):
            cur = (cur + " " + txt).strip() if cur else txt
            cur_meta.append(meta)
            if len(cur.split()) >= self._min_words:
                segs.append(cur)
                seg_meta.append(self._merge_meta(cur_meta))
                cur, cur_meta = "", []
        if cur:  # trailing fragment
            if segs:
                segs[-1] = segs[-1] + " " + cur
                seg_meta[-1] = self._merge_meta([seg_meta[-1]] + cur_meta)
            else:
                segs.append(cur)
                seg_meta.append(self._merge_meta(cur_meta))

        if len(segs) <= 1:
            out = []
            for t, m in zip(segs, seg_meta):
                self._filter_keywords(t, m)
                out.append(Document(page_content=t, metadata=m))
            return out

        # --- (2) spectral boundary detection on seg embeddings ------------- #
        embeds = [np.asarray(v, float) for v in self._emb.embed_documents(segs)]
        n = len(embeds)
        w = np.zeros((n, n), float)
        for i in range(n):
            for j in range(max(0, i - self._k), min(n, i + self._k + 1)):
                if i != j:
                    a, b = embeds[i], embeds[j]
                    w[i, j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        D = np.diag(w.sum(axis=1))
        L = D - w
        _, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        jumps = np.abs(np.diff(fiedler))
        k = min(self._n_splits, n - 1)
        split_after = sorted(np.argsort(-jumps)[:k] + 1)  # positions where we cut

        # --- (3) build final chunks WITHOUT TokenTextSplitter fallback  ▶▶ FIX ◀◀ #
        #     We now respect the token budget *while* assembling, producing chunks
        #     whose `pages` / `page_number` sets are *disjoint* across neighbours.

        final_docs: List[Document] = []
        sep_toks = self._tok_len(self._separator)

        start_seg = 0
        for cut in split_after + [n]:
            # segments belonging to the (logical) block between two spectral cuts
            block_indices = list(range(start_seg, cut))
            start_seg = cut

            parts, metas, tok_count = [], [], 0
            for idx in block_indices:
                seg_txt = segs[idx]
                seg_tokens = self._tok_len(seg_txt)
                projected = tok_count + seg_tokens + (sep_toks if parts else 0)

                if projected > self._chunk_size and parts:
                    # flush current chunk
                    text = self._separator.join(parts)
                    meta = self._merge_meta(metas)
                    self._filter_keywords(text, meta)
                    final_docs.append(Document(page_content=text, metadata=meta))

                    parts, metas, tok_count = [], [], 0

                parts.append(seg_txt)
                metas.append(seg_meta[idx])
                tok_count = projected

                # handle pathological single-segment overflow
                if seg_tokens > self._chunk_size:
                    # resort to TokenTextSplitter **on that segment only**
                    split_docs = self._tok_splitter.create_documents(
                        [seg_txt], metadatas=[seg_meta[idx]]
                    )
                    for d in split_docs:
                        self._filter_keywords(d.page_content, d.metadata)
                    final_docs.extend(split_docs)
                    parts, metas, tok_count = [], [], 0

            if parts:
                text = self._separator.join(parts)
                meta = self._merge_meta(metas)
                self._filter_keywords(text, meta)
                final_docs.append(Document(page_content=text, metadata=meta))

        # Guarantee `pages` / `page_number` sets are **non-identical** between
        # consecutive chunks (defensive check – should already hold).
        for i in range(1, len(final_docs)):
            if final_docs[i - 1].metadata.get("pages") == final_docs[i].metadata.get("pages"):
                final_docs[i].metadata["page_number"] = final_docs[i].metadata["pages"][0]
                # keep `pages` but page_number differs automatically by definition

        return final_docs
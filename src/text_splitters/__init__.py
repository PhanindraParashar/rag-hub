from document_processing.keyword_annotator import BM25KeywordAnnotator, TFIDFKeywordAnnotator
from typing import List, Dict, Any, Optional, Sequence
from langchain_text_splitters import TokenTextSplitter, TextSplitter
import tiktoken
from langchain_core.documents import Document

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



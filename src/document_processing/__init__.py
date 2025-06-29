from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document
from document_processing.keyword_annotator import BM25KeywordAnnotator, TFIDFKeywordAnnotator


# --------------------------------------------------------------------------- #
class TextDirectoryLoader(BaseModel):
    """Load ``.txt`` files and (optionally) annotate each page with keywords.

    Attributes
    ----------
    directory : str | Path
        Folder containing the ``.txt`` files.
    annotator : Optional[Union['BM25KeywordAnnotator', 'TFIDFKeywordAnnotator']] | None
        Keyword annotator instance; None = skip annotation
    page_splitter : str
        Delimiter string used when *annotator* is ``None`` (i.e. we fall back
        to simple page splitting).
    file_suffix : str
        Only files ending with this suffix are processed.
    recursive : bool
        If *True*, walk sub‑directories as well.
    """

    directory: Path = Field(..., description="Directory containing .txt files")
    annotator: Optional[Union['BM25KeywordAnnotator', 'TFIDFKeywordAnnotator']] = Field(
        None, description="Keyword annotator instance; None = skip annotation"
    )
    page_splitter: str = Field("\n\n------\n\n", min_length=1)
    file_suffix: str = Field(".txt", min_length=1)
    recursive: bool = Field(False, description="Search sub‑folders as well")

    # allow arbitrary callable annotator objects
    model_config = {"arbitrary_types_allowed": True}

    # -------------------- validators ------------------------------------ #
    @validator("directory")
    def _dir_must_exist(cls, v: Path):
        if not v.exists() or not v.is_dir():
            raise ValueError(f"Directory not found: {v}")
        return v

    @validator("file_suffix")
    def _suffix_must_start_dot(cls, v: str):
        return v if v.startswith(".") else f".{v}"

    # ---------------------- public API ----------------------------------- #
    def load(self) -> Dict[str, List[Document]]:
        """Return ``{file_name: list[Document]}`` for every matching file."""
        paths = self._gather_paths()
        results: Dict[str, List[Document]] = {}

        for path in paths:
            text = self._read_file(path)
            file_name = path.name

            if self.annotator is not None:
                docs = self.annotator(text, document_name=file_name)
            else:
                docs = self._fallback_format(text, file_name)

            results[file_name] = docs
        return results

    # --------------------- internal helpers ----------------------------- #
    def _gather_paths(self) -> List[Path]:
        if self.recursive:
            return [p for p in self.directory.rglob(f"*{self.file_suffix}") if p.is_file()]
        return [p for p in self.directory.glob(f"*{self.file_suffix}") if p.is_file()]

    @staticmethod
    def _read_file(path: Path) -> str:
        with path.open("r", encoding="utf-8") as f:
            return f.read()

    def _fallback_format(self, text: str, document_name: str) -> List[Document]:
        """When no annotator is given, just split pages & wrap into Documents."""
        pages = text.split(self.page_splitter)
        return [
            Document(
                page_content=pg.strip(),
                metadata={"source": document_name, "page_number": i + 1},
            )
            for i, pg in enumerate(pages)
        ]

    # -------------------------------------------------------------------- #
    def __repr__(self):  # nicer console output
        return (
            f"TextDirectoryLoader(directory='{self.directory}', files={len(self._gather_paths())}, "
            f"annotator={'None' if self.annotator is None else self.annotator.__class__.__name__})"
        )

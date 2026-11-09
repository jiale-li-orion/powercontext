# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Whole-document PDF parsing, page-level evidence and exact-citation retrieval."""

from __future__ import annotations

import hashlib
import io
import re

from pydantic import BaseModel

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def _identity(value: PDFPageInput) -> str:
    # Store the original document ID in the Source; use its full cryptographic
    # digest for the bounded identity so long document paths remain supported.
    document_key = hashlib.sha256(value.document_id.encode('utf-8')).hexdigest()
    identity = f"pdf:{document_key}:{value.document_sha256}:{value.page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("PDF Source identity exceeds maximum length")
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Return every physical page, including empty pages, or fail atomically."""
    from pypdf import PdfReader

    if not value.document_id or not value.document_id.strip():
        raise ValueError("document_id must be nonblank")
    if not value.content.startswith(b"%PDF-"):
        raise ValueError("invalid PDF header")
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a nonempty password")
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=index,
                text=page.extract_text() or "",
            )
            for index, page in enumerate(reader.pages, start=1)
        )
        if not pages:
            raise ValueError("PDF has no pages")
        return pages
    except Exception as error:
        raise ValueError("PDF cannot be parsed completely") from error


class PDFPageAdapter:
    name = "pdf-page"
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, page: PDFPageInput, /) -> PDFPageSource:
        if page.page_number < 1 or not re.fullmatch(r"[0-9a-f]{64}", page.document_sha256):
            raise ValueError("invalid PDF page identity")
        identity = _identity(page)
        return PDFPageSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            **page.model_dump(),
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageAdapter(), projections=(PDFPageTextEvidenceProjection(),)
)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Retrieve persisted page citations, preserving exact Memory identities."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonblank")
    from powercontext.builtin.runtime import SearchMemoryRequest

    hits = (await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=limit))).hits
    return tuple(
        MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
        for hit in hits
    )


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDF_PAGE_SOURCE_DEFINITION",
    "parse_pdf_document",
    "search_pdf_pages",
]

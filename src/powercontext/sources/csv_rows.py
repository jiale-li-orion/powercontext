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

"""Capture CSV rows as stable, individually citable Source evidence."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


@dataclass(frozen=True, slots=True)
class CSVDocumentInput:
    document_id: str
    primary_key: str
    content: bytes


@dataclass(frozen=True, slots=True)
class CSVRowInput:
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Encode both components separately to avoid delimiter collisions."""
    if not isinstance(document_id, str) or not document_id.strip() or document_id != document_id.strip():
        raise ValueError("document_id must be nonempty and have no surrounding whitespace")
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary key value must be nonblank")
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("CSV row identity exceeds the Source ID length limit")
    return identity


def parse_csv_document(document: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate the entire offline document before exposing rows."""
    if not isinstance(document.primary_key, str) or not document.primary_key:
        raise ValueError("primary_key must be nonempty")
    csv_row_source_id(document.document_id, "valid")
    try:
        text = document.content.decode("utf-8-sig", errors="strict")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', strict=True)
        columns = tuple(next(reader))
        if not columns or any(not column.strip() for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV header must have unique, nonblank columns")
        if document.primary_key not in columns:
            raise ValueError("primary key must match a header column exactly")
        key_index = columns.index(document.primary_key)
        rows: list[CSVRowInput] = []
        seen: set[str] = set()
        for record in reader:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV data record width does not match header")
            values = tuple(record)
            primary_key_value = values[key_index]
            csv_row_source_id(document.document_id, primary_key_value)
            if primary_key_value in seen:
                raise ValueError("duplicate CSV primary key")
            seen.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=document.document_id,
                    primary_key=document.primary_key,
                    primary_key_value=primary_key_value,
                    columns=columns,
                    values=values,
                )
            )
        return tuple(rows)
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError("invalid CSV document") from error


class CSVRowAdapter:
    name = "csv-row"
    input_class = CSVRowInput
    source_class = CSVRowSource

    async def resolve(self, row: CSVRowInput, /) -> CSVRowSource:
        if (
            len(row.columns) != len(row.values)
            or len(set(row.columns)) != len(row.columns)
            or row.primary_key not in row.columns
            or row.values[row.columns.index(row.primary_key)] != row.primary_key_value
        ):
            raise ValueError("CSV row fields are inconsistent")
        identity = csv_row_source_id(row.document_id, row.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=row.document_id,
            primary_key=row.primary_key,
            primary_key_value=row.primary_key_value,
            columns=row.columns,
            values=row.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]

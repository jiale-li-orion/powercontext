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

"""Capture lossless, branch-aware Pi session events through durable connectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _read_archive(content: bytes) -> tuple[str, dict[str, dict[str, JsonValue]], str | None]:
    try:
        lines = [json.loads(line) for line in content.decode("utf-8-sig", errors="strict").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Pi session must contain valid UTF-8 JSONL") from error
    if not lines or not isinstance(lines[0], dict):
        raise ValueError("Pi session must begin with a header")
    header = lines[0]
    session_id = header.get("id")
    if (
        header.get("type") != "session"
        or type(header.get("version")) is not int
        or header.get("version") != 3
        or not isinstance(session_id, str)
        or not session_id.strip()
    ):
        raise ValueError("invalid Pi v3 session header")
    nodes: dict[str, dict[str, JsonValue]] = {}
    last: str | None = None
    for record in lines[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi event must be a JSON object")
        event_id = record.get("id")
        parent_id = record.get("parentId")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in nodes:
            raise ValueError("invalid or duplicate Pi event ID")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in nodes):
            raise ValueError("Pi parent must refer to an earlier event")
        nodes[event_id] = record
        last = event_id
    return session_id, nodes, last


def _text_blocks(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    chunks: list[str] = []
    for block in value:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                chunks.append(block["text"])
            elif block.get("type") == "toolCall":
                name = block.get("name")
                call_id = block.get("id")
                args = json.dumps(block.get("arguments", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                chunks.append(f"[tool_call {name} {call_id}] {args}")
    return "\n".join(chunks)


def _visible_event(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "bashExecution":
            command = message.get("command", "")
            output = message.get("output", "")
            body = f"$ {command}\n{output}" if output else f"$ {command}"
        else:
            body = _text_blocks(message.get("content"))
            if role == "toolResult" and body:
                body = f"[tool_result {message.get('toolName')} {message.get('toolCallId')}] {body}"
    elif event_type in ("compaction", "branch_summary"):
        role = event_type
        body = record.get("summary", "")
    elif event_type == "custom_message":
        role = "custom"
        body = _text_blocks(record.get("content"))
    else:
        return None
    if not isinstance(body, str) or not body:
        return None
    return role, f"{role}: {body}"


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Materialize the selected leaf's complete visible ancestral path."""
    session_id, nodes, default_leaf = _read_archive(value.content)
    leaf = default_leaf if value.leaf_id is None else value.leaf_id
    if leaf not in nodes:
        raise ValueError("selected Pi leaf does not exist")
    path: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    while leaf is not None:
        if leaf in seen:
            raise ValueError("Pi session contains a cycle")
        seen.add(leaf)
        record = nodes[leaf]
        path.append(record)
        leaf = record.get("parentId")
    events: list[PiEventInput] = []
    for record in reversed(path):
        visible = _visible_event(record)
        if visible is None:
            continue
        role, content = visible
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=record["id"],
                parent_id=record.get("parentId"),
                event_type=record["type"],
                role=role,
                content=content,
                record=record,
            )
        )
    return tuple(events)


def _event_identity(session_id: str, event_id: str) -> str:
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("Pi event identity exceeds Source ID limit")
    return identity


class PiEventAdapter:
    name = "pi-event"
    input_class = PiEventInput
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = _event_identity(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            **value.model_dump(),
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput.model_validate(source.model_dump(exclude={"name", "source_id", "materialization", "definition_version", "description"}))


class PiEventEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventAdapter(), projections=(PiEventEvidenceProjection(),)
)


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self.path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self.leaf_id))
        session_id, nodes, _ = _read_archive(raw)
        previous = session.checkpoint
        if previous is not None and (
            not isinstance(previous, dict) or previous.get("session_id") != session_id
            or not isinstance(previous.get("records"), dict)
        ):
            raise ValueError("invalid previous Pi session checkpoint")
        imported: dict[str, str] = dict(previous["records"]) if previous is not None else {}
        current_hashes = {
            event_id: hashlib.sha256(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for event_id, record in nodes.items()
        }
        if any(event_id not in current_hashes or current_hashes[event_id] != digest for event_id, digest in imported.items()):
            raise ValueError("previously imported Pi event was modified or removed")
        for event in events:
            if event.event_id in imported:
                continue
            await session.submit(event.event_id, "pi-event", event)
            imported[event.event_id] = current_hashes[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"session_id": session_id, "records": imported},
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]

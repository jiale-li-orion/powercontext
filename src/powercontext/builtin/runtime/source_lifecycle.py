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

"""Per-Scope, persisted evidence-dependency lifecycle over native Memory states."""

# SourceLifecycleError carries actionable, stable validation explanations.
# ruff: noqa: TRY003

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """Invalid registered evidence dependency or lifecycle transition."""


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: StrictInt = Field(ge=1)
    sequence: StrictInt = Field(ge=1)
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain nonblank text")
        return value


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: dict[str, object]
    parents: tuple[str, ...]
    policy: Literal["all", "any"] | None
    source_revision: int | None
    sequence: int = 0
    last_event: dict[str, object] | None = None
    available: bool = True

    @classmethod
    def from_row(cls, row) -> _Node:
        return cls(
            node_id=str(row["node_id"]),
            kind=row["kind"],
            reference=json.loads(row["reference"]),
            parents=tuple(json.loads(row["parents"])),
            policy=row["policy"],
            source_revision=row["source_revision"],
            sequence=int(row["event_sequence"]),
            last_event=None if row["last_event"] is None else json.loads(row["last_event"]),
            available=bool(row["available"]),
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_node_id(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > 255:
        raise SourceLifecycleError("node_id must be a nonblank bounded string")


def _desired_states(nodes: dict[str, _Node], root_override: tuple[str, bool] | None = None) -> dict[str, bool]:
    calculated: dict[str, bool] = {}
    visiting: set[str] = set()

    def available(node_id: str) -> bool:
        if node_id in calculated:
            return calculated[node_id]
        if node_id in visiting:
            raise SourceLifecycleError("evidence graph contains a cycle")
        visiting.add(node_id)
        node = nodes[node_id]
        if node.kind == "source":
            result = root_override[1] if root_override is not None and root_override[0] == node_id else node.available
        else:
            parents = (available(parent) for parent in node.parents)
            result = all(parents) if node.policy == "all" else any(parents)
        visiting.remove(node_id)
        calculated[node_id] = result
        return result

    for node_id in nodes:
        available(node_id)
    return calculated


class ScopedSourceLifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id

    async def _nodes(self) -> dict[str, _Node]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database.transaction() as connection:
            rows = (await connection.execute(select(table).where(table.c.scope_id == self.scope_id))).mappings().all()
        return {str(row["node_id"]): _Node.from_row(row) for row in rows}

    async def _register(self, node: _Node) -> None:
        _validate_node_id(node.node_id)
        nodes = await self._nodes()
        existing = nodes.get(node.node_id)
        if existing is not None:
            if (
                existing.kind == node.kind
                and existing.reference == node.reference
                and existing.parents == node.parents
                and existing.policy == node.policy
                and existing.source_revision == node.source_revision
            ):
                return
            raise SourceLifecycleError("node_id is already bound to a different record")
        if node.kind != "source" and (
            not node.parents
            or len(node.parents) != len(set(node.parents))
            or node.node_id in node.parents
            or any(parent not in nodes for parent in node.parents)
        ):
            raise SourceLifecycleError("invalid evidence dependency parents")
        nodes[node.node_id] = node
        desired = _desired_states(nodes)
        if node.kind == "memory" and not desired[node.node_id]:
            await self._sync_memories(nodes, desired)
        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database.transaction() as connection:
            await connection.execute(
                insert(table).values(
                    scope_id=self.scope_id,
                    node_id=node.node_id,
                    kind=node.kind,
                    reference=_canonical_bytes(node.reference),
                    parents=_canonical_bytes(list(node.parents)),
                    policy=node.policy,
                    source_revision=node.source_revision,
                    event_sequence=node.sequence,
                    last_event=None,
                    available=desired[node.node_id],
                )
            )

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("source revision must be a positive integer")
        async with self._runtime._context(self.scope_id) as context:
            entries = await context.sources.journal.entries()
            if not any(entry.source_ref == source_ref for entry in entries):
                raise SourceLifecycleError("unknown registered Source")
        await self._register(_Node(node_id, "source", source_ref.model_dump(mode="json"), (), None, revision))

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: Literal["all", "any"]
    ) -> None:
        if policy not in ("all", "any"):
            raise SourceLifecycleError("unsupported dependency policy")
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except Exception as error:
            raise SourceLifecycleError("unknown Memory citation") from error
        await self._register(_Node(node_id, "memory", citation.model_dump(mode="json"), parents, policy, None))

    async def register_topic(
        self, node_id: str, reference: ArtifactRef, *, parents: tuple[str, ...], policy: Literal["all", "any"]
    ) -> None:
        if policy not in ("all", "any"):
            raise SourceLifecycleError("unsupported dependency policy")
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                "topic-memory", reference.artifact_id, reference.revision
            )
        except Exception as error:
            raise SourceLifecycleError("unknown Topic Memory revision") from error
        await self._register(_Node(node_id, "topic", reference.model_dump(mode="json"), parents, policy, None))

    async def _sync_memories(self, nodes: dict[str, _Node], target: dict[str, bool]) -> None:
        memory_nodes = [node for node in nodes.values() if node.kind == "memory"]
        if not memory_nodes:
            return
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            versions = {version.entry_id: version for version in await service.entries(current)}
            states = {entry.entry_id: entry.state for entry in current.content.manifest.entries}
            desired_by_entry: dict[str, bool] = {}
            for node in memory_nodes:
                citation = MemoryCitation.model_validate(node.reference)
                version = versions.get(citation.entry_id)
                if version is None or version.entry_version_id != citation.entry_version_id:
                    raise SourceLifecycleError("registered Memory version changed")
                desired_by_entry[citation.entry_id] = target[node.node_id]
            remove = tuple(
                versions[key] for key, enable in desired_by_entry.items() if not enable and states[key] == "active"
            )
            restore = tuple(
                versions[key] for key, enable in desired_by_entry.items() if enable and states[key] == "inactive"
            )
            if remove:
                current = await service.forget(current, entries=remove, reason="evidence source withdrawn")
            if restore:
                await service.reactivate(current, entries=restore, reason="evidence source restored")

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        nodes = await self._nodes()
        source = nodes.get(event.node_id)
        if source is None or source.kind != "source" or source.source_revision != event.revision:
            raise SourceLifecycleError("event has unknown source or mismatched revision")
        actual_event = event.model_dump(mode="json")
        if event.sequence <= source.sequence:
            if event.sequence == source.sequence and source.last_event == actual_event:
                return sorted(node_id for node_id, node in nodes.items() if node.available)
            raise SourceLifecycleError("event sequence conflicts with durable history")
        desired = _desired_states(nodes, (event.node_id, event.available))
        await self._sync_memories(nodes, desired)
        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database.transaction() as connection:
            for node_id, enabled in desired.items():
                values = {"available": enabled}
                if node_id == event.node_id:
                    values.update(event_sequence=event.sequence, last_event=_canonical_bytes(actual_event))
                await connection.execute(
                    update(table).where(table.c.scope_id == self.scope_id, table.c.node_id == node_id).values(**values)
                )
        return sorted(node_id for node_id, enabled in desired.items() if enabled)

    async def resolve(self, node_id: str, *, purpose: Literal["generation", "audit"]):
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("unknown lifecycle use")
        node = (await self._nodes()).get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown lifecycle node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("evidence node is not available for generation")
        if node.kind == "source":
            return SourceRef.model_validate(node.reference)
        if node.kind == "memory":
            return MemoryCitation.model_validate(node.reference)
        return ArtifactRef.model_validate(node.reference)


class LifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycleApplication:
        if self._database is None:
            raise SourceLifecycleError("Lifecycle requires a relational database")
        return ScopedSourceLifecycleApplication(self._runtime, self._database, scope_id)

    async def inactive_topic_artifacts(self, scope_id: str) -> frozenset[str]:
        if self._database is None:
            return frozenset()
        nodes = await self.for_scope(scope_id)._nodes()
        return frozenset(
            str(node.reference["artifact_id"]) for node in nodes.values() if node.kind == "topic" and not node.available
        )


__all__ = ["LifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent"]

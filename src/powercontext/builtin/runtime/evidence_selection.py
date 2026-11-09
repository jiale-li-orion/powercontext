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

"""Query-driven selection of literal, addressable evidence under a wire budget."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from powercontext.artifacts import ArtifactAddress
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_context import (
    MemoryEntryAddress,
    PreparedContextBuild,
    PreparedEvidenceFragment,
    PreparedMemoryCandidates,
)
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

_BEGIN = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_TOKENS = re.compile(r"[a-z]+|\d+|[\u4e00-\u9fff]+", flags=re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how if in into is it its of on or our "
    "the their there these this those to was were what when where which who why will with would "
    "can could may more most between than that some after before through compared effect effects "
    "study studies shows show found finding findings report reports using results result".split()
)


def _terms(value: str) -> tuple[str, ...]:
    return tuple(word for word in _TOKENS.findall(value.casefold()) if word not in _STOPWORDS)


@dataclass(frozen=True, slots=True)
class _Candidate:
    address: MemoryEntryAddress
    start: int
    end: int
    text: str
    rank: int
    terms: Counter[str]


def _window_spans(text: str) -> tuple[tuple[int, int], ...]:
    # Segment at natural punctuation, never alter stored characters. Add neighbor
    # context to recover evidence that depends on adjacent explanatory sentences.
    pieces = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        begin, end = match.span()
        while begin < end and text[begin].isspace():
            begin += 1
        if begin < end and text[begin:end].strip():
            pieces.append((begin, end))
    if not pieces and text.strip():
        return ((0, len(text)),)
    windows: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(pieces):
        # Most scientific evidence sentences are contiguous; keep a bounded
        # adjacent sentence when it fits the fragment budget.
        if index + 1 < len(pieces) and pieces[index + 1][1] - start <= 410:
            end = pieces[index + 1][1]
        if end - start <= 410:
            windows.append((start, end))
        else:
            for offset in range(start, end, 350):
                windows.append((offset, min(end, offset + 410)))
    return tuple(dict.fromkeys(windows))


def _render(items: list[dict[str, object]]) -> str:
    body = json.dumps({"trust": "untrusted_history", "items": items}, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN}\n{body}\n{_END}"))


def _item(candidate: _Candidate, end: int | None = None) -> dict[str, object]:
    address = candidate.address
    final_end = end if end is not None else candidate.end
    return {
        "citation": {
            "scope_id": address.memory.scope_id,
            "memory_ref": address.memory.artifact.model_dump(mode="json"),
            "entry_id": address.entry_id,
            "entry_version_id": address.entry_version_id,
        },
        "content": candidate.text[: final_end - candidate.start],
        "start": candidate.start,
        "end": final_end,
    }


def build_evidence_selection(
    request: PrepareContextRequest,
    groups: tuple[PreparedMemoryCandidates, ...] | list[PreparedMemoryCandidates],
    current_scope_id: str | None,
) -> PreparedContextBuild:
    """Select high-support original spans, then fit complete citations + text."""
    query = set(_terms(request.query))
    candidates: list[_Candidate] = []
    for group in groups:
        for rank, hit in enumerate(group.hits):
            if group.memory_ref != hit.memory_ref:
                raise ValueError("candidate memory revision mismatch")
            address = MemoryEntryAddress(
                memory=ArtifactAddress(scope_id=group.scope_id or current_scope_id or "", artifact=hit.memory_ref),
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            for start, end in _window_spans(hit.text):
                segment = hit.text[start:end]
                candidates.append(_Candidate(address, start, end, segment, rank, Counter(_terms(segment))))

    if not candidates:
        return PreparedContextBuild(context=PreparedContext(status="empty", content=None, content_bytes=0), origins=())
    frequency = Counter(term for candidate in candidates for term in candidate.terms.keys())
    used_ranges: dict[tuple[str, str, int, str, str], list[tuple[int, int]]] = {}
    chosen: list[dict[str, object]] = []
    fragments: list[PreparedEvidenceFragment] = []

    def priority(candidate: _Candidate) -> float:
        matches = query.intersection(candidate.terms)
        lexical = sum(1.0 + math.log((len(candidates) + 1) / (frequency[word] + 1)) for word in matches)
        # A phrase match is stronger than a document-wide bag of words.
        terms = tuple(_terms(candidate.text))
        query_phrase = tuple(_terms(request.query))
        bigrams = set(zip(terms, terms[1:]))
        query_pairs = set(zip(query_phrase, query_phrase[1:]))
        proximity = 1.6 * len(bigrams & query_pairs)
        return (lexical + proximity + 0.08 / (candidate.rank + 1)) / (max(70, len(candidate.text)) ** 0.23)

    # Greedy selection accounts for the exact JSON and citation cost at each step.
    for candidate in sorted(candidates, key=lambda value: (-priority(value), value.rank, value.start)):
        if len(chosen) >= 5:
            break
        reference = candidate.address
        key = (reference.memory.scope_id, reference.memory.artifact.artifact_id,
               reference.memory.artifact.revision, reference.entry_id, reference.entry_version_id)
        if any(candidate.start < end and start < candidate.end for start, end in used_ranges.get(key, [])):
            continue
        item = _item(candidate)
        if len(_render([*chosen, item]).encode("utf-8")) > request.max_bytes:
            lo, hi, fitted = 1, candidate.end - candidate.start, None
            while lo <= hi:
                length = (lo + hi) // 2
                trial = _item(candidate, candidate.start + length)
                if len(_render([*chosen, trial]).encode("utf-8")) <= request.max_bytes:
                    fitted = trial
                    lo = length + 1
                else:
                    hi = length - 1
            if fitted is None or len(fitted["content"]) < 35:
                continue
            item = fitted
        chosen.append(item)
        end = item["end"]
        used_ranges.setdefault(key, []).append((candidate.start, end))
        fragments.append(
            PreparedEvidenceFragment(candidate.address, candidate.start, end, item["content"])
        )

    if not chosen:
        return PreparedContextBuild(context=PreparedContext(status="empty", content=None, content_bytes=0), origins=())
    result = _render(chosen)
    return PreparedContextBuild(
        context=PreparedContext(status="ready", content=result, content_bytes=len(result.encode("utf-8"))),
        origins=tuple(fragment.address for fragment in fragments),
        fragments=tuple(fragments),
    )

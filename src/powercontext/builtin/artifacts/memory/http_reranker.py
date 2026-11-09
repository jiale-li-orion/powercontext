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

"""HTTP scoring adapter with global rank identity and stable cross-batch ordering."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be nonempty")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be nonempty")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not candidates or type(limit) is not int or limit < 1 or limit > len(candidates):
            raise ValueError("limit must be between one and candidate count")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be nonempty")

        global_scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            expected_ranks = set(range(start + 1, start + len(chunk) + 1))
            response = await self.client.post(
                self.endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text}
                        for rank, candidate in enumerate(chunk, start=start + 1)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as error:
                raise ValueError("reranker returned invalid JSON") from error
            if not isinstance(payload, dict) or set(payload) != {"scores"}:
                raise ValueError("reranker response must contain only scores")
            records = payload["scores"]
            if not isinstance(records, list) or len(records) != len(chunk):
                raise ValueError("reranker returned the wrong number of scores")
            observed: set[int] = set()
            for entry in records:
                if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
                    raise ValueError("invalid score record")
                rank = entry["rank"]
                score = entry["score"]
                if type(rank) is not int or rank not in expected_ranks or rank in observed:
                    raise ValueError("rank missing, duplicated, or outside its original batch")
                if type(score) not in (float, int) or not math.isfinite(score):
                    raise ValueError("score must be a finite numeric value")
                observed.add(rank)
                global_scores[rank] = float(score)
            if observed != expected_ranks:
                raise ValueError("reranker omitted a candidate rank")

        selection = tuple(sorted(global_scores, key=lambda rank: (-global_scores[rank], rank))[:limit])
        return MemoryRerankDecision(selected_ranks=selection, usage=InferenceUsage(requests=requests))


__all__ = ["HTTPMemoryReranker"]

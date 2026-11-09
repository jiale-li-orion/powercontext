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

"""Deduplicate and batch concurrent embeddings without coupling caller cancellation."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _Computed:
    vector: EmbeddingVector
    batch_number: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if type(coalesce_ms) not in (int, float) or not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be finite and nonnegative")
        self.model = model
        self.profile = model.profile
        self.max_batch_size = max_batch_size
        self.coalesce_ms = coalesce_ms
        self.cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, asyncio.Future[_Computed]] = {}
        self._queue: list[str] = []
        self._dispatch_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._batch_number = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        local: dict[str, EmbeddingVector | asyncio.Future[_Computed]] = {}
        for text in dict.fromkeys(texts):
            if text in self._cache:
                local[text] = self._cache[text]
                self._cache.move_to_end(text)
                continue
            future = self._pending.get(text)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._pending[text] = future
                self._queue.append(text)
            local[text] = future

        if self._queue and (self._dispatch_task is None or self._dispatch_task.done()):
            self._dispatch_task = asyncio.create_task(self._dispatch())

        resolved: dict[str, EmbeddingVector] = {}
        per_batch: dict[int, InferenceUsage] = {}
        for text, value in local.items():
            if isinstance(value, asyncio.Future):
                # A cancelled caller only cancels its shield, never the provider work.
                result = await asyncio.shield(value)
                resolved[text] = result.vector
                per_batch[result.batch_number] = result.usage
            else:
                resolved[text] = value

        usages = tuple(per_batch.values())
        return EmbeddingResult(
            vectors=tuple(resolved[text] for text in texts),
            usage=InferenceUsage(
                requests=sum(usage.requests for usage in usages),
                input_tokens=(sum(usage.input_tokens or 0 for usage in usages) if any(usage.input_tokens is not None for usage in usages) else None),
                output_tokens=(sum(usage.output_tokens or 0 for usage in usages) if any(usage.output_tokens is not None for usage in usages) else None),
            ),
        )

    async def _dispatch(self) -> None:
        await asyncio.sleep(self.coalesce_ms / 1000)
        # All awaiting calls have registered their texts. Execute independent
        # batches concurrently, keeping their futures visible to late arrivals.
        while self._queue:
            batch = tuple(self._queue[: self.max_batch_size])
            del self._queue[: self.max_batch_size]
            self._batch_number += 1
            task = asyncio.create_task(self._embed_batch(batch, self._batch_number))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _embed_batch(self, texts: tuple[str, ...], batch_number: int) -> None:
        try:
            result = await self.model.embed(texts)
            if len(result.vectors) != len(texts):
                raise ValueError("embedding model returned wrong vector count")
        except BaseException as error:
            for text in texts:
                future = self._pending.pop(text)
                if not future.done():
                    future.set_exception(error)
            return
        for text, vector in zip(texts, result.vectors, strict=True):
            future = self._pending.pop(text)
            if self.cache_size:
                self._cache[text] = vector
                self._cache.move_to_end(text)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            if not future.done():
                future.set_result(_Computed(vector=vector, batch_number=batch_number, usage=result.usage))


__all__ = ["CoalescingEmbeddingModel"]

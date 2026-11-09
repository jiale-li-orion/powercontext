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

"""Canonical, independently decodable full and prefix-delta context packets."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ContextPackError(ValueError):
    """Invalid context packet or invalid relationship between delta and base."""


def _wire(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("items must be a base64 array")
    result = []
    for encoded in value:
        if not isinstance(encoded, str):
            raise ContextPackError("item must be base64 text")
        try:
            data = base64.b64decode(encoded, validate=True)
            data.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, binascii.Error, ValueError) as error:
            raise ContextPackError("invalid item encoding") from error
        if base64.b64encode(data).decode("ascii") != encoded:
            raise ContextPackError("noncanonical base64 encoding")
        result.append(data)
    return tuple(result)


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_json(payload: bytes, keys: set[str], schema: str) -> dict[str, Any]:
    try:
        def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, entry in pairs:
                if key in value:
                    raise ContextPackError("duplicate JSON field")
                value[key] = entry
            return value

        decoded = json.loads(payload.decode("utf-8", errors="strict"), object_pairs_hook=unique_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ContextPackError("invalid UTF-8 JSON packet") from error
    if not isinstance(decoded, dict) or set(decoded) != keys or decoded.get("schema") != schema:
        raise ContextPackError("unexpected packet schema or fields")
    return decoded


def _check_digest(digest: object) -> None:
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ContextPackError("invalid SHA-256 digest")


def _check_items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple):
        raise ContextPackError("items must be a bytes tuple")
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("each item must be bytes")
        try:
            item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ContextPackError("item must contain valid UTF-8") from error


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        _check_items(items)
        unsigned = {"schema": PACK_SCHEMA, "items": _encode_items(items)}
        digest = "sha256:" + hashlib.sha256(_wire(unsigned)).hexdigest()
        payload = {**unsigned, "digest": digest, "byte_count": 0}
        while True:
            count = len(_wire(payload))
            if payload["byte_count"] == count:
                return cls(items=items, digest=digest, byte_count=count)
            payload["byte_count"] = count

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if content is None or content == "":
            return cls.build(())
        if not isinstance(content, str):
            raise ContextPackError("content must be text or None")
        items: list[bytes] = []
        current = bytearray()
        # Respect original line boundaries where possible. This preserves prefixes
        # when later turns append new context without rewriting earlier paragraphs.
        for line in content.splitlines(keepends=True):
            for char in line:
                encoded = char.encode("utf-8")
                if len(encoded) > chunk_bytes:
                    raise ContextPackError("chunk limit cannot fit a UTF-8 code point")
                if len(current) + len(encoded) > chunk_bytes:
                    items.append(bytes(current))
                    current.clear()
                current.extend(encoded)
            if current:
                items.append(bytes(current))
                current.clear()
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        return _wire(
            {
                "schema": PACK_SCHEMA,
                "items": _encode_items(self.items),
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContextPack:
        data = _decode_json(payload, {"schema", "items", "digest", "byte_count"}, PACK_SCHEMA)
        _check_digest(data["digest"])
        if type(data["byte_count"]) is not int or data["byte_count"] < 0:
            raise ContextPackError("byte_count must be a nonnegative integer")
        pack = cls.build(_decode_items(data["items"]))
        if data["digest"] != pack.digest or data["byte_count"] != pack.byte_count:
            raise ContextPackError("digest or byte_count mismatch")
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        shared = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=target.items[shared:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _wire(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContextPackDelta:
        data = _decode_json(
            payload,
            {"schema", "base_digest", "target_digest", "prefix_items", "appended"},
            DELTA_SCHEMA,
        )
        _check_digest(data["base_digest"])
        _check_digest(data["target_digest"])
        if type(data["prefix_items"]) is not int or data["prefix_items"] < 0:
            raise ContextPackError("prefix_items must be a nonnegative integer")
        return cls(
            base_digest=data["base_digest"],
            target_digest=data["target_digest"],
            prefix_items=data["prefix_items"],
            appended=_decode_items(data["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if ContextPack.build(base.items) != base:
        raise ContextPackError("base pack is inconsistent")
    if (
        base.digest != delta.base_digest
        or type(delta.prefix_items) is not int
        or delta.prefix_items < 0
        or delta.prefix_items > len(base.items)
    ):
        raise ContextPackError("delta does not apply to this base pack")
    rebuilt = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("target digest mismatch")
    return rebuilt


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]

"""Merkle-tree daily anchoring (DECISIONS.md D6).

Rather than anchor every clip individually (many OpenTimestamps submissions,
slow, hammers the public calendar servers), the day's clip hashes are combined
into one Merkle tree; only the *root* gets anchored. Anyone can still prove any
single clip was part of that day's root via a short inclusion proof — the
same trick TLS Certificate Transparency and most real audit logs use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _hash_pair(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(a + b).digest()


@dataclass(frozen=True)
class InclusionProof:
    leaf_hash: str
    siblings: list[str]  # sibling hashes from leaf to root, hex
    directions: list[str]  # "L" or "R" — which side the sibling is on, per level

    def verify(self, root_hex: str) -> bool:
        current = bytes.fromhex(self.leaf_hash)
        for sibling_hex, direction in zip(self.siblings, self.directions):
            sibling = bytes.fromhex(sibling_hex)
            current = _hash_pair(sibling, current) if direction == "L" else _hash_pair(current, sibling)
        return current.hex() == root_hex


class MerkleTree:
    """Built from a list of hex-encoded leaf hashes (clip sha256 digests).
    Odd levels duplicate the last node (the standard Bitcoin/CT convention).
    """

    def __init__(self, leaf_hashes: list[str]):
        if not leaf_hashes:
            raise ValueError("Cannot build a Merkle tree with zero leaves")
        self.leaves = list(leaf_hashes)
        self._levels: list[list[bytes]] = [[bytes.fromhex(h) for h in leaf_hashes]]
        self._build()

    def _build(self) -> None:
        level = self._levels[0]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                next_level.append(_hash_pair(left, right))
            self._levels.append(next_level)
            level = next_level

    @property
    def root(self) -> str:
        return self._levels[-1][0].hex()

    def prove(self, leaf_hash: str) -> InclusionProof:
        try:
            index = self.leaves.index(leaf_hash)
        except ValueError as exc:
            raise ValueError(f"{leaf_hash} is not a leaf in this tree") from exc

        siblings: list[str] = []
        directions: list[str] = []
        for level in self._levels[:-1]:
            is_right = index % 2 == 1
            sibling_index = index - 1 if is_right else min(index + 1, len(level) - 1)
            siblings.append(level[sibling_index].hex())
            directions.append("L" if is_right else "R")
            index //= 2

        return InclusionProof(leaf_hash=leaf_hash, siblings=siblings, directions=directions)

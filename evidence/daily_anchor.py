"""Daily anchor job — ties signing.py, merkle.py, anchor.py, and custody.py
together.

Run once per day: collects the day's signed clip hashes, builds a Merkle
tree, anchors the root via OpenTimestamps, and logs it in chain-of-custody
alongside a per-clip inclusion proof.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from config import settings
from evidence.anchor import AnchorError, anchor_file
from evidence.custody import CustodyLog
from evidence.merkle import InclusionProof, MerkleTree


@dataclass(frozen=True)
class DailyAnchorResult:
    day: str
    clip_count: int
    merkle_root: str
    ots_proof_path: str | None
    inclusion_proofs: dict[str, InclusionProof]


def _manifests_for_day(clips_dir: Path, day: date) -> list[Path]:
    prefix_date = day.strftime("%Y%m%d")
    return sorted(p for p in clips_dir.glob("*.manifest.json") if f"_{prefix_date}_" in p.name)


def run_daily_anchor(day: date | None = None, custody_log: CustodyLog | None = None) -> DailyAnchorResult:
    day = day or datetime.now(timezone.utc).date()
    clips_dir = settings.clips_dir
    manifest_paths = _manifests_for_day(clips_dir, day)

    if not manifest_paths:
        raise ValueError(f"No signed clips found for {day.isoformat()} in {clips_dir}")

    leaf_hashes = []
    clip_names = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        leaf_hashes.append(manifest["sha256"])
        clip_names.append(manifest["file"])

    tree = MerkleTree(leaf_hashes)
    inclusion_proofs = {name: tree.prove(h) for name, h in zip(clip_names, leaf_hashes)}

    root_file = clips_dir / f"merkle_root_{day.strftime('%Y%m%d')}.txt"
    root_file.write_text(tree.root)

    ots_proof_path: str | None = None
    if settings.anchor_enabled:
        try:
            proof_path = anchor_file(root_file)
            ots_proof_path = str(proof_path)
        except AnchorError:
            ots_proof_path = None

    log = custody_log or CustodyLog()
    for name in clip_names:
        log.record(name, "anchored", actor="daily_anchor_job")

    proofs_file = clips_dir / f"merkle_proofs_{day.strftime('%Y%m%d')}.json"
    proofs_file.write_text(
        json.dumps({name: asdict(p) for name, p in inclusion_proofs.items()}, indent=2)
    )

    return DailyAnchorResult(
        day=day.isoformat(),
        clip_count=len(clip_names),
        merkle_root=tree.root,
        ots_proof_path=ots_proof_path,
        inclusion_proofs=inclusion_proofs,
    )

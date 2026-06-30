"""Public timestamp anchoring via OpenTimestamps (DECISIONS.md D6).

Free forever, no account, no API key — calls public Bitcoin-calendar servers.
This is the one piece of the evidence chain that needed no free substitute
(D8): it was already free, so this is the real target implementation, not a
stand-in. Requires the `ots` CLI (`pip install opentimestamps-client`).

Deliberately optional (SENTINEL_ANCHOR_ENABLED) — signing/hashing (signing.py)
already gives local tamper-evidence; anchoring adds an independent, external
proof of *when*, which matters for court/insurance use but isn't needed to
develop the rest of the pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from config import settings


class AnchorError(RuntimeError):
    pass


def anchor_file(path: Path) -> Path:
    """Stamp `path` with OpenTimestamps, producing a `<path>.ots` proof file.

    Confirmation isn't immediate — Bitcoin anchoring takes time to confirm.
    Use `ots upgrade <file>.ots` later to attach the completed Bitcoin proof.
    """
    if not settings.anchor_enabled:
        raise AnchorError("Anchoring is disabled (SENTINEL_ANCHOR_ENABLED=false).")

    if shutil.which("ots") is None:
        raise AnchorError(
            "The `ots` CLI is not installed. Run: pip install opentimestamps-client"
        )

    if not path.exists():
        raise AnchorError(f"Cannot anchor a file that does not exist: {path}")

    result = subprocess.run(["ots", "stamp", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise AnchorError(f"ots stamp failed for {path}: {result.stderr.strip()}")

    proof_path = path.with_suffix(path.suffix + ".ots")
    if not proof_path.exists():
        raise AnchorError(f"ots stamp reported success but no proof file was written: {proof_path}")
    return proof_path

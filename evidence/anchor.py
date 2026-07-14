"""Public timestamp anchoring via OpenTimestamps.

Calls public Bitcoin-calendar servers — free, no account, no API key.
Requires the `ots` CLI (`pip install opentimestamps-client`). Optional,
gated behind SENTINEL_ANCHOR_ENABLED.
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

    Confirmation isn't immediate — use `ots upgrade <file>.ots` later to
    attach the completed Bitcoin proof.
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

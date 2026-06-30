"""IPFS upload — fixes the two bugs in the original MVP's ipfs_convertion.py:

1. `cid` was only assigned inside the `try` block, so a failed `ipfs add` raised
   UnboundLocalError on `return cid` instead of a clear failure.
2. The function built an unused HTTP POST (`files`, `url` to the local IPFS API)
   that was never sent — the CLI subprocess call was used instead. Dead code,
   removed; this version commits to the HTTP API call, which doesn't require the
   `ipfs` binary to be on PATH (only the daemon, which IPFS Desktop already runs).

Free by construction (DECISIONS.md D8) — a local `ipfs daemon` costs nothing.
Gated behind SENTINEL_IPFS_ENABLED so the pipeline runs without it installed.
"""

from __future__ import annotations

from pathlib import Path

import requests

from config import settings


class IPFSError(RuntimeError):
    pass


def add_to_ipfs(filepath: Path) -> str:
    """Upload a file to a local IPFS node via its HTTP API. Returns the CID.

    Raises IPFSError on any failure — never returns an undefined value.
    """
    if not settings.ipfs_enabled:
        raise IPFSError("IPFS is disabled (SENTINEL_IPFS_ENABLED=false) — nothing was uploaded.")

    url = f"{settings.ipfs_api_url.rstrip('/')}/api/v0/add"
    try:
        with filepath.open("rb") as fp:
            response = requests.post(url, files={"file": (filepath.name, fp)}, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise IPFSError(f"IPFS add failed for {filepath}: {exc}") from exc

    payload = response.json()
    cid = payload.get("Hash")
    if not cid:
        raise IPFSError(f"IPFS add returned no CID for {filepath}: {payload!r}")
    return cid


def public_gateway_url(cid: str) -> str:
    return f"https://ipfs.io/ipfs/{cid}"

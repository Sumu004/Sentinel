from __future__ import annotations

from pathlib import Path

import requests

from config import settings


class IPFSError(RuntimeError):
    pass


def add_to_ipfs(filepath: Path) -> str:
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

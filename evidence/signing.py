from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from config import settings


@dataclass(frozen=True)
class Manifest:
    file: str
    sha256: str
    signed_at: str
    site_id: str
    public_key_hex: str
    signature_hex: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _load_or_create_key() -> Ed25519PrivateKey:
    key_path = settings.evidence_key_path
    if key_path.exists():
        return serialization.load_pem_private_key(key_path.read_bytes(), password=None)

    key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(pem)
    key_path.chmod(0o600)
    return private_key


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_clip(clip_path: Path) -> Manifest:
    if not clip_path.exists():
        raise FileNotFoundError(f"Cannot sign a clip that does not exist: {clip_path}")

    private_key = _load_or_create_key()
    public_key: Ed25519PublicKey = private_key.public_key()

    digest_hex = sha256_file(clip_path)
    signature = private_key.sign(bytes.fromhex(digest_hex))

    manifest = Manifest(
        file=clip_path.name,
        sha256=digest_hex,
        signed_at=datetime.now(timezone.utc).isoformat(),
        site_id=settings.site_id,
        public_key_hex=public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        ).hex(),
        signature_hex=signature.hex(),
    )

    manifest_path = clip_path.with_suffix(clip_path.suffix + ".manifest.json")
    manifest_path.write_text(manifest.to_json())
    return manifest


def verify_clip(clip_path: Path) -> bool:
    manifest_path = clip_path.with_suffix(clip_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for {clip_path}")

    data = json.loads(manifest_path.read_text())
    current_hash = sha256_file(clip_path)
    if current_hash != data["sha256"]:
        return False

    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(data["public_key_hex"]))
    try:
        public_key.verify(bytes.fromhex(data["signature_hex"]), bytes.fromhex(data["sha256"]))
    except Exception:
        return False
    return True

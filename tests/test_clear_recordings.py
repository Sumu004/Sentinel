"""POST /recordings/clear deletes real clip files from settings.clips_dir and
clears the pipeline log buffer — but must never touch the events DB or
evidence/custody.py's chain-of-custody log (that's the tamper-evident
integrity record, not disposable UI state). Uses a real temp directory and
real files, not mocks, since this is exactly the kind of destructive-action
endpoint that needs to prove it deletes precisely what it claims to.
"""

from fastapi.testclient import TestClient

from cloud.backend.app import app
from config import settings


def test_clear_recordings_deletes_clip_files(tmp_path):
    original_clips_dir = settings.clips_dir
    object.__setattr__(settings, "clips_dir", tmp_path)
    try:
        (tmp_path / "person_20260101_000000.mp4").write_bytes(b"fake video")
        (tmp_path / "person_20260101_000000.mp4.manifest.json").write_text("{}")
        assert len(list(tmp_path.iterdir())) == 2

        client = TestClient(app)
        resp = client.post("/recordings/clear")

        assert resp.status_code == 200
        assert resp.json()["files_deleted"] == 2
        assert list(tmp_path.iterdir()) == []
    finally:
        object.__setattr__(settings, "clips_dir", original_clips_dir)


def test_clear_recordings_refuses_while_pipeline_running(monkeypatch):
    import cloud.backend.app as app_module

    class _FakeProc:
        def poll(self):
            return None

    monkeypatch.setattr(app_module, "_pipeline_process", _FakeProc())
    client = TestClient(app)

    resp = client.post("/recordings/clear")

    assert resp.status_code == 409
    monkeypatch.setattr(app_module, "_pipeline_process", None)

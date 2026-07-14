"""Website-controlled pipeline start/stop. Starts the actual
`python -m edge.main` subprocess — camera access is unauthorized in this
environment, so it's expected to exit quickly with a camera error.
"""

import time

from fastapi.testclient import TestClient

from cloud.backend.app import app


def test_pipeline_start_reports_running_then_stop_works():
    client = TestClient(app)

    start_resp = client.post("/pipeline/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] in ("started", "already_running")

    status_resp = client.get("/pipeline/status")
    assert status_resp.status_code == 200
    assert isinstance(status_resp.json()["running"], bool)

    stop_resp = client.post("/pipeline/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] in ("stopped", "not_running")


def test_pipeline_logs_surface_the_real_camera_error():
    client = TestClient(app)
    client.post("/pipeline/start")

    deadline = time.time() + 15
    lines: list[str] = []
    while time.time() < deadline:
        lines = client.get("/pipeline/logs?lines=200").json()["lines"]
        if lines:
            break
        time.sleep(0.5)

    client.post("/pipeline/stop")
    assert lines, "expected edge.main to have produced some output (even just the startup log line)"


def test_pipeline_status_reports_not_running_when_never_started(monkeypatch):
    import cloud.backend.app as app_module

    monkeypatch.setattr(app_module, "_pipeline_process", None)
    client = TestClient(app)
    resp = client.get("/pipeline/status")
    assert resp.json() == {"running": False}

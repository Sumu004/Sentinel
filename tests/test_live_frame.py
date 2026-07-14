"""Live-tracking view: PUT /live-frame and GET /live-frame/stream (MJPEG).
Hits the real FastAPI app via TestClient.
"""

from fastapi.testclient import TestClient

from cloud.backend.app import app


def test_put_then_get_live_frame_round_trips():
    client = TestClient(app)
    fake_jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"

    put_resp = client.put("/live-frame", content=fake_jpeg, headers={"Content-Type": "application/octet-stream"})
    assert put_resp.status_code == 200

    get_resp = client.get("/live-frame")
    assert get_resp.status_code == 200
    assert get_resp.headers["content-type"] == "image/jpeg"
    assert get_resp.content == fake_jpeg


def test_put_live_frame_rejects_empty_body():
    client = TestClient(app)
    resp = client.put("/live-frame", content=b"")
    assert resp.status_code == 400

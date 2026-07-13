"""OSNetReID (perception/reid.py) — real torchreid + real ImageNet-pretrained
OSNet weights, not mocked. `torchreid` installs via pip cleanly but its own
import chain calls `import gdown` without declaring it as a dependency;
`ModuleNotFoundError: No module named 'gdown'` on first import was the real
blocker, not a packaging dead end — `pip install torchreid gdown` closes it.
These tests exercise the actual model (small, ~3MB, already cached after
first run) rather than mocking torch/torchreid.
"""

import numpy as np

from perception.reid import Gallery, OSNetReID, cosine_similarity


def _solid_color_crop(bgr: tuple[int, int, int], size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), bgr, dtype=np.uint8)


def test_osnet_embed_returns_512d_float_vector():
    embedder = OSNetReID()
    crop = _solid_color_crop((0, 0, 200))
    embedding = embedder.embed(crop)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32


def test_osnet_identical_crops_are_maximally_similar():
    embedder = OSNetReID()
    crop = _solid_color_crop((0, 0, 200))
    e1 = embedder.embed(crop)
    e2 = embedder.embed(crop)
    assert cosine_similarity(e1, e2) > 0.999


def test_osnet_distinguishes_visually_different_crops():
    embedder = OSNetReID()
    red_crop = _solid_color_crop((0, 0, 220))
    blue_crop = _solid_color_crop((220, 0, 0))
    same_red = _solid_color_crop((0, 0, 210))

    sim_same_color = cosine_similarity(embedder.embed(red_crop), embedder.embed(same_red))
    sim_diff_color = cosine_similarity(embedder.embed(red_crop), embedder.embed(blue_crop))
    assert sim_same_color > sim_diff_color


def test_osnet_embed_handles_empty_crop():
    embedder = OSNetReID()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    embedding = embedder.embed(empty)
    assert embedding.shape == (512,)
    assert np.all(embedding == 0)


def test_gallery_works_with_osnet_backend():
    gallery = Gallery(embedder=OSNetReID(), threshold=0.9)
    crop_a = _solid_color_crop((0, 0, 200))
    crop_b = _solid_color_crop((200, 0, 0))

    id1, is_new1 = gallery.match_or_register(crop_a)
    id2, is_new2 = gallery.match_or_register(crop_b)
    id3, is_new3 = gallery.match_or_register(crop_a)

    assert is_new1 is True
    assert is_new2 is True
    assert is_new3 is False
    assert id3 == id1
    assert id2 != id1

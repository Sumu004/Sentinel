"""L1 perception fleet beyond object detection (VISION.md L1 table).

`edge/detector.py` only ever did object detection (person/vehicle/animal/
package). The other L1 rows DECISIONS.md picks models for — pose/action,
re-ID, anomaly, audio, fire/smoke — had no code at all, not even a stub
interface. Each submodule here ships a real, free, testable default (no paid
API, no GPU, no dataset download) plus a documented upgrade path to the
DECISIONS.md-chosen model where a free-tier default isn't credible (audio,
fire/smoke — those stay heuristics, not classifiers, until real event
recordings exist to train on).
"""

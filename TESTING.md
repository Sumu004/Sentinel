# Sentinel — Testing & scoring

How to test on real, unseen footage — not the training set.

## Why unseen footage

A model that scores well on its own validation split tells you little about
your camera at 2am in the rain. Test on a different camera, real lighting
changes, real weather, and staged events you control so you know the ground truth.

## Three separate scores

| Score | Layer | Question |
|---|---|---|
| Detection rate | L1 | Did it find and classify what was actually there? |
| Description rate | L2 | Did it describe the event correctly, without making things up? |
| System score | end-to-end | Did the right alert, with evidence, reach the operator in time? |

Kept separate because they fail independently — a great detector can still
have a hallucinating description layer.

## Running it

```python
from eval.scoring import score_detections, false_alarms_per_day
from eval.scoring import score_description, DescriptionGT
from eval.scoring import system_score, EventOutcome

score = score_detections(preds, gts, iou_threshold=0.5)
false_alarms_per_day(spurious_events=3, hours_observed=8)
```

```bash
python -m training.eval false-alarms --footage ./event_free_night.mp4 --hours 8
```

## Targets (remote-home pilot)

| Metric | Acceptable | Good |
|---|---|---|
| person recall | ≥ 0.85 | ≥ 0.95 |
| person precision | ≥ 0.80 | ≥ 0.92 |
| false alarms / camera / day | ≤ 10 | ≤ 2 |
| description subject accuracy | ≥ 0.90 | ≥ 0.97 |
| hallucination rate | ≤ 0.05 | ≤ 0.01 |

Latency budget: under ~2s from detection to alert on the target edge box.

## Status

Scoring code and false-alarm measurement are done and unit-tested. What's
missing is a real labelled unseen-footage set from an actual pilot camera —
that needs a real deployment, not more code.

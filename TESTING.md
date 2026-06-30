# Sentinel — Testing protocol & scoring

How to test the system on **unseen, real-world, live footage** and score it.
The scoring code lives in [eval/scoring.py](eval/scoring.py) (unit-tested in
`tests/test_scoring.py`); this document is the protocol that produces the inputs
to it.

## Core principle: test on footage the model has never seen

A model that scores 0.95 mAP on its own validation split tells you almost
nothing about how it behaves on *your* camera at 2am in the rain. Every number
that matters here comes from footage that was **not** in any training or
validation set:

- A different camera, or the same camera re-recording on different days.
- Real lighting transitions (dawn, dusk, headlights, IR night mode).
- Real weather (rain, wind-moved foliage — the classic false-alarm source).
- Real, un-staged activity plus a set of **staged test events** you control so
  you know the ground truth.

## Three things we measure (kept separate on purpose)

| Score | Layer | Question it answers |
|---|---|---|
| **Detection rate** | L1 perception | Did it find and correctly classify what was actually there? |
| **Description rate** | L2 reasoning | Given a real event, did it describe it correctly without making things up? |
| **System score** | end-to-end | Did the *right alert*, correct severity, sealed evidence, actually reach the operator in time? |

They're separate because they fail independently: a detector can be excellent
while descriptions hallucinate, or both can be perfect while severity routing
sends a pet through as a high-priority alarm. One blended number would hide
exactly the regression you need to see.

---

## Protocol A — Detection rate

**Setup.** Collect 2 buckets of unseen footage from the test camera:

1. **Event footage** — staged events you script and label: a person walking to
   the door, a car pulling in, a package dropped then picked up, a pet crossing
   the yard. Record what *should* be detected, frame-by-frame or as labelled
   clips (any tool: CVAT, Roboflow, Label Studio — all free tiers).
2. **Event-free footage** — several continuous hours where *nothing* of
   interest happens (empty yard, swaying trees, passing clouds, day→night
   transition). This is for the false-alarm rate.

**Run.**

```python
from eval.scoring import score_detections, false_alarms_per_day, PredBox, GTBox

# preds = model output on a labelled test frame; gts = your labels
score = score_detections(preds, gts, iou_threshold=0.5)
print(score.precision, score.recall, score.f1, score.per_class)

# from the event-free footage:
print("false alarms/day:", false_alarms_per_day(spurious_events=3, hours_observed=8))
```

For the false-alarm pass you can drive the live pipeline directly:

```bash
python -m training.eval false-alarms --footage ./event_free_night.mp4 --hours 8
```

**Targets (remote-home pilot, starting bar — tighten over time):**

| Metric | Acceptable | Good |
|---|---|---|
| `person` recall | ≥ 0.85 | ≥ 0.95 |
| `person` precision | ≥ 0.80 | ≥ 0.92 |
| `package` recall | ≥ 0.70 | ≥ 0.85 |
| False alarms / camera / day | ≤ 10 | ≤ 2 |

The false-alarm target matters most: an unattended-site product dies on alert
fatigue, not on a missed mAP point.

---

## Protocol B — Description rate

For each **true** event (from Protocol A's event footage), a human labels the
ground-truth attributes once:

```python
from eval.scoring import DescriptionGT, score_description, description_rate

gt = DescriptionGT(subject="person", action="taking", severity="high")  # porch theft
result = score_description(
    predicted_text=vlm_output,              # whatever the reasoning layer wrote
    gt=gt,
    hallucination_terms=["weapon", "fire"], # things that would be a dangerous false claim here
)
agg = description_rate([result, ...])
print(agg)  # mean_score, subject_acc, action_acc, severity_acc, hallucination_rate
```

**Why hallucination zeroes the score:** a confident wrong description ("person
with a weapon" when it's a delivery) is operationally worse than a vague correct
one — it triggers the wrong response. The scorer treats any hallucinated term as
a failed event regardless of what else it got right.

**Targets:**

| Metric | Acceptable | Good |
|---|---|---|
| subject accuracy | ≥ 0.90 | ≥ 0.97 |
| action accuracy | ≥ 0.70 | ≥ 0.85 |
| severity accuracy | ≥ 0.80 | ≥ 0.90 |
| hallucination rate | ≤ 0.05 | ≤ 0.01 |

Note: the keyword-based scorer in `score_description` is a transparent,
reproducible harness. For nuanced grading, swap in an LLM-judge that returns the
same three booleans — the rest of the pipeline is unchanged.

---

## Protocol C — System score (end-to-end)

Run the **whole live pipeline** (`python -m edge.main` + the backend) against
the staged events, and for each one record how far it got:

```python
from eval.scoring import EventOutcome, system_score

outcomes = [
    EventOutcome(
        detected=True,                    # detector fired on the real event
        described_correctly=True,         # DescriptionResult.score >= your threshold
        severity_routed_correctly=True,   # escalated/suppressed correctly
        evidence_sealed=True,             # verifiable clip produced (evidence/signing.py verify passes)
        latency_ok=True,                  # alert arrived within budget
    ),
    # ... one per staged event
]
print(system_score(outcomes))
```

The composite weights detection highest (0.35) because nothing downstream
happens without it, then description and severity (0.20 each), evidence (0.15),
latency (0.10). `fully_successful_rate` is the strict view: the fraction of
events that cleared **every** stage. Watch both — a high composite with a low
fully-successful rate means each stage is individually okay but they rarely all
succeed on the same event.

**Latency budget (per VISION.md "fast and safer"):** detection-to-alert under
~2s on the target edge box. Measure it on the actual hardware (D5), not a dev
laptop.

---

## Running cadence

- **Per model candidate** (RF-DETR vs YOLO12 bake-off): Protocol A on the same
  held-out test set → picks the detector.
- **Before any pilot deployment:** all three protocols on a fresh day of unseen
  footage from the actual install.
- **Continuously after deployment:** the false-alarm rate and
  `fully_successful_rate` are the two numbers to dashboard — they're the
  earliest warning that a camera, lighting change, or model drift has degraded
  the system.

## What exists vs. what's pending

- [x] Scoring code — `eval/scoring.py`, unit-tested.
- [x] False-alarm measurement against live footage — `training/eval.py`.
- [x] A working model-backed detector to test — `edge/detector.py ModelDetector`
  (verified running real YOLO inference on real images).
- [ ] The actual labelled unseen-footage test set — needs real recordings from
  the pilot camera. This is the human-in-the-loop part no code can substitute.
- [ ] Description scoring against a live VLM — needs the L2 reasoning layer
  (Phase 2.4) wired up first.

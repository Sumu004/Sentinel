"""Scoring system for Sentinel (see TESTING.md for the protocol).

Three separate scores:

1. Detection rate (L1) — precision/recall/F1 per class via IoU matching,
   plus false alarms per camera per day.
2. Description rate (L2) — subject/action/severity accuracy and
   hallucination rate.
3. System score (end-to-end) — right alert, right severity, sealed
   evidence, in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class GTBox:
    label: str
    box: Box


@dataclass(frozen=True)
class PredBox:
    label: str
    confidence: float
    box: Box


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionScore:
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


def score_detections(
    preds: list[PredBox], gts: list[GTBox], iou_threshold: float = 0.5
) -> DetectionScore:
    """Greedy IoU matching (highest-confidence prediction first). Each GT box
    is matched at most once; unmatched predictions are false positives,
    unmatched GTs are false negatives.
    """
    counts: dict[str, dict[str, int]] = {}

    def bump(label: str, key: str) -> None:
        counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})[key] += 1

    used_gt: set[int] = set()
    for pred in sorted(preds, key=lambda p: p.confidence, reverse=True):
        best_idx, best_iou = -1, iou_threshold
        for i, gt in enumerate(gts):
            if i in used_gt or gt.label != pred.label:
                continue
            cur = iou(pred.box, gt.box)
            if cur >= best_iou:
                best_idx, best_iou = i, cur
        if best_idx >= 0:
            used_gt.add(best_idx)
            bump(pred.label, "tp")
        else:
            bump(pred.label, "fp")

    for i, gt in enumerate(gts):
        if i not in used_gt:
            bump(gt.label, "fn")

    score = DetectionScore()
    for label, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        score.per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        score.true_positives += tp
        score.false_positives += fp
        score.false_negatives += fn

    tp, fp, fn = score.true_positives, score.false_positives, score.false_negatives
    score.precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    score.recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    score.f1 = (
        round(2 * score.precision * score.recall / (score.precision + score.recall), 4)
        if (score.precision + score.recall)
        else 0.0
    )
    return score


def false_alarms_per_day(spurious_events: int, hours_observed: float) -> float:
    """How often the system alerts when nothing happened, computed over
    event-free footage. Lower is better.
    """
    if hours_observed <= 0:
        raise ValueError("hours_observed must be > 0")
    return round(spurious_events * (24.0 / hours_observed), 2)


@dataclass(frozen=True)
class DescriptionGT:
    """Ground truth for one event's description, labelled by a human reviewer."""

    subject: str
    action: str
    severity: str


@dataclass
class DescriptionResult:
    subject_correct: bool
    action_correct: bool
    severity_correct: bool
    hallucinated: bool
    score: float


def score_description(
    predicted_text: str, gt: DescriptionGT, hallucination_terms: list[str] | None = None
) -> DescriptionResult:
    """Scores a generated description against human-labelled attributes by
    keyword presence.

    `hallucination_terms`: words that, if present, indicate the description
    asserted something not in the event. A hallucination zeroes the score
    regardless of other matches.
    """
    text = predicted_text.lower()
    subject_ok = gt.subject.lower() in text
    action_ok = gt.action.lower() in text
    severity_ok = gt.severity.lower() in text

    hallucinated = bool(hallucination_terms) and any(t.lower() in text for t in hallucination_terms)

    if hallucinated:
        return DescriptionResult(subject_ok, action_ok, severity_ok, True, 0.0)

    score = round(sum([subject_ok, action_ok, severity_ok]) / 3, 4)
    return DescriptionResult(subject_ok, action_ok, severity_ok, False, score)


def description_rate(results: list[DescriptionResult]) -> dict[str, float]:
    """Aggregate description performance across a test set of events."""
    if not results:
        return {"mean_score": 0.0, "subject_acc": 0.0, "action_acc": 0.0, "severity_acc": 0.0, "hallucination_rate": 0.0}
    n = len(results)
    return {
        "mean_score": round(sum(r.score for r in results) / n, 4),
        "subject_acc": round(sum(r.subject_correct for r in results) / n, 4),
        "action_acc": round(sum(r.action_correct for r in results) / n, 4),
        "severity_acc": round(sum(r.severity_correct for r in results) / n, 4),
        "hallucination_rate": round(sum(r.hallucinated for r in results) / n, 4),
    }


@dataclass(frozen=True)
class EventOutcome:
    """One real event from the live test, scored at every stage of the pipeline."""

    detected: bool
    described_correctly: bool
    severity_routed_correctly: bool
    evidence_sealed: bool
    latency_ok: bool


def system_score(outcomes: list[EventOutcome], weights: dict[str, float] | None = None) -> dict[str, float]:
    """End-to-end score. An event only fully counts if it cleared every
    stage; partial credit is given per-stage. The composite is the weighted
    mean of the per-stage pass rates.
    """
    if not outcomes:
        return {"system_score": 0.0}

    w = weights or {
        "detected": 0.35,
        "described_correctly": 0.20,
        "severity_routed_correctly": 0.20,
        "evidence_sealed": 0.15,
        "latency_ok": 0.10,
    }
    n = len(outcomes)
    stage_rates = {
        "detected": sum(o.detected for o in outcomes) / n,
        "described_correctly": sum(o.described_correctly for o in outcomes) / n,
        "severity_routed_correctly": sum(o.severity_routed_correctly for o in outcomes) / n,
        "evidence_sealed": sum(o.evidence_sealed for o in outcomes) / n,
        "latency_ok": sum(o.latency_ok for o in outcomes) / n,
    }
    composite = sum(stage_rates[k] * w[k] for k in w)
    fully_ok = sum(
        all([o.detected, o.described_correctly, o.severity_routed_correctly, o.evidence_sealed, o.latency_ok])
        for o in outcomes
    ) / n

    out = {f"{k}_rate": round(v, 4) for k, v in stage_rates.items()}
    out["fully_successful_rate"] = round(fully_ok, 4)
    out["system_score"] = round(composite, 4)
    return out

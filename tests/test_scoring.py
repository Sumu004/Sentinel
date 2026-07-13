from eval.scoring import (
    DescriptionGT,
    EventOutcome,
    GTBox,
    PredBox,
    description_rate,
    false_alarms_per_day,
    iou,
    score_description,
    score_detections,
    system_score,
)


def test_iou_perfect_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_detection_perfect_match():
    preds = [PredBox("person", 0.9, (0, 0, 10, 10))]
    gts = [GTBox("person", (0, 0, 10, 10))]
    s = score_detections(preds, gts)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0
    assert s.true_positives == 1


def test_detection_false_positive_and_negative():
    preds = [PredBox("person", 0.9, (0, 0, 10, 10)), PredBox("person", 0.8, (200, 200, 10, 10))]
    gts = [GTBox("person", (0, 0, 10, 10)), GTBox("person", (50, 50, 10, 10))]
    s = score_detections(preds, gts)
    assert s.true_positives == 1
    assert s.false_positives == 1
    assert s.false_negatives == 1


def test_detection_wrong_class_is_not_a_match():
    preds = [PredBox("animal", 0.9, (0, 0, 10, 10))]
    gts = [GTBox("person", (0, 0, 10, 10))]
    s = score_detections(preds, gts)
    assert s.true_positives == 0
    assert s.false_positives == 1
    assert s.false_negatives == 1


def test_false_alarms_per_day():
    assert false_alarms_per_day(2, 4) == 12.0


def test_description_all_correct():
    gt = DescriptionGT(subject="person", action="loitering", severity="high")
    r = score_description("A person loitering near the door, high severity", gt)
    assert r.score == 1.0 and not r.hallucinated


def test_description_hallucination_zeroes_score():
    gt = DescriptionGT(subject="person", action="approaching", severity="low")
    r = score_description(
        "A person approaching with a weapon, low severity", gt, hallucination_terms=["weapon"]
    )
    assert r.hallucinated is True
    assert r.score == 0.0


def test_description_rate_aggregate():
    gt = DescriptionGT(subject="person", action="loitering", severity="high")
    results = [
        score_description("person loitering high", gt),
        score_description("person standing", gt),
    ]
    agg = description_rate(results)
    assert agg["subject_acc"] == 1.0
    assert 0.0 < agg["mean_score"] < 1.0


def test_system_score_all_pass():
    outcomes = [EventOutcome(True, True, True, True, True)]
    s = system_score(outcomes)
    assert s["system_score"] == 1.0
    assert s["fully_successful_rate"] == 1.0


def test_system_score_partial():
    outcomes = [
        EventOutcome(True, False, True, True, True),
        EventOutcome(True, False, True, True, True),
    ]
    s = system_score(outcomes)
    assert s["detected_rate"] == 1.0
    assert s["described_correctly_rate"] == 0.0
    assert s["fully_successful_rate"] == 0.0
    assert 0.0 < s["system_score"] < 1.0

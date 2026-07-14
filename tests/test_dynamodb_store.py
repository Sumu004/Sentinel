from unittest.mock import MagicMock, patch

from cloud.backend.db import DynamoDBStore, EventRecord


def _make_store() -> tuple[DynamoDBStore, MagicMock]:
    mock_table = MagicMock()
    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    with patch("boto3.resource", return_value=mock_resource) as mock_boto_resource:
        store = DynamoDBStore(table_name="sentinel-events", region="us-east-1")
    mock_boto_resource.assert_called_once_with("dynamodb", region_name="us-east-1")
    mock_resource.Table.assert_called_once_with("sentinel-events")
    return store, mock_table


def test_dynamodb_store_save_puts_expected_item():
    store, mock_table = _make_store()
    record = EventRecord(
        event_id="e1",
        site_id="dev-site-01",
        label="motion",
        track_id=1,
        started_at="2026-01-01T00:00:00+00:00",
        detected_at="2026-01-01T00:00:03+00:00",
        assigned=True,
    )

    store.save(record)

    mock_table.put_item.assert_called_once_with(
        Item={
            "event_id": "e1",
            "site_id": "dev-site-01",
            "label": "motion",
            "track_id": 1,
            "started_at": "2026-01-01T00:00:00+00:00",
            "detected_at": "2026-01-01T00:00:03+00:00",
            "assigned": True,
        }
    )


def test_dynamodb_store_list_recent_sorts_by_detected_at_desc():
    store, mock_table = _make_store()
    mock_table.scan.return_value = {
        "Items": [
            {
                "event_id": "e1",
                "site_id": "s1",
                "label": "motion",
                "track_id": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "detected_at": "2026-01-01T00:00:01+00:00",
                "assigned": False,
            },
            {
                "event_id": "e2",
                "site_id": "s1",
                "label": "motion",
                "track_id": 2,
                "started_at": "2026-01-01T00:00:00+00:00",
                "detected_at": "2026-01-01T00:00:09+00:00",
                "assigned": False,
            },
        ]
    }

    recent = store.list_recent(limit=10)

    mock_table.scan.assert_called_once_with(Limit=10)
    assert [r.event_id for r in recent] == ["e2", "e1"]


def test_dynamodb_store_assign_returns_true_on_success():
    store, mock_table = _make_store()

    assert store.assign("e1") is True
    mock_table.update_item.assert_called_once_with(
        Key={"event_id": "e1"},
        UpdateExpression="SET assigned = :v",
        ExpressionAttributeValues={":v": True},
    )


def test_dynamodb_store_assign_returns_false_on_failure():
    store, mock_table = _make_store()
    mock_table.update_item.side_effect = RuntimeError("boto3 ClientError")

    assert store.assign("missing") is False

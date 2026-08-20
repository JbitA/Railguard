import json
from edge.railguard_edge.outbox import DurableOutbox


def test_outbox_is_bounded_and_records_drop_count(tmp_path):
    outbox=DurableOutbox(tmp_path/'spool.sqlite',max_records=3)
    try:
        for i in range(5):
            outbox.enqueue(json.dumps({'device_id':'railguard-test','ts':f'2026-08-20T12:00:0{i}Z','seq':i}))
        assert outbox.depth()==3
        assert outbox.dropped_records()==2
        rows=outbox.batch(10)
        assert [json.loads(r[1])['seq'] for r in rows]==[2,3,4]
    finally:
        outbox.close()


def test_duplicate_ack_identity_is_immutable_except_transport_queue_counters(tmp_path):
    outbox = DurableOutbox(tmp_path / "identity.sqlite", max_records=10)
    try:
        base = {
            "device_id": "railguard-test",
            "ts": "2026-08-20T12:00:00Z",
            "seq": 7,
            "health": {"spool_depth": 0, "spool_dropped": 0, "packet_loss": 0},
            "vibration": {"rms_ms2": 1.2},
        }
        outbox.enqueue(json.dumps(base))
        replay = json.loads(json.dumps(base))
        replay["health"]["spool_depth"] = 99
        replay["health"]["spool_dropped"] = 4
        outbox.enqueue(json.dumps(replay))
        assert outbox.depth() == 1

        conflict = json.loads(json.dumps(base))
        conflict["vibration"]["rms_ms2"] = 9.9
        import pytest
        with pytest.raises(ValueError, match="immutable ACK identity"):
            outbox.enqueue(json.dumps(conflict))
        assert outbox.depth() == 1
    finally:
        outbox.close()

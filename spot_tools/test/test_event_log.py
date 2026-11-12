"""Mission event log writer (spot_skills.event_log). No ROS, no robot."""

import json
import os

import pytest

from spot_skills import event_log as el


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "euclid_20261111-185801_a3f2"
    monkeypatch.setenv("ADT4_RUN_DIR", str(run))
    monkeypatch.setenv("ADT4_SESSION_ID", "s20261111-185000_0001")
    monkeypatch.setenv("ADT4_ROBOT_NAME", "euclid")
    monkeypatch.delenv("ADT4_RUN_ID", raising=False)
    monkeypatch.delenv("ADT4_MISSION_ID", raising=False)
    return run


def test_paths_follow_run_dir(run_env, tmp_path, monkeypatch):
    assert el.component_dir("spot_executor", "/x") == str(run_env / "spot_executor")
    assert el.events_path("spot_executor", "/x") == str(
        run_env / "events" / "spot_executor.jsonl"
    )
    monkeypatch.delenv("ADT4_RUN_DIR")
    assert el.component_dir("spot_executor", "/x") == "/x"
    assert el.events_path("spot_executor", "/x") == "/x/events/spot_executor.jsonl"


def test_envelope_and_seq(run_env):
    path = el.events_path("spot_executor", "/unused")
    log = el.EventLog(path, "spot_executor", machine="euclid", ros_time=lambda: 42.0)
    log.set_context(plan_id="p1", mission_id="m1")
    assert log.emit("action.start", {"i": 0, "n": 2})
    assert log.emit("action.end", {"ok": True}, plan_id="p2", goal_id="g7")
    log.close()
    a, b = _read(path)
    for key in (
        "v t_wall t_mono t_ros machine robot run_id session_id mission_id "
        "goal_id plan_id source type seq payload"
    ).split():
        assert key in a
    assert a["v"] == 1 and a["t_ros"] == 42.0
    assert a["robot"] == "euclid" and a["machine"] == "euclid"
    assert a["run_id"] == "euclid_20261111-185801_a3f2"
    assert a["session_id"] == "s20261111-185000_0001"
    assert a["mission_id"] == "m1" and a["plan_id"] == "p1" and a["goal_id"] == ""
    assert (b["plan_id"], b["goal_id"]) == ("p2", "g7")
    assert (a["seq"], b["seq"]) == (1, 2)
    assert a["source"] == "spot_executor" and a["type"] == "action.start"


def test_redaction_by_key(tmp_path):
    path = str(tmp_path / "e.jsonl")
    log = el.EventLog(path, "s")
    log.emit(
        "x",
        {
            "Password": "p",
            "nested": [{"api_key": "k", "keep": 1}],
            "Authorization": "a",
        },
    )
    log.close()
    (rec,) = _read(path)
    assert rec["payload"] == {
        "Password": el.REDACTED,
        "nested": [{"api_key": el.REDACTED, "keep": 1}],
        "Authorization": el.REDACTED,
    }


def test_numpy_payload_is_serialized(tmp_path):
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "e.jsonl")
    log = el.EventLog(path, "s")
    assert log.emit("x", {"p": np.array([1.0, 2.0]), "f": np.float64(0.5)})
    log.close()
    assert _read(path)[0]["payload"] == {"p": [1.0, 2.0], "f": 0.5}


def test_rotation(tmp_path):
    path = str(tmp_path / "events" / "spot_executor.jsonl")
    log = el.EventLog(path, "spot_executor", rotate_bytes=600)
    for i in range(20):
        assert log.emit("follow.progress", {"i": i})
    log.close()
    rotated = sorted(
        p for p in os.listdir(tmp_path / "events") if p != "spot_executor.jsonl"
    )
    assert rotated and all(p.startswith("spot_executor.") for p in rotated)
    seqs = []
    for name in rotated + ["spot_executor.jsonl"]:
        seqs += [r["seq"] for r in _read(tmp_path / "events" / name)]
    assert sorted(seqs) == list(range(1, 21))


def test_fsync_is_rate_limited(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(el.os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    now = [100.0]
    log = el.EventLog(str(tmp_path / "e.jsonl"), "s", mono=lambda: now[0])
    for _ in range(5):
        log.emit("x")
    assert len(calls) == 1  # first write, then nothing within 2 s
    now[0] += 2.5
    log.emit("x")
    assert len(calls) == 2


def test_appends_across_instances(tmp_path):
    path = str(tmp_path / "e.jsonl")
    el.EventLog(path, "s").emit("a")
    el.EventLog(path, "s").emit("b")
    assert [r["type"] for r in _read(path)] == ["a", "b"]


def test_unwritable_path_never_raises_and_warns_once(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    warnings = []
    log = el.EventLog(
        str(blocker / "events" / "x.jsonl"), "s", warn=warnings.append, retry_s=0
    )
    assert log.emit("a") is False
    assert log.emit("b") is False
    log.close()
    assert len(warnings) == 1


def test_open_append_log_never_truncates(tmp_path):
    path = str(tmp_path / "sub" / "lease_log.txt")
    el.open_append_log(path, "time,event")
    with open(path, "a") as f:
        f.write("1.0,first_run_event\n")
    el.open_append_log(path, "time,event")
    lines = open(path).read().splitlines()
    assert lines[0] == "time,event"
    assert lines.count("time,event") == 1
    assert "1.0,first_run_event" in lines
    markers = [line for line in lines if line.startswith("# start ")]
    assert len(markers) == 2 and f"pid {os.getpid()}" in markers[0]


def test_emit_helper_is_duck_typed_and_safe():
    el.emit(None, "x")  # no hook: nothing happens
    got = []

    class Fb:
        def event(self, etype, payload, **ids):
            got.append((etype, payload, ids))

    el.emit(Fb(), "x", {"a": 1}, plan_id="p")
    assert got == [("x", {"a": 1}, {"plan_id": "p"})]

    class Broken:
        def event(self, *a, **k):
            raise RuntimeError("disk full")

    el.emit(Broken(), "x")  # must not raise

"""str_log.txt / lease_log.txt / events/spot_executor.jsonl survive node
restarts (append, never truncate) and land under $ADT4_RUN_DIR."""

import json
import logging
import os

import pytest

import spot_tools_ros.spot_executor_ros as ser
from spot_tools_ros.spot_executor_ros import (
    RosFeedbackCollector,
    guards_signature,
    parse_mission_id,
)


@pytest.fixture(autouse=True)
def _clean_file_logger():
    yield
    logger = logging.getLogger(ser.__name__)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def _start(output_dir):
    """What a node start does to the log files, without a node."""
    fc = RosFeedbackCollector("odom", output_dir, "INFO")
    fc.logger = logging.getLogger("test_executor_logs")
    fc.start_lease_log()
    return fc


def _lines(path):
    with open(path) as f:
        return f.read().splitlines()


def test_logs_append_across_two_starts_under_run_dir(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "hamilton_20261111-190000_beef"
    monkeypatch.setenv("ADT4_RUN_DIR", str(run))
    param_dir = str(tmp_path / "param_output")

    fc = _start(param_dir)
    assert fc.output_dir == str(run / "spot_executor")
    fc.print("INFO", "first run message")
    with open(os.path.join(fc.output_dir, "lease_log.txt"), "a") as f:
        f.write("1.0,manual_intervention\n")
    fc.event("action.start", {"i": 0})
    fc.event_log.close()
    logger = logging.getLogger(ser.__name__)  # the node process ends
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    fc2 = _start(param_dir)
    fc2.print("INFO", "second run message")
    fc2.event("action.start", {"i": 0})
    for h in logging.getLogger(ser.__name__).handlers:
        h.flush()

    str_log = _lines(run / "spot_executor" / "str_log.txt")
    assert str_log.count("time - level - message") == 1
    assert sum(line.startswith("# start ") for line in str_log) == 2
    assert any("first run message" in line for line in str_log)
    assert any("second run message" in line for line in str_log)

    lease_log = _lines(run / "spot_executor" / "lease_log.txt")
    assert lease_log[0] == "time,event"
    assert "1.0,manual_intervention" in lease_log
    assert sum(line.startswith("# start ") for line in lease_log) == 2

    events = [json.loads(x) for x in _lines(run / "events" / "spot_executor.jsonl")]
    assert [e["type"] for e in events] == ["action.start", "action.start"]
    assert all(e["run_id"] == run.name for e in events)
    assert not os.path.exists(param_dir)


def test_without_run_dir_uses_output_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("ADT4_RUN_DIR", raising=False)
    out = str(tmp_path / "spot_executor")
    fc = _start(out)
    assert fc.output_dir == out
    assert os.path.exists(os.path.join(out, "str_log.txt"))
    assert os.path.exists(os.path.join(out, "lease_log.txt"))
    fc.event("x", {})
    assert os.path.exists(os.path.join(out, "events", "spot_executor.jsonl"))


def test_grasp_episode_event(tmp_path, monkeypatch):
    monkeypatch.delenv("ADT4_RUN_DIR", raising=False)
    fc = RosFeedbackCollector("odom", str(tmp_path), "")
    fc.logger = logging.getLogger("test_executor_logs")
    fc.print("INFO", "Executing `pick` command")
    episode = fc.episode_log.directory
    fc.print("INFO", "Pick skill success: True")
    events = [
        json.loads(x) for x in _lines(tmp_path / "events" / "spot_executor.jsonl")
    ]
    (ev,) = [e for e in events if e["type"] == "grasp.episode"]
    assert ev["payload"]["dir"] == episode and ev["payload"]["ok"] is True
    assert ev["payload"]["rel"].startswith("grasp_episodes/")


def test_parse_mission_id():
    assert parse_mission_id("") == ""
    assert parse_mission_id("m20261111-191900_7c1e") == "m20261111-191900_7c1e"
    assert parse_mission_id('{"mission_id": "m1", "state": "open"}') == "m1"
    assert parse_mission_id('{"mission_id": "m1", "state": "ended"}') == ""
    assert parse_mission_id("{broken") == ""


def test_guards_signature_ignores_battery_jitter():
    g = {
        "battery_percent": 81.3,
        "battery_known": True,
        "lease_owner": "understanding-x",
        "faults": [],
    }
    assert guards_signature(g) == guards_signature({**g, "battery_percent": 84.9})
    assert guards_signature(g) != guards_signature({**g, "battery_percent": 79.0})
    assert guards_signature(g) != guards_signature({**g, "lease_owner": "tablet"})


def test_events_inherit_running_plan_id(tmp_path, monkeypatch):
    monkeypatch.delenv("ADT4_RUN_DIR", raising=False)
    fc = RosFeedbackCollector("odom", str(tmp_path), "")
    fc.event("sequence.start", {"n": 1}, plan_id="plan-9")
    fc.event("follow.progress", {"frac": 0.5})
    fc.event("sequence.end", {"reason": "completed"}, plan_id="plan-9")
    fc.event("guards", {})
    events = [
        json.loads(x) for x in _lines(tmp_path / "events" / "spot_executor.jsonl")
    ]
    assert [e["plan_id"] for e in events] == ["plan-9", "plan-9", "plan-9", ""]

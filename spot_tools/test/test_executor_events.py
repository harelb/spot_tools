"""Executor mission events: lease owner changes and interventions, action
start/end around a sequence, follow progress on FakeSpot, and a broken event
writer that must not change execution."""

import json
from types import SimpleNamespace as NS

import numpy as np
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Pick
from robot_executor_interface.mid_level_planner import IdentityPlanner

import spot_executor.spot_executor as se
from spot_executor.fake_spot import FakeSpot
from spot_skills.event_log import EventLog

TABLET = "bosdyn.android.spotapp spot-tablet"
OURS = "understanding-hamilton"


class Feedback:
    """Records events; everything else is a no-op."""

    def __init__(self):
        self.events = []
        self.lease_log = []
        self.results = []
        self.plan_valid = True
        self.break_out_of_waiting_loop = False

    def event(self, etype, payload, plan_id=None, goal_id=None):
        self.events.append((etype, payload, plan_id))

    def types(self):
        return [e[0] for e in self.events]

    def of(self, etype):
        return [e[1] for e in self.events if e[0] == etype]

    def print(self, *_):
        pass

    def log_lease_takeover(self, event):
        self.lease_log.append(event)

    def action_result(self, *args):
        self.results.append(args)

    def follow_path_feedback(self, *_):
        pass

    def path_following_progress_feedback(self, *_):
        pass

    def path_follow_MLP_feedback(self, *_):
        pass


class LeaseClient:
    def __init__(self, owner):
        self.owner = owner

    def list_leases(self):
        return [NS(lease_owner=NS(client_name=self.owner))]


def _lease_manager(owner, feedback, plan_id="plan-7"):
    lm = se.LeaseManager.__new__(se.LeaseManager)  # no monitor thread
    lm.spot_interface = NS(lease_client=LeaseClient(owner))
    lm.feedback = feedback
    lm.taking_back_lease = False
    lm.plan_id_fn = lambda: plan_id
    lm._seen_owner = None
    return lm


def test_lease_owner_change_logs_owner_and_manual_intervention(monkeypatch):
    fb = Feedback()
    lm = _lease_manager(OURS, fb)
    monkeypatch.setattr(se.time, "sleep", lambda *_: None)

    lm.poll_once()  # initial owner: logged, not an intervention
    lm.poll_once()  # unchanged: nothing
    assert fb.types() == ["lease.owner"]
    assert fb.of("lease.owner")[0] == {"owner": OURS, "previous": None, "ours": True}

    lm.spot_interface.lease_client.owner = TABLET
    lm.poll_once()
    assert fb.types()[1:] == ["lease.owner", "lease.intervention"]
    assert fb.of("lease.owner")[1] == {"owner": TABLET, "previous": OURS, "ours": False}
    assert fb.events[-1] == (
        "lease.intervention",
        {"kind": "manual", "owner": TABLET, "plan_id": "plan-7"},
        "plan-7",
    )
    assert fb.plan_valid is False


def test_tablet_release_then_executor_takes_back(monkeypatch):
    fb = Feedback()
    lm = _lease_manager(TABLET, fb)
    lm.poll_once()
    monkeypatch.setattr(se.time, "sleep", lambda *_: None)

    # _take_back_lease talks to the robot (stow, stand): stub the robot side.
    monkeypatch.setattr(se, "stow_arm", lambda spot: None)
    spot = lm.spot_interface
    spot.take_lease = lambda: setattr(spot.lease_client, "owner", OURS)
    spot.stand = lambda: None

    spot.lease_client.owner = ""  # tablet released the lease
    lm.poll_once()
    lm.poll_once()
    owners = [p["owner"] for p in fb.of("lease.owner")]
    assert owners == [TABLET, "", OURS]
    kinds = [p["kind"] for p in fb.of("lease.intervention")]
    assert kinds == ["executor_takes"]
    assert fb.lease_log == ["spot_executor_takes_lease"]


def test_operator_take_logs_intervention():
    fb = Feedback()
    lm = _lease_manager(TABLET, fb)
    spot = lm.spot_interface
    spot.take_lease = lambda: setattr(spot.lease_client, "owner", OURS)
    assert lm.force_take() == (TABLET, OURS)
    assert fb.of("lease.intervention") == [
        {"kind": "operator_takes", "owner": TABLET, "plan_id": "plan-7"}
    ]
    assert fb.lease_log == [f"operator_takes_lease_from:{TABLET}"]


def _executor(feedback=None):
    spot = NS(
        robot=NS(time_sync=NS(wait_for_sync=lambda: None)), take_lease=lambda: None
    )
    ex = se.SpotExecutor(spot, None, None, None)
    ex.execute_follow = lambda command, fb: True
    ex.execute_pick = lambda command, fb: True
    return ex


def _sequence():
    follow = Follow("map", np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]))
    pick = Pick("map", "mug", np.zeros(3), np.zeros(3), "O12")
    return ActionSequence("plan-1", "hamilton", [follow, pick])


def test_action_start_end_around_sequence():
    fb = Feedback()
    ex = _executor()
    ex.process_action_sequence(_sequence(), fb)
    assert fb.types() == [
        "sequence.start",
        "action.start",
        "action.end",
        "action.start",
        "action.end",
        "sequence.end",
    ]
    starts, ends = fb.of("action.start"), fb.of("action.end")
    assert [(s["i"], s["n"], s["type"]) for s in starts] == [
        (0, 2, "FOLLOW"),
        (1, 2, "PICK"),
    ]
    assert starts[0]["plan_id"] == "plan-1"
    assert starts[0]["target"]["end"] == [2.0, 1.0, 0.0]
    assert starts[1]["target"]["object_id"] == "O12"
    assert all(e["ok"] and e["reason"] == "success" for e in ends)
    assert all(e["duration_s"] >= 0 for e in ends)
    assert fb.of("sequence.end") == [
        {"plan_id": "plan-1", "reason": "completed", "completed": 2, "n": 2}
    ]
    assert all(e[2] == "plan-1" for e in fb.events)
    assert ex.active_plan_id == ""


def test_retry_then_give_up_reasons(monkeypatch):
    monkeypatch.setattr(se.time, "sleep", lambda *_: None)
    fb = Feedback()
    ex = _executor()
    ex.execute_follow = lambda command, fb: False
    ex.process_action_sequence(ActionSequence("p", "r", [_sequence().actions[0]]), fb)
    reasons = [e["reason"] for e in fb.of("action.end")]
    assert reasons == ["failed_retrying", "failed_retrying", "failed_gave_up"]
    assert [s["attempt"] for s in fb.of("action.start")] == [1, 2, 3]


def test_broken_event_writer_does_not_change_execution(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    warnings = []
    log = EventLog(str(blocker / "events.jsonl"), "spot_executor", warn=warnings.append)

    class LoggingFeedback(Feedback):
        def event(self, etype, payload, plan_id=None, goal_id=None):
            return log.emit(etype, payload, plan_id=plan_id, goal_id=goal_id)

    class RaisingFeedback(Feedback):
        def event(self, *a, **k):
            raise OSError("disk full")

    for fb in (LoggingFeedback(), RaisingFeedback()):
        ex = _executor()
        ex.process_action_sequence(_sequence(), fb)
        assert [r[2] for r in fb.results] == ["SUCCESS", "SUCCESS"]
        assert not ex.processing_action_sequence
    assert len(warnings) == 1


def test_fake_spot_follow_emits_progress_and_reached(tmp_path):
    fb = Feedback()
    spot = FakeSpot(init_pose=(0.0, 0.0, 0.0, 0.0))
    ex = se.SpotExecutor(
        spot,
        None,
        lambda parent, child: (np.zeros(3), NS(x=0.0, y=0.0, z=0.0, w=1.0)),
        IdentityPlanner(fb),
        follower_lookahead=1.0,
        goal_tolerance=0.3,
    )
    follow = Follow("map", np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    ex.process_action_sequence(ActionSequence("plan-f", "fake", [follow]), fb)
    progress = fb.of("follow.progress")
    assert progress and 0.0 <= progress[0]["frac"] <= 1.0
    assert progress[0]["dist_left"] > 0
    (end,) = fb.of("action.end")
    assert end["ok"] and end["reason"] == "success"
    assert ex.last_follow_outcome["reason"] == "reached"
    assert "follow.timeout" not in fb.types()
    json.dumps(fb.events)  # payloads are JSON-able


def test_follow_timeout_event(monkeypatch):
    fb = Feedback()
    spot = FakeSpot(init_pose=(0.0, 0.0, 0.0, 0.0))
    from spot_skills import navigation_utils as nu

    outcome = {}
    ok = nu.follow_trajectory_continuous(
        spot,
        np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
        1.0,
        0.3,
        0.0,  # times out on the first check
        IdentityPlanner(fb),
        feedback=fb,
        outcome=outcome,
    )
    assert ok is False and outcome["reason"] == "timeout"
    (timeout,) = fb.of("follow.timeout")
    assert timeout["timeout_s"] == 0.0

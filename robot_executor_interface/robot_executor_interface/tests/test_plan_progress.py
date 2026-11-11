"""The plan-progress gate: ordering and coalition waits across robots."""

import threading
from types import SimpleNamespace as NS

from robot_executor_interface.plan_progress import DONE, READY, PlanProgress


def test_no_ordering_never_blocks():
    assert PlanProgress().blocked_on("p", NS(), "hilbert") == []


def test_waits_for_predecessor_then_coalition_partner():
    sent = []
    progress = PlanProgress(publish=lambda *m: sent.append(m))
    joint = NS(
        after_task_ids=["1"], coalition_robots=["hilbert", "euclid"], task_id="2"
    )
    assert progress.blocked_on("p", joint, "hilbert") == ["task 1 not finished"]
    progress.record("p", "euclid", "1", DONE)
    assert progress.blocked_on("p", joint, "hilbert") == [
        "coalition partners ['euclid'] not ready"
    ]
    assert sent == [("p", "hilbert", "2", "READY")]
    progress.record("p", "euclid", "2", READY)  # heard from euclid
    assert progress.blocked_on("p", joint, "hilbert") == []


def test_after_a_joint_task_waits_for_every_member():
    hilbert, euclid = PlanProgress(), PlanProgress()
    for robot, own in (("hilbert", hilbert), ("euclid", euclid)):
        # each robot hears its own report and its partner's
        own.announce_done("p", robot, "1")
        other = euclid if own is hilbert else hilbert
        other.record("p", robot, "1", DONE)
        other.record("p", robot, f"1@{robot}", DONE)
    visit = NS(after_task_ids=["1@hilbert", "1@euclid"], task_id="2")
    assert euclid.blocked_on("p", visit, "euclid") == []
    solo_successor = NS(after_task_ids=["1"], task_id="3")
    assert hilbert.blocked_on("p", solo_successor, "hilbert") == []


def test_one_member_finishing_does_not_release_the_successor():
    progress = PlanProgress()
    progress.announce_done("p", "hilbert", "1")
    visit = NS(after_task_ids=["1@hilbert", "1@euclid"], task_id="2")
    assert progress.blocked_on("p", visit, "euclid") == ["task 1@euclid not finished"]


def test_reports_while_forgetting_old_plans_do_not_race():
    progress = PlanProgress()
    stop = threading.Event()

    def report():
        i = 0
        while not stop.is_set():
            progress.announce_ready(f"old{i % 50}", "euclid", str(i))
            i += 1

    t = threading.Thread(target=report)
    t.start()
    try:
        for _ in range(2000):
            progress.retain_only("new")
    finally:
        stop.set()
        t.join()

"""Fleet progress through a plan, as ordering rather than as a clock.

A temporal planner produces both an ordering and a set of timepoints. Only the
ordering is worth executing against. Timepoints assume the durations the solver
was given, and a real pick that takes ninety seconds instead of eight makes
every later timepoint wrong -- gating on them would either release an action
early or stall the plan behind a deadline that can no longer be met. Precedence
survives that: "after the object is out of the way" stays true however long the
moving took.

So an action waits for the tasks it must follow to be reported done, and a
coalition task waits for its members to report ready. Nothing here reads a
clock.

Empty ordering fields mean "no constraint", which is what the fast-downward
path produces -- it has no schedule at all -- so that planner is unaffected.
"""

READY = "READY"
DONE = "DONE"


class PlanProgress:
    """What the fleet has finished, and who is standing by.

    One instance per executor. Every robot's reports arrive here, its own
    included, so the gate sees the same picture on each robot without any of
    them being a coordinator.
    """

    def __init__(self, publish=None):
        # publish(plan_id, robot_name, task_id, state) -> None. Left None the
        # tracker still gates on what it hears, which is what makes it testable
        # and what keeps a single-robot plan working with no ROS at all.
        self._publish = publish
        self._done = {}  # plan_id -> {task_id}
        self._ready = {}  # plan_id -> {task_id: {robot_name}}
        self._announced = set()  # (plan_id, task_id) we have already reported READY for

    # ---------- what we hear ----------

    def record(self, plan_id, robot_name, task_id, state):
        """Fold in one report, whoever it came from."""
        if not plan_id or not task_id:
            return
        if state == DONE:
            self._done.setdefault(plan_id, set()).add(task_id)
        elif state == READY:
            self._ready.setdefault(plan_id, {}).setdefault(task_id, set()).add(
                robot_name
            )

    def forget(self, plan_id):
        """Drop one plan's state."""
        self._done.pop(plan_id, None)
        self._ready.pop(plan_id, None)
        self._announced = {k for k in self._announced if k[0] != plan_id}

    def retain_only(self, plan_id):
        """Drop every plan but this one.

        Correctness does not depend on this -- plan ids are uuids, so a stale
        DONE can never open a gate in a later plan -- but a run that repairs its
        plan repeatedly would otherwise accumulate one entry set per superseded
        plan for as long as the node lives.
        """
        for stale in [k for k in list(self._done) + list(self._ready) if k != plan_id]:
            self.forget(stale)

    # ---------- what we say ----------

    def announce_ready(self, plan_id, robot_name, task_id):
        """Say we are in position for a task. Idempotent: the gate re-runs every
        poll and the fleet does not need one message per poll."""
        key = (plan_id, task_id)
        if not plan_id or not task_id or key in self._announced:
            return
        self._announced.add(key)
        self.record(plan_id, robot_name, task_id, READY)
        if self._publish is not None:
            self._publish(plan_id, robot_name, task_id, READY)

    def announce_done(self, plan_id, robot_name, task_id):
        if not plan_id or not task_id:
            return
        self.record(plan_id, robot_name, task_id, DONE)
        if self._publish is not None:
            self._publish(plan_id, robot_name, task_id, DONE)

    # ---------- the gate ----------

    def blocked_on(self, plan_id, action, robot_name):
        """Why this action may not start yet; empty means go.

        Returns reasons rather than a bool so the executor can say what it is
        waiting for. A robot waiting silently and a robot deadlocked look
        identical in a log otherwise.
        """
        after = list(getattr(action, "after_task_ids", None) or [])
        coalition = list(getattr(action, "coalition_robots", None) or [])
        if not after and not coalition:
            return []

        reasons = []
        done = self._done.get(plan_id, set())
        for task_id in after:
            if task_id not in done:
                reasons.append(f"task {task_id} not finished")

        if coalition:
            task_id = getattr(action, "task_id", "") or ""
            # Announce only once our own predecessors are met: telling the
            # fleet we are in position while still walking to it would let a
            # partner start the joint task without us.
            if not reasons:
                self.announce_ready(plan_id, robot_name, task_id)
                waiting = set(coalition) - self._ready.get(plan_id, {}).get(
                    task_id, set()
                )
                if waiting:
                    reasons.append(f"coalition partners {sorted(waiting)} not ready")
        return reasons

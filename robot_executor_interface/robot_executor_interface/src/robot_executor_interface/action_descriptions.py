from dataclasses import dataclass, field

import numpy as np

# Ordering an action is subject to. Appended to each action type rather than
# inherited from a common base, because a base class would put these fields
# first and silently change what every positional construction means.
#
# See plan_progress.PlanProgress for why the executor gates on precedence and
# never on the solver's timepoints. All three are empty for a planner that emits
# no schedule, which leaves the gate inert -- the fast-downward path behaves
# exactly as before.
#
#   task_id           scheduled task this action belongs to; a relocation's
#                     pick, carry and place all share one
#   after_task_ids    tasks that must be DONE first, set on a task's first
#                     action only
#   coalition_robots  robots that must be READY first, for a task allocated to
#                     more than one; empty for a solo task


@dataclass
class ActionSequence:
    plan_id: str
    robot_name: str
    actions: list


@dataclass
class Follow:
    frame: str
    path2d: np.ndarray
    task_id: str = ""
    after_task_ids: list = field(default_factory=list)
    coalition_robots: list = field(default_factory=list)


@dataclass
class Gaze:
    frame: str
    robot_point: np.ndarray
    gaze_point: np.ndarray
    object_id: str
    stow_after: bool = False
    task_id: str = ""
    after_task_ids: list = field(default_factory=list)
    coalition_robots: list = field(default_factory=list)


@dataclass
class Pick:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str
    task_id: str = ""
    after_task_ids: list = field(default_factory=list)
    coalition_robots: list = field(default_factory=list)


@dataclass
class Place:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str
    task_id: str = ""
    after_task_ids: list = field(default_factory=list)
    coalition_robots: list = field(default_factory=list)

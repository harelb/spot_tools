from dataclasses import dataclass

import numpy as np


@dataclass
class ActionSequence:
    plan_id: str
    robot_name: str
    actions: list


@dataclass
class Follow:
    frame: str
    path2d: np.ndarray


@dataclass
class Carry:
    frame: str


@dataclass
class Stow:
    frame: str


@dataclass
class Gaze:
    frame: str
    robot_point: np.ndarray
    gaze_point: np.ndarray
    object_id: str
    stow_after: bool = False


@dataclass
class Pick:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str


@dataclass
class Place:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str

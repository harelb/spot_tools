"""Lossless, ROS-independent transport of compiled executor sequences.

This encodes already compiled actions; it does not compile planner prose or
invent approach poses. The robot-side ROS serializer remains authoritative.
"""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from .action_descriptions import ActionSequence, Follow, Gaze, Pick, Place, Carry, Stow

_TYPES = {c.__name__.upper(): c for c in (Follow, Gaze, Pick, Place, Carry, Stow)}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode_sequence(sequence):
    payload = dict(plan_id=sequence.plan_id, robot_name=sequence.robot_name,
                   actions=[dict(kind=type(a).__name__.upper(), **asdict(a))
                            for a in sequence.actions])
    # Reject NaN/Inf before they can reach a reviewed command.
    return json.loads(json.dumps(payload, default=_plain, allow_nan=False))


def decode_sequence(payload):
    if set(payload) != {"plan_id", "robot_name", "actions"}:
        raise ValueError("invalid compiled sequence fields")
    if any(not isinstance(payload[k], str) or not payload[k] for k in ("plan_id", "robot_name")):
        raise ValueError("compiled sequence identity is missing")
    if not isinstance(payload["actions"], list) or not 1 <= len(payload["actions"]) <= 100:
        raise ValueError("invalid compiled action count")
    actions = []
    for item in payload["actions"]:
        values = dict(item)
        kind = values.pop("kind", None)
        if kind not in _TYPES:
            raise ValueError("unsupported compiled action")
        cls = _TYPES[kind]
        # Gaze.stow_after has a default, but wire messages must carry it explicitly.
        if set(values) != set(cls.__dataclass_fields__):
            raise ValueError("incomplete or unexpected compiled action fields")
        if not isinstance(values["frame"], str) or not values["frame"]:
            raise ValueError("missing action frame")
        for name in ("path2d", "robot_point", "object_point", "gaze_point"):
            if name not in values:
                continue
            arr = np.asarray(values[name], dtype=float)
            valid = (arr.ndim == 2 and arr.shape[1] in (2, 3) and 1 <= len(arr) <= 1000) if name == "path2d" else arr.shape == (3,)
            if not valid or not np.isfinite(arr).all():
                raise ValueError("invalid compiled geometry")
            values[name] = arr
        if "stow_after" in values and type(values["stow_after"]) is not bool:
            raise ValueError("invalid stow flag")
        for name in ("object_id", "object_class"):
            if name in values and (not isinstance(values[name], str) or not values[name]):
                raise ValueError("missing manipulation identity")
        actions.append(cls(**values))
    return ActionSequence(payload["plan_id"], payload["robot_name"], actions)


def sequence_digest(payload):
    canonical = encode_sequence(decode_sequence(payload))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

"""Runtime-guard extraction from a bosdyn RobotState snapshot (PR B8).

Pure proto -> dict, deliberately defensive: the sim's RobotState carries only
kinematic/manipulator state (no battery/power/estop sub-messages), and a real
Spot's snapshot can omit fields transiently — every guard therefore reports a
``*_known`` flag alongside its value, and the consumer decides what an
unknown means (the planner's dispatch gate fails OPEN on unknown, with a
warning: a missing guard stream must not brick planning).
"""
from __future__ import annotations


def extract_runtime_guards(
    robot_state, lease_owned: bool | None = None, lease_owner: str | None = None
) -> dict:
    """RobotState proto (real or sim) -> guard snapshot dict.

    ``lease_owned`` comes from the executor's own lease manager (the proto's
    lease view describes clients generically; the executor knows whether IT
    holds the body lease), ``None`` = unknown.  ``lease_owner`` is the client
    name holding the body lease (e.g. the tablet's ``bosdyn.android.spotapp
    ...``), "" when nobody holds it or it is unknown.
    """
    out = {
        "battery_percent": -1.0, "battery_known": False,
        "estop_pressed": False, "estop_known": False,
        "powered_on": False, "power_known": False,
        "lease_owned": bool(lease_owned), "lease_known": lease_owned is not None,
        "lease_owner": str(lease_owner or ""),
        "faults": [],
    }
    try:
        batteries = list(getattr(robot_state, "battery_states", []) or [])
        if batteries:
            charge = getattr(batteries[0], "charge_percentage", None)
            value = getattr(charge, "value", None)
            if value is not None:
                out["battery_percent"] = float(value)
                out["battery_known"] = True
    except Exception:  # noqa: BLE001 -- proto shape drift must not raise
        pass
    try:
        estops = list(getattr(robot_state, "estop_states", []) or [])
        if estops:
            # bosdyn EStopState.STATE_ESTOPPED == 1, STATE_NOT_ESTOPPED == 2.
            out["estop_pressed"] = any(
                getattr(e, "state", 0) == 1 for e in estops)
            out["estop_known"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        power = getattr(robot_state, "power_state", None)
        motor_state = getattr(power, "motor_power_state", 0) if power else 0
        if motor_state:  # 0 = unknown in the proto enum
            # bosdyn PowerState.MOTOR_POWER_STATE_ON == 2 (OFF == 1).
            out["powered_on"] = motor_state == 2
            out["power_known"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        faults = getattr(robot_state, "behavior_fault_state", None)
        for fault in list(getattr(faults, "faults", []) or []):
            name = getattr(fault, "cause", None)
            out["faults"].append(str(name) if name is not None else "behavior_fault")
    except Exception:  # noqa: BLE001
        pass
    return out

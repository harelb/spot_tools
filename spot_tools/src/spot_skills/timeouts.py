"""Explicit wall-clock budgets shared by physical and simulated skills.

Defaults preserve the robot configuration. Slow simulation deployments may allow
more time for planning and physical execution; completion still requires the
same observed SDK feedback and cancellation checks.
"""
import math
import os


def skill_timeout(default, kind='arm'):
    if kind not in ('arm','grasp'):raise ValueError('Unknown skill timeout kind')
    value=float(os.environ.get('SPOT_SKILL_'+kind.upper()+'_TIMEOUT_S',default))
    if not math.isfinite(value) or not 0<value<=120:
        raise ValueError('Skill timeout must be finite and in (0, 120] seconds')
    return value

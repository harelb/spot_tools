"""Explicit wall-clock budgets shared by physical and simulated skills.

Defaults preserve the robot configuration. Slow simulation deployments may allow
more time for planning and physical execution; completion still requires the
same observed SDK feedback and cancellation checks.
"""
import math
import os


def skill_timeout(default, kind='arm'):
    if kind not in ('arm','grasp','navigation'):raise ValueError('Unknown skill timeout kind')
    value=float(os.environ.get('SPOT_SKILL_'+kind.upper()+'_TIMEOUT_S',default))
    limit=30 if kind=='navigation' else 120
    if not math.isfinite(value) or not 0<value<=limit:
        raise ValueError(f'Skill timeout must be finite and in (0, {limit}] seconds')
    return value

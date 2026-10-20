from types import SimpleNamespace

from spot_tools_ros.spot_executor_ros import take_lease_report


def _lm(result=None, exc=None):
    def force_take():
        if exc:
            raise exc
        return result

    return SimpleNamespace(force_take=force_take)


def test_take_from_tablet():
    ok, msg = take_lease_report(_lm(("bosdyn.android.spotapp x", "understanding-hamilton")))
    assert ok
    assert "before: bosdyn.android.spotapp x" in msg and "after: understanding-hamilton" in msg


def test_take_not_ours_after():
    ok, msg = take_lease_report(_lm(("tablet", "tablet")))
    assert not ok and "not ours" in msg


def test_take_from_nobody():
    ok, msg = take_lease_report(_lm(("", "understanding-x")))
    assert ok and "(nobody)" in msg


def test_take_raises_is_reported():
    ok, msg = take_lease_report(_lm(exc=RuntimeError("ResourceAlreadyClaimed")))
    assert not ok and "RuntimeError" in msg

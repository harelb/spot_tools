import pytest

from spot_tools_ros.spot_executor_ros import resolve_spot_interface


def test_default_is_real():
    assert resolve_spot_interface("", False) == "real"


def test_legacy_fake_flag():
    assert resolve_spot_interface("", True) == "fake"


def test_explicit_sim_wins():
    assert resolve_spot_interface("sim", False) == "sim"
    assert resolve_spot_interface("sim", True) == "sim"


def test_invalid_raises():
    with pytest.raises(ValueError):
        resolve_spot_interface("banana", False)

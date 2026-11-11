"""Operator lease take (~/take_lease). No robot: the lease client is a small fake."""

from types import SimpleNamespace

from spot_executor.spot_executor import LeaseManager

TABLET = "bosdyn.android.spotapp spot-tablet"
OURS = "understanding-hamilton"


class FakeLeaseClient:
    def __init__(self, owner):
        self.owner = owner

    def list_leases(self):
        return [SimpleNamespace(lease_owner=SimpleNamespace(client_name=self.owner))]


class FakeSpotInterface:
    def __init__(self, owner, take_to=OURS, fail=None):
        self.lease_client = FakeLeaseClient(owner)
        self.take_to = take_to
        self.fail = fail
        self.calls = []

    def take_lease(self):
        self.calls.append("take_lease")
        if self.fail:
            raise self.fail
        self.lease_client.owner = self.take_to

    def stand(self):  # must never be called by an operator take
        self.calls.append("stand")


class FakeFeedback:
    def __init__(self):
        self.events = []

    def log_lease_takeover(self, event):
        self.events.append(event)


def _manager(spot, feedback=None):
    # LeaseManager.__init__ starts the (endless) monitor thread; tests build the
    # object without it.
    lm = LeaseManager.__new__(LeaseManager)
    lm.spot_interface = spot
    lm.feedback = feedback
    lm.taking_back_lease = False
    lm.owner_name = spot.lease_client.owner
    return lm


def test_force_take_from_tablet_reports_owners_and_does_not_stand():
    spot = FakeSpotInterface(TABLET)
    fb = FakeFeedback()
    lm = _manager(spot, fb)
    before, after = lm.force_take()
    assert (before, after) == (TABLET, OURS)
    assert lm.owner_name == OURS
    assert spot.calls == ["take_lease"]
    assert fb.events == [f"operator_takes_lease_from:{TABLET}"]


def test_force_take_when_nobody_holds_it():
    lm = _manager(FakeSpotInterface(""))
    assert lm.force_take() == ("", OURS)


def test_concurrent_takes_leave_only_one_keepalive(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from spot_executor.spot import Spot

    entered = threading.Event()
    release = threading.Event()
    taking = []
    keepers = []

    class Keeper:
        closed = False

        def shutdown(self):
            self.closed = True

    def create(*args, **kwargs):
        keeper = Keeper()
        keepers.append(keeper)
        if len(keepers) == 1:
            entered.set()
            assert release.wait(2)
        return keeper

    monkeypatch.setattr("bosdyn.client.lease.LeaseKeepAlive", create)
    spot = Spot.__new__(Spot)
    spot._lease_lock = threading.RLock()
    spot.lease_keep_alive = None
    spot.lease_client = SimpleNamespace(
        take=lambda: taking.append(True), list_leases=lambda: []
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(spot.take_lease)
        assert entered.wait(1)
        second_started = threading.Event()

        def second_take():
            second_started.set()
            spot.take_lease()

        second = pool.submit(second_take)
        assert second_started.wait(1)
        try:
            assert taking == [True]
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert len(keepers) == 2
    assert keepers[0].closed and not keepers[1].closed
    assert spot.lease_keep_alive is keepers[1]

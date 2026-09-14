from types import SimpleNamespace
from unittest.mock import Mock
import threading
from spot_executor.spot_executor import LeaseManager


def test_initial_state_is_ready_before_thread_and_close_joins():
    owner=SimpleNamespace(client_name='understanding-test')
    spot=Mock();spot.lease_client.list_leases.return_value=[SimpleNamespace(lease_owner=owner)]
    manager=LeaseManager(spot)
    try:
        assert manager.owner_name=='understanding-test'
        assert manager.error is None
        assert manager.monitoring_thread.is_alive()
    finally:manager.close()
    assert not manager.monitoring_thread.is_alive()


def test_transport_failure_invalidates_lease_and_monitor_survives():
    owner=SimpleNamespace(client_name='understanding-test')
    spot=Mock();failed=threading.Event()
    calls=[0]
    def leases():
        calls[0]+=1
        if calls[0]>1:raise RuntimeError('endpoint disconnected')
        return [SimpleNamespace(lease_owner=owner)]
    spot.lease_client.list_leases.side_effect=leases
    feedback=Mock();feedback.print.side_effect=lambda *args:failed.set()
    manager=LeaseManager(spot,feedback)
    try:
        assert failed.wait(2)
        assert manager.owner_name=='' and manager.error=='endpoint disconnected'
        assert feedback.break_out_of_waiting_loop
        assert manager.monitoring_thread.is_alive()
    finally:manager.close()

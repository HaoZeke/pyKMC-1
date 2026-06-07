import numpy as np

from pykmc.enginemanager.lmpi.pool import Manager


class FakeSession:
    def __init__(self):
        self.close_calls = []
        self.use_local_calls = 0

    def close(self, wait_status=False):
        self.close_calls.append(wait_status)

    def use_local(self):
        self.use_local_calls += 1


def test_close_all_switches_to_local_sessions_without_global_close():
    manager = Manager.__new__(Manager)
    manager.sessions = [FakeSession(), FakeSession()]
    manager.global_session = FakeSession()
    manager.using_global = True

    manager.close_all()

    assert manager.global_session.use_local_calls == 1
    assert manager.global_session.close_calls == []
    assert [session.close_calls for session in manager.sessions] == [[True], [True]]
    assert manager.using_global is False


def test_get_forces_submits_position_job():
    manager = Manager.__new__(Manager)
    positions = np.array([[0.0, 0.1, 0.2]], dtype=float)
    submitted = []
    sentinel = object()

    def fake_submit_job(method_name, params=None):
        submitted.append((method_name, params))
        return sentinel

    manager.submit_job = fake_submit_job

    got = manager.get_forces(positions=positions)

    assert got is sentinel
    assert submitted[0][0] == "get_forces"
    np.testing.assert_allclose(submitted[0][1]["positions"], positions)

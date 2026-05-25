import queue
import threading

from mpi4py import MPI

from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.lmpi.sessions import MpiApiSession
from pykmc.enginemanager.messenger import MpiMessenger, QueueMessenger
from pykmc.result import ErrorType


class DummyComm:
    def Get_rank(self):
        return 0


def test_rank_zero_queue_local_engine_runs_loop_on_thread(monkeypatch):
    local_messenger = QueueMessenger()
    engine = MpiApiEngine(
        local_messenger=local_messenger,
        local_engine_comm=DummyComm(),
        local_engine_id=1,
        global_messenger=MpiMessenger(MPI.COMM_SELF),
        global_engine_comm=DummyComm(),
        global_engine_id=0,
    )

    monkeypatch.setattr(engine, "start_engine", lambda: setattr(engine, "_is_alive", True))

    caller_thread = threading.current_thread()
    loop_threads = queue.Queue()

    def record_loop_thread():
        loop_threads.put(threading.current_thread() is caller_thread)

    monkeypatch.setattr(engine, "run_engine_loop", record_loop_thread)

    engine.start()

    assert loop_threads.get(timeout=1.0) is False
    assert engine.message_reader_thread is not None


class FakeLammps:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_close_does_not_join_current_reader_thread():
    engine = MpiApiEngine.__new__(MpiApiEngine)
    local_lmp = FakeLammps()
    global_lmp = FakeLammps()
    engine.local_lmp = local_lmp
    engine.global_lmp = global_lmp
    engine.message_reader_thread = threading.current_thread()
    engine.rank = 0
    engine._is_alive = True

    engine.close()

    assert local_lmp.closed is True
    assert global_lmp.closed is True
    assert engine.message_reader_thread is None
    assert engine._is_alive is False


def test_engine_handler_returns_error_envelope_for_operation_exception():
    engine = MpiApiEngine.__new__(MpiApiEngine)
    engine.engine_comm = type("FakeComm", (), {"barrier": lambda self: None})()
    engine.rank = 0

    def failing_operation(_engine):
        raise RuntimeError("pARTn failed")

    engine._operations_map = {"partn_search": failing_operation}

    result = engine._handle_message({"type": "partn_search"})

    assert result["__pykmc_error__"]["handler"] == "partn_search"
    assert "pARTn failed" in result["__pykmc_error__"]["message"]


def test_partn_search_error_envelope_returns_failed_search_result():
    class FakeMessenger:
        def send(self, *_args, **_kwargs):
            pass

        def recv(self, source, tag):
            if tag == 0:
                return {"type": "status", "value": {"alive": True, "busy": False}}
            return {
                "type": "result",
                "value": {
                    "__pykmc_error__": {
                        "handler": "partn_search",
                        "message": "pARTn failed",
                    }
                },
            }

    session = MpiApiSession(
        messenger=FakeMessenger(),
        engine_ranks=[0],
        session_id=0,
    )

    result = session.partn_search(
        config=object(),
        central_atom_idx=0,
        positions=None,
    )

    assert result.is_err()
    assert result.err_value().type is ErrorType.EVENT_NOT_FOUND
    assert "pARTn failed" in result.err_value().message

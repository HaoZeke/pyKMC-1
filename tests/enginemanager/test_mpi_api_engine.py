import queue
import threading

from mpi4py import MPI

from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.messenger import MpiMessenger, QueueMessenger


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

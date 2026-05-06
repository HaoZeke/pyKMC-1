import pandas as pd

from pykmc.event_table import duplicate_event_error_info
from pykmc.result import ErrorType


def test_duplicate_event_error_info_preserves_matched_process_identity():
    event = pd.Series(
        {
            "event_id": "env-a",
            "id_final": "env-b",
            "energy_barrier": 0.21,
            "k": 2.0,
        }
    )
    matched = pd.Series(
        {
            "idx_ref": 5,
            "event_id": "env-a",
            "id_final": "env-b",
            "energy_barrier": 0.20,
            "k": 3.0,
        }
    )

    error = duplicate_event_error_info(event, matched)

    assert error.type is ErrorType.EVENT_NOT_NEW
    assert error.variables == {
        "matched_idx_ref": 5,
        "event_id": "env-a",
        "id_final": "env-b",
        "energy_barrier": 0.20,
        "k": 3.0,
    }

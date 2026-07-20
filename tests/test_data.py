from src.data import build_windows, generate_synthetic_data


def test_synthetic_data_and_windows() -> None:
    frame = generate_synthetic_data(n_machines=4, min_cycles=35, max_cycles=40, seed=7)
    assert {"machine_id", "cycle", "temperature", "vibration", "rul"}.issubset(frame.columns)
    windows = build_windows(frame, sequence_length=12)
    assert windows.x.ndim == 3
    assert windows.x.shape[1] == 12
    assert len(windows.x) == len(windows.y)

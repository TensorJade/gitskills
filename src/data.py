from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "current", "rpm"]


@dataclass
class WindowedData:
    x: np.ndarray
    y: np.ndarray
    scaler: StandardScaler
    feature_columns: list[str]


def generate_synthetic_data(
    n_machines: int = 48,
    min_cycles: int = 90,
    max_cycles: int = 145,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate run-to-failure manufacturing sensor data."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for machine_id in range(1, n_machines + 1):
        total_cycles = int(rng.integers(min_cycles, max_cycles + 1))
        temp_base = rng.normal(62.0, 2.2)
        vibration_base = rng.normal(1.10, 0.10)
        pressure_base = rng.normal(6.5, 0.15)
        current_base = rng.normal(18.0, 0.8)
        rpm_base = rng.normal(1480.0, 18.0)
        fault_start = rng.uniform(0.48, 0.68)

        for cycle in range(1, total_cycles + 1):
            age = cycle / total_cycles
            degradation = max(0.0, (age - fault_start) / (1.0 - fault_start))
            degradation = min(degradation, 1.0)
            rows.append(
                {
                    "machine_id": machine_id,
                    "cycle": cycle,
                    "temperature": temp_base + 2.2 * age + 16.0 * degradation**2 + rng.normal(0, 0.65),
                    "vibration": vibration_base + 0.22 * age + 2.1 * degradation**3 + rng.normal(0, 0.07),
                    "pressure": pressure_base - 0.10 * age - 1.25 * degradation**2 + rng.normal(0, 0.045),
                    "current": current_base + 0.6 * age + 6.2 * degradation**2 + rng.normal(0, 0.28),
                    "rpm": rpm_base - 8.0 * age - 75.0 * degradation**2 + rng.normal(0, 5.0),
                    "rul": total_cycles - cycle,
                }
            )
    return pd.DataFrame(rows)


def load_cmapss_fd001(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    columns = ["machine_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    frame["rul"] = frame.groupby("machine_id")["cycle"].transform("max") - frame["cycle"]
    mapping = {
        "sensor_2": "temperature",
        "sensor_4": "vibration",
        "sensor_7": "pressure",
        "sensor_11": "current",
        "sensor_15": "rpm",
    }
    result = frame[["machine_id", "cycle", *mapping.keys(), "rul"]].rename(columns=mapping)
    return result.astype({"machine_id": int, "cycle": int})


def load_demo_data(cmapss_path: str | Path = "data/raw/train_FD001.txt") -> tuple[pd.DataFrame, str]:
    path = Path(cmapss_path)
    if path.exists():
        return load_cmapss_fd001(path), "NASA C-MAPSS FD001"
    return generate_synthetic_data(), "合成制造设备退化数据"


def split_machine_ids(machine_ids: Iterable[int], train_ratio: float = 0.75, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(set(machine_ids)), dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = max(1, int(len(ids) * train_ratio))
    return ids[:cut], ids[cut:]


def build_windows(
    frame: pd.DataFrame,
    sequence_length: int = 24,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
    feature_columns: list[str] | None = None,
) -> WindowedData:
    feature_columns = feature_columns or ["cycle", *SENSOR_COLUMNS]
    ordered = frame.sort_values(["machine_id", "cycle"]).copy()
    ordered = ordered.astype({column: float for column in feature_columns})
    scaler = scaler or StandardScaler()
    if fit_scaler:
        scaler.fit(ordered[feature_columns])
    ordered.loc[:, feature_columns] = scaler.transform(ordered[feature_columns])

    x_rows, y_rows = [], []
    for _, machine in ordered.groupby("machine_id", sort=True):
        features = machine[feature_columns].to_numpy(dtype=np.float32)
        targets = machine["rul"].to_numpy(dtype=np.float32)
        for end in range(sequence_length, len(machine) + 1):
            x_rows.append(features[end - sequence_length : end])
            y_rows.append(float(targets[end - 1]))
    if not x_rows:
        raise ValueError("No windows were created. Reduce sequence_length.")
    return WindowedData(np.stack(x_rows), np.asarray(y_rows, dtype=np.float32), scaler, feature_columns)


def latest_window(machine_frame: pd.DataFrame, scaler: StandardScaler, feature_columns: list[str], sequence_length: int) -> np.ndarray:
    if len(machine_frame) < sequence_length:
        raise ValueError("Not enough cycles for an LSTM prediction window.")
    values = machine_frame.sort_values("cycle")[feature_columns].tail(sequence_length)
    return scaler.transform(values).astype(np.float32)[None, ...]

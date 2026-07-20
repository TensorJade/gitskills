from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import WindowedData, build_windows, latest_window, split_machine_ids
from .model import RULLSTM, TrainingResult, predict, train_model


@dataclass
class DemoBundle:
    frame: pd.DataFrame
    source_name: str
    model: RULLSTM
    window_data: WindowedData
    training: TrainingResult
    sequence_length: int


def train_demo_bundle(frame: pd.DataFrame, source_name: str, sequence_length: int = 24, epochs: int = 12) -> DemoBundle:
    train_ids, valid_ids = split_machine_ids(frame["machine_id"].unique())
    train_frame = frame[frame["machine_id"].isin(train_ids)]
    valid_frame = frame[frame["machine_id"].isin(valid_ids)]
    train_windows = build_windows(train_frame, sequence_length=sequence_length)
    valid_windows = build_windows(
        valid_frame,
        sequence_length=sequence_length,
        scaler=train_windows.scaler,
        fit_scaler=False,
        feature_columns=train_windows.feature_columns,
    )
    training = train_model(
        train_windows.x,
        train_windows.y,
        valid_windows.x,
        valid_windows.y,
        epochs=epochs,
    )
    return DemoBundle(frame, source_name, training.model, train_windows, training, sequence_length)


def predict_machine_cycle(bundle: DemoBundle, machine_id: int, cycle: int) -> float:
    machine = bundle.frame[
        (bundle.frame["machine_id"] == machine_id) & (bundle.frame["cycle"] <= cycle)
    ]
    window = latest_window(
        machine,
        scaler=bundle.window_data.scaler,
        feature_columns=bundle.window_data.feature_columns,
        sequence_length=bundle.sequence_length,
    )
    return float(predict(bundle.model, window)[0])


def risk_probability(predicted_rul: float, midpoint: float = 32.0, scale: float = 7.0) -> float:
    value = 1.0 / (1.0 + np.exp((predicted_rul - midpoint) / scale))
    return float(np.clip(value, 0.0, 1.0))


def maintenance_advice(predicted_rul: float) -> tuple[str, str]:
    if predicted_rul <= 15:
        return "严重", "建议立即停机检查，并创建紧急维护工单。"
    if predicted_rul <= 30:
        return "高风险", "建议在下一生产班次前安排维护，重点检查振动与温升。"
    if predicted_rul <= 60:
        return "关注", "建议提高巡检频率，并提前准备备件与维护窗口。"
    return "正常", "设备状态稳定，保持当前监测频率。"

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch

from src.data import load_demo_data
from src.pipeline import train_demo_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the predictive-maintenance LSTM model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    frame, source_name = load_demo_data()
    bundle = train_demo_bundle(
        frame,
        source_name=source_name,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.model.state_dict(), args.output / "lstm_rul_state.pt")
    joblib.dump(bundle.window_data.scaler, args.output / "feature_scaler.joblib")
    metadata = {
        "source": source_name,
        "sequence_length": args.sequence_length,
        "feature_columns": bundle.window_data.feature_columns,
        "validation_rmse": bundle.training.validation_rmse,
        "validation_mae": bundle.training.validation_mae,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

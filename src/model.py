from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class RULLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, num_layers: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.register_buffer("target_scale", torch.tensor(1.0, dtype=torch.float32))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


@dataclass
class TrainingResult:
    model: RULLSTM
    train_losses: list[float]
    validation_rmse: float
    validation_mae: float


def predict(model: RULLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(x.astype(np.float32))) * model.target_scale
    return output.cpu().numpy()


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model = RULLSTM(input_size=x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=0.10)
    target_scale = max(float(np.max(y_train)), 1.0)
    model.target_scale.fill_(target_scale)
    y_train_scaled = (y_train / target_scale).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled)),
        batch_size=batch_size,
        shuffle=True,
    )

    losses: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_rmse = float("inf")
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(features)
            sample_count += len(features)
        losses.append(epoch_loss / max(sample_count, 1))

        validation_prediction = predict(model, x_valid)
        validation_rmse = float(np.sqrt(np.mean((validation_prediction - y_valid) ** 2)))
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    predictions = predict(model, x_valid)
    rmse = float(np.sqrt(np.mean((predictions - y_valid) ** 2)))
    mae = float(np.mean(np.abs(predictions - y_valid)))
    return TrainingResult(model, losses, rmse, mae)

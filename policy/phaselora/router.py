"""Weak-supervision router for PhaseLoRA P / E descriptors."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

GRIPPERS = (6, 13)
HISTORY = 6


def pe_targets(action: np.ndarray, lam_g: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Paper Eqs. (5)–(8), using joint-space motion instead of EE xyz/rot."""
    length, dim = action.shape
    arm = [index for index in range(dim) if index not in GRIPPERS]
    grippers = [index for index in GRIPPERS if index < dim]
    motion = np.zeros(length, dtype=np.float32)
    jerk = np.zeros(length, dtype=np.float32)
    for time in range(length):
        delta = np.zeros(dim, dtype=np.float32) if time == 0 else action[time] - action[time - 1]
        motion[time] = np.linalg.norm(delta[arm]) + lam_g * float(np.abs(delta[grippers]).sum())
        if time >= 2:
            jerk[time] = float(np.linalg.norm(action[time] - 2.0 * action[time - 1] + action[time - 2]))

    def _robust(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-6
        return (values - median) / mad

    precision = 1.0 / (1.0 + np.exp(_robust(motion)))
    event = 1.0 / (1.0 + np.exp(-_robust(jerk)))
    return precision.astype(np.float32), event.astype(np.float32)


def action_history(action: np.ndarray, start: int, length: int = HISTORY) -> np.ndarray:
    history = np.zeros((length, action.shape[-1]), dtype=np.float32)
    if start <= 0:
        return history
    taken = action[max(0, start - length) : start]
    history[-len(taken) :] = taken
    return history


class PhaseRouter(nn.Module):
    """GRU over recent actions plus a visual/state context, sigmoid P and E."""

    def __init__(self, visual_dim: int, state_dim: int = 14, action_dim: int = 14, hidden: int = 128):
        super().__init__()
        self.visual_dim = visual_dim
        self.gru = nn.GRU(action_dim, hidden, batch_first=True)
        self.context = nn.Linear(visual_dim + state_dim, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, visual: torch.Tensor, state: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_hist = self.gru(history)[0][:, -1]
        hidden_ctx = F.silu(self.context(torch.cat([visual, state], dim=-1)))
        pe = torch.sigmoid(self.head(hidden_hist + hidden_ctx))
        return pe[:, 0], pe[:, 1]

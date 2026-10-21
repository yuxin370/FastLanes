#!/usr/bin/env python3
"""RGB-no-more optimizer, independent decay, and update-indexed schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


class IndependentWeightDecay:
    """Published additive weight-decay optimizer without momentum state."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        base_learning_rate: float,
        coefficient: float,
    ) -> None:
        self.base_learning_rate = float(base_learning_rate)
        self.coefficient = float(coefficient)
        selected = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ".weight" in name and "lrnorm" not in name
        ]
        if not selected:
            raise RuntimeError("published independent WeightDecay selected no parameters")
        self.parameter_names = tuple(name for name, _parameter in selected)
        self.parameters = tuple(parameter for _name, parameter in selected)
        self.current_learning_rate = self.base_learning_rate

    @torch.no_grad()
    def step(self, current_learning_rate: float) -> None:
        self.current_learning_rate = float(current_learning_rate)
        scale = (
            self.current_learning_rate / self.base_learning_rate * self.coefficient
        )
        for parameter in self.parameters:
            parameter.add_(parameter, alpha=-scale)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_learning_rate": self.base_learning_rate,
            "coefficient": self.coefficient,
            "current_learning_rate": self.current_learning_rate,
            "parameter_names": list(self.parameter_names),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if float(payload["base_learning_rate"]) != self.base_learning_rate:
            raise ValueError("WeightDecay base learning rate differs from checkpoint")
        if float(payload["coefficient"]) != self.coefficient:
            raise ValueError("WeightDecay coefficient differs from checkpoint")
        if tuple(payload["parameter_names"]) != self.parameter_names:
            raise ValueError("WeightDecay parameter selection differs from checkpoint")
        self.current_learning_rate = float(payload["current_learning_rate"])


@dataclass
class PublishedUpdateScheduler:
    optimizer: torch.optim.Optimizer
    base_learning_rate: float
    warmup_updates: int
    total_updates: int
    completed_updates: int = 0

    def __post_init__(self) -> None:
        if self.base_learning_rate <= 0:
            raise ValueError("base learning rate must be positive")
        if self.warmup_updates < 0 or self.total_updates <= 0:
            raise ValueError("scheduler update counts are invalid")
        if self.warmup_updates >= self.total_updates:
            raise ValueError("warmup must be shorter than the complete training schedule")

    def learning_rate_for_update(self, update: int) -> float:
        """Learning rate used by one-based optimizer update ``update``.

        This preserves the published loop's warmup convention: current_itr is
        incremented before the adjustment, so updates 1..9999 use
        ``base_lr*(update+1)/10000`` and update 10000 uses ``base_lr``.
        """

        if not 1 <= update <= self.total_updates:
            raise ValueError(
                f"optimizer update {update} is outside [1,{self.total_updates}]"
            )
        if update < self.warmup_updates:
            return self.base_learning_rate * (update + 1) / self.warmup_updates
        if update == self.warmup_updates:
            return self.base_learning_rate
        progress = (update - self.warmup_updates) / (
            self.total_updates - self.warmup_updates
        )
        return self.base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def prepare_next_update(self) -> float:
        update = self.completed_updates + 1
        learning_rate = self.learning_rate_for_update(update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def complete_update(self) -> None:
        if self.completed_updates >= self.total_updates:
            raise RuntimeError("scheduler advanced past total optimizer updates")
        self.completed_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_learning_rate": self.base_learning_rate,
            "warmup_updates": self.warmup_updates,
            "total_updates": self.total_updates,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {
            "base_learning_rate": self.base_learning_rate,
            "warmup_updates": self.warmup_updates,
            "total_updates": self.total_updates,
        }
        observed = {key: payload[key] for key in expected}
        if observed != expected:
            raise ValueError(
                f"scheduler checkpoint contract mismatch: expected {expected}, got {observed}"
            )
        completed = int(payload["completed_updates"])
        if not 0 <= completed <= self.total_updates:
            raise ValueError("scheduler checkpoint completed_updates is invalid")
        self.completed_updates = completed


def build_published_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-4,
    total_updates: int,
    warmup_updates: int = 10_000,
) -> tuple[torch.optim.AdamW, IndependentWeightDecay, PublishedUpdateScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    decayer = IndependentWeightDecay(
        model,
        base_learning_rate=learning_rate,
        coefficient=weight_decay,
    )
    scheduler = PublishedUpdateScheduler(
        optimizer=optimizer,
        base_learning_rate=float(learning_rate),
        warmup_updates=int(warmup_updates),
        total_updates=int(total_updates),
    )
    return optimizer, decayer, scheduler

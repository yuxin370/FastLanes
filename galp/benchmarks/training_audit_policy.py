"""Shared host-synchronization policy for benchmark training loops.

The policy deliberately separates numerical safety from high-frequency audit
reads.  Gradient finiteness remains an optimizer correctness gate: every
optimizer update makes one host decision before ``optimizer.step()``.  Loss
and logits observations can stay on device after the strict prefix and are
folded into that decision with a sticky flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


AUDIT_POLICY_SCHEMA = "galp-training-audit-policy-v1"
AUDIT_MODES = ("strict", "benchmark")
DEFAULT_AUDIT_MODE = "benchmark"
DEFAULT_STRICT_UPDATES = 100


@dataclass(frozen=True)
class TrainingAuditPolicy:
    mode: str = DEFAULT_AUDIT_MODE
    strict_updates: int = DEFAULT_STRICT_UPDATES

    def __post_init__(self) -> None:
        if self.mode not in AUDIT_MODES:
            raise ValueError(f"audit mode must be one of {AUDIT_MODES}")
        if self.strict_updates < 0:
            raise ValueError("strict_updates must be non-negative")

    def is_strict_update(self, completed_updates: int) -> bool:
        if completed_updates < 0:
            raise ValueError("completed_updates must be non-negative")
        return self.mode == "strict" or completed_updates < self.strict_updates

    def as_contract(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_POLICY_SCHEMA,
            "mode": self.mode,
            "strict_updates": self.strict_updates,
            "gradient_finite_check": "before-every-optimizer-step",
            "deferred_finite_boundary": "optimizer-update",
        }

    @classmethod
    def from_contract(cls, value: Mapping[str, Any] | None) -> "TrainingAuditPolicy":
        if value is None:
            return cls()
        if value.get("schema_version") != AUDIT_POLICY_SCHEMA:
            raise ValueError("unsupported training audit policy schema")
        policy = cls(
            mode=str(value.get("mode", "")),
            strict_updates=int(value.get("strict_updates", -1)),
        )
        expected = policy.as_contract()
        if dict(value) != expected:
            raise ValueError("training audit policy contract is not canonical")
        return policy


@dataclass
class TrainingAuditCounters:
    observed_microbatches: int = 0
    strict_microbatches: int = 0
    loss_host_reads: int = 0
    finite_host_reads: int = 0
    gradient_gate_reads: int = 0
    parameter_gate_reads: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "observed_microbatches": self.observed_microbatches,
            "strict_microbatches": self.strict_microbatches,
            "loss_host_reads": self.loss_host_reads,
            "finite_host_reads": self.finite_host_reads,
            "gradient_gate_reads": self.gradient_gate_reads,
            "parameter_gate_reads": self.parameter_gate_reads,
        }


def _all_finite_device(values: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
    result = torch.ones((), dtype=torch.bool, device=device)
    found = False
    for value in values:
        found = True
        result.logical_and_(torch.isfinite(value.detach()).all())
    if not found:
        return torch.zeros((), dtype=torch.bool, device=device)
    return result


class TrainingAuditState:
    """Per-training-loop audit state shared by every data backend.

    ``completed_updates`` is the global optimizer update count.  It is never
    reset at epoch, pool, or backend boundaries.
    """

    def __init__(
        self,
        policy: TrainingAuditPolicy,
        *,
        completed_updates: int,
        device: torch.device,
    ) -> None:
        if completed_updates < 0:
            raise ValueError("completed_updates must be non-negative")
        self.policy = policy
        self.completed_updates = completed_updates
        self.device = device
        self.counters = TrainingAuditCounters()
        self._finite_sticky = torch.ones((), dtype=torch.bool, device=device)
        self._deferred_loss = torch.zeros((), dtype=torch.float64, device=device)
        self._deferred_samples = 0

    @property
    def strict_update(self) -> bool:
        return self.policy.is_strict_update(self.completed_updates)

    def observe(self, loss: torch.Tensor, logits: torch.Tensor, sample_count: int) -> float:
        """Observe one microbatch and return any immediately materialized loss sum."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.counters.observed_microbatches += 1
        loss_detached = loss.detach()
        logits_detached = logits.detach()
        if self.strict_update:
            self.counters.strict_microbatches += 1
            self.counters.finite_host_reads += 2
            if not bool(torch.isfinite(loss_detached).item()) or not bool(
                torch.isfinite(logits_detached).all().item()
            ):
                raise FloatingPointError("non-finite loss/logits")
            self.counters.loss_host_reads += 1
            return float(loss_detached.item()) * sample_count

        observed_finite = torch.isfinite(loss_detached) & torch.isfinite(
            logits_detached
        ).all()
        self._finite_sticky.logical_and_(observed_finite)
        self._deferred_loss.add_(loss_detached.to(torch.float64) * sample_count)
        self._deferred_samples += sample_count
        return 0.0

    def check_gradients_and_clip(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        max_norm: float,
    ) -> torch.Tensor:
        """Fail closed before an optimizer update, using one host decision."""
        parameter_list = list(parameters)
        gradients = [
            parameter.grad
            for parameter in parameter_list
            if parameter.grad is not None
        ]
        if not gradients:
            raise FloatingPointError("optimizer update has no gradients")
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameter_list,
            max_norm=max_norm,
            error_if_nonfinite=False,
        )
        update_finite = torch.isfinite(total_norm) & self._finite_sticky
        self.counters.finite_host_reads += 1
        self.counters.gradient_gate_reads += 1
        if not bool(update_finite.item()):
            raise FloatingPointError("non-finite loss, logits, or gradients")
        self._finite_sticky.fill_(True)
        return total_norm

    def complete_update(self) -> None:
        self.completed_updates += 1

    def read_deferred_loss(self) -> tuple[float, int]:
        """Read and reset loss accumulated after the strict prefix."""
        samples = self._deferred_samples
        if samples == 0:
            return 0.0, 0
        self.counters.loss_host_reads += 1
        value = float(self._deferred_loss.item())
        self._deferred_loss.zero_()
        self._deferred_samples = 0
        return value, samples

    def check_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Run a fused diagnostic parameter scan with one host read."""
        finite = _all_finite_device(parameters, self.device)
        self.counters.finite_host_reads += 1
        self.counters.parameter_gate_reads += 1
        if not bool(finite.item()):
            raise FloatingPointError("non-finite model parameters")

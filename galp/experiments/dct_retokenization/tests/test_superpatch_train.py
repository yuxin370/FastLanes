#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from galp.experiments.dct_retokenization.train_superpatch import (
    DifferentialWarmupCosine,
    checkpoint_pending_validation_epoch,
    load_config,
    logit_distillation_loss,
    signatures_differ_only_by_device,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class DifferentialScheduleTest(unittest.TestCase):
    def test_group_ratio_is_preserved(self) -> None:
        first = torch.nn.Parameter(torch.ones(()))
        second = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW(
            [
                {"params": [first], "lr": 1e-3},
                {"params": [second], "lr": 1e-4},
            ]
        )
        scheduler = DifferentialWarmupCosine(optimizer, [1e-3, 1e-4], 2, 10)
        for update in range(10):
            rates = scheduler.prepare_next_update()
            if update < 9:
                self.assertAlmostEqual(rates[0] / rates[1], 10.0)
            else:
                self.assertEqual(rates, [0.0, 0.0])
            scheduler.complete_update()
        self.assertEqual(scheduler.completed_updates, 10)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_state_round_trip(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = DifferentialWarmupCosine(optimizer, [1e-3], 1, 4)
        scheduler.prepare_next_update()
        scheduler.complete_update()
        state = scheduler.state_dict()
        restored = DifferentialWarmupCosine(optimizer, [1e-3], 1, 4)
        restored.load_state_dict(state)
        self.assertEqual(restored.completed_updates, 1)

    def test_device_only_signature_migration(self) -> None:
        previous = {"device_name": "H100", "config_sha256": "same", "updates": 10}
        current = {"device_name": "PRO6000", "config_sha256": "same", "updates": 10}
        self.assertTrue(signatures_differ_only_by_device(previous, current))
        changed_config = dict(current, config_sha256="different")
        self.assertFalse(signatures_differ_only_by_device(previous, changed_config))
        self.assertFalse(signatures_differ_only_by_device(previous, previous))

    def test_pending_validation_checkpoint_is_replayed_before_next_epoch(self) -> None:
        payload = {
            "completed_epoch": 3,
            "resume_epoch": 3,
            "next_microbatch_index": 0,
            "epoch_samples": 128,
            "epoch_elapsed_seconds": 12.5,
            "pending_validation_epoch": 3,
        }
        self.assertEqual(checkpoint_pending_validation_epoch(payload, 128), 3)
        committed = dict(payload, pending_validation_epoch=None, epoch_samples=0)
        self.assertIsNone(checkpoint_pending_validation_epoch(committed, 128))

    def test_pending_validation_checkpoint_rejects_incomplete_epoch(self) -> None:
        payload = {
            "completed_epoch": 3,
            "resume_epoch": 3,
            "next_microbatch_index": 0,
            "epoch_samples": 127,
            "epoch_elapsed_seconds": 12.5,
            "pending_validation_epoch": 3,
        }
        with self.assertRaisesRegex(ValueError, "incomplete epoch statistics"):
            checkpoint_pending_validation_epoch(payload, 128)


class LogitDistillationLossTest(unittest.TestCase):
    def test_prespecified_kd_config_contract(self) -> None:
        config = load_config(EXPERIMENT_ROOT / "configs" / "superpatch_kd.json")
        self.assertEqual(config["stage"], "stage_kd")
        self.assertEqual(
            config["distillation"]["teacher"], "K32/N196 pretrained checkpoint"
        )
        self.assertEqual(config["distillation"]["temperature"], 2.0)
        self.assertEqual(config["distillation"]["lambda"], 1.0)

    def test_identical_logits_have_zero_loss(self) -> None:
        logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]])
        loss = logit_distillation_loss(logits, logits, temperature=2.0)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_teacher_is_detached_and_student_receives_gradient(self) -> None:
        student = torch.tensor([[1.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[2.0, -2.0]], requires_grad=True)
        loss = logit_distillation_loss(student, teacher, temperature=2.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_rejects_invalid_temperature_and_shape(self) -> None:
        logits = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            logit_distillation_loss(logits, logits, temperature=0.0)
        with self.assertRaises(ValueError):
            logit_distillation_loss(logits, torch.zeros(2, 4), temperature=1.0)


if __name__ == "__main__":
    unittest.main()

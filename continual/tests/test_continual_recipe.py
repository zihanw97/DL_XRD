"""Verify label availability and distillation without constructing the large CNN."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

CONTINUAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTINUAL))

import data_setup as data
import launch_suite
from replay_memory import ReplayMemory
import training_engine as engine
import worker


class ContinualRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = data.read_assignments(data.DEFAULT_ASSIGNMENTS)

    def test_ten_percent_masks_preserve_original_inputs_and_roles(self):
        self.assertTrue(data.validate_assignments(self.rows))
        # Independent of CSV serialization and the two intentionally changed
        # mask fields: all input identities, pixels, truths and roles must stay.
        with data.DEFAULT_ASSIGNMENTS.open(newline="") as stream:
            original_fields = [
                {k: v for k, v in row.items()
                 if k not in {"training_label", "anchor_status"}}
                for row in csv.DictReader(stream)
            ]
        digest = hashlib.sha256(json.dumps(original_fields, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest,
            "8a80e32f6b7fc7c7bf77bf421a87c971b26b30165d1c749b4c5503f73369666e")
        counts = Counter((r["domain"], r["split"]) for r in self.rows)
        self.assertEqual([counts["source", s] for s in data.SPLITS], [723, 89, 89])
        self.assertEqual([counts["target", s] for s in data.SPLITS], [642, 79, 79])
        training = Counter()
        anchors = Counter()
        for row in self.rows:
            if row["domain"] == "target" and row["split"] == "train":
                group = row["block"], row["label"]
                training[group] += 1
                anchors[group] += row["anchor_status"] == "anchor"
                self.assertEqual(row["training_label"],
                    row["label"] if row["anchor_status"] == "anchor" else -1)
            else:
                self.assertEqual(row["training_label"], row["label"])
                self.assertEqual(row["anchor_status"],
                    "labeled_source" if row["domain"] == "source" and
                    row["split"] == "train" else "not_training")
        self.assertEqual(sum(anchors.values()), 64)
        with data.DEFAULT_ASSIGNMENTS.open(newline="") as stream:
            legacy = list(csv.DictReader(stream))
        legacy_anchors = {r["legacy_cache_id"] for r in legacy
                          if r["anchor_status"] == "anchor"}
        active_anchors = {r["legacy_cache_id"] for r in self.rows
                          if r["anchor_status"] == "anchor"}
        self.assertEqual(len(legacy_anchors), 127)
        self.assertLessEqual(active_anchors, legacy_anchors)
        self.assertEqual(data.sha256(data.DEFAULT_ASSIGNMENTS),
                         data.EXPECTED_ASSIGNMENTS_SHA256)
        for group, n in training.items():
            with self.subTest(block=group[0], label=group[1]):
                self.assertEqual(anchors[group], min(n, max(1, math.floor(.1 * n + .5))))

    def test_material_disjoint_profile_keeps_the_same_label_availability(self):
        remapped = data.material_disjoint_assignments(self.rows)
        original = {r["sample_id"]: r for r in self.rows}
        self.assertEqual(len(remapped), len(original))
        with tempfile.TemporaryDirectory() as folder:
            profiles = data.prepare_profiles(folder)
            for profile in profiles.values():
                loaded = data.read_assignments(profile["path"])
                self.assertTrue(data.validate_assignments(loaded))
                self.assertEqual({r["legacy_cache_id"] for r in loaded
                                  if r["anchor_status"] == "anchor"},
                                 {r["legacy_cache_id"] for r in self.rows
                                  if r["anchor_status"] == "anchor"})
        original_anchors = {r["sample_id"] for r in self.rows
                            if r["anchor_status"] == "anchor"}
        self.assertEqual({r["sample_id"] for r in remapped
                          if r["anchor_status"] == "anchor"}, original_anchors)
        for row in remapped:
            self.assertEqual(
                {k: v for k, v in row.items() if k not in {"block", "new_block"}},
                {k: v for k, v in original[row["sample_id"]].items()
                 if k not in {"block", "new_block"}})

    def test_temperature_two_reaches_every_method_and_stage(self):
        self.assertEqual(engine.effective_config()["temperature"], 2.)
        recipe = launch_suite.training_recipe()
        self.assertEqual(recipe["temperature"], 2.)
        provenance = json.loads((CONTINUAL / "recipe_provenance.json").read_text())
        self.assertEqual(provenance["configuration"]["distillation"]["temperature"], 2.)
        for method in launch_suite.METHODS:
            for stage in range(1, 5):
                with self.subTest(method=method, stage=stage):
                    cfg = worker.base_phase_config({"training": recipe}, method, stage)
                    self.assertEqual(engine.effective_config(cfg)["temperature"], 2.)

    def test_target_replay_uses_only_available_training_labels(self):
        scope = SimpleNamespace(domain="target", split="train", block=1,
            key="train_target_1", ids=["revealed", "hidden"],
            training_y=torch.tensor([1, -1]), y=torch.tensor([1, 2]),
            x=torch.tensor([[1., 0.], [0., 1.]]))
        memory = ReplayMemory("target", capacity=2)
        memory.add([(scope, 0), (scope, 1)])
        self.assertEqual(memory.tensors()[1].tolist(), [1, -1])
        scope.y[1] = 0
        self.assertEqual(memory.tensors()[1].tolist(), [1, -1])
        self.assertEqual(memory.manifest()[1]["label_origin"], "unlabeled")
        scope.training_y[1] = 0
        with self.assertRaises(ValueError):
            memory.assert_label_budget()

    def test_current_distillation_is_temperature_two_equal_domain_kl(self):
        class Teacher(torch.nn.Module):
            def forward(self, inputs):
                return inputs, inputs, torch.zeros_like(inputs)

        source = torch.tensor([[2., -1., .5], [-.5, 1., 2.]])
        target = torch.tensor([[0., 3., -2.]])
        captured = [torch.tensor([[0., 1., -1.], [2., -.5, .2]], requires_grad=True),
                    torch.tensor([[1., 0., -.5]], requires_grad=True)]
        batches = {"source": (source, torch.tensor([0, 2])),
                   "target": (target, torch.tensor([-1]))}
        cfg = engine.effective_config({"kd_current": True})
        actual = engine.current_distillation(Teacher(), batches, captured,
                                             torch.device("cpu"), cfg)
        temperature = 2.
        expected = sum((F.softmax(truth / temperature, dim=1) *
            (F.log_softmax(truth / temperature, dim=1) -
             F.log_softmax(student / temperature, dim=1)))
            .sum(dim=1).mean() for truth, student in zip((source, target), captured)) / 2 * temperature ** 2
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
        actual.backward()
        self.assertTrue(all(logits.grad is not None and
                            bool(torch.isfinite(logits.grad).all()) for logits in captured))


if __name__ == "__main__":
    unittest.main()

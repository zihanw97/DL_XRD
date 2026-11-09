"""Bounded training-only replay with auditable original label availability.

Memories hold references to at most ``capacity`` cached images per domain.
Selection consults source training labels, target training masks, and encoder
features; it never consults hidden target truth or validation/test inputs.
"""
from __future__ import annotations

import random
import torch
from torch.utils.data import DataLoader, TensorDataset


def _sid(entry):
    scope, index = entry
    return scope.ids[index]


class ReplayMemory:
    def __init__(self, domain, capacity, selection="herding", seed=42):
        if domain not in ("source", "target"):
            raise ValueError("Memory domain must be source or target")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError("Memory capacity must be a nonnegative integer")
        if selection not in ("herding", "reservoir"):
            raise ValueError("Unknown replay selection")
        self.domain, self.capacity, self.selection = domain, capacity, selection
        self.seed, self.entries = int(seed), []
        self._rng = random.Random(seed)
        self._priorities, self._original_training_labels = {}, {}
        self.seen_count = 0

    def label(self, entry):
        scope, index = entry
        label = int(scope.training_y[index])
        sid = _sid(entry)
        if label not in (-1, 0, 1, 2) or (self.domain == "source" and label < 0):
            raise ValueError("Invalid available replay training label")
        if sid in self._original_training_labels and self._original_training_labels[sid] != label:
            raise ValueError("An original replay label mask changed")
        return label

    def _check(self, entry):
        scope, index = entry
        if scope.split != "train" or scope.domain != self.domain or not 0 <= index < len(scope.ids):
            raise ValueError("Replay may only retain its own domain's training rows")
        self.label(entry)

    def add(self, entries):
        """Bounded compatibility helper; campaign refresh uses selection below."""
        entries = list(entries)
        for entry in entries:
            self._check(entry)
        if self.capacity == 0:
            self.entries = []
            return
        unique = {_sid(e): e for e in self.entries}
        unique.update({_sid(e): e for e in entries})
        self.entries = list(unique.values())[-self.capacity:]
        if self.selection == "reservoir":
            for entry in self.entries:
                self._priorities.setdefault(_sid(entry), self._rng.random())
            self._priorities = {_sid(e): self._priorities[_sid(e)] for e in self.entries}
        self._original_training_labels = {_sid(e): self.label(e) for e in self.entries}
        self.assert_label_budget()

    def assert_label_budget(self):
        if len(self.entries) > self.capacity or len({_sid(e) for e in self.entries}) != len(self.entries):
            raise ValueError("Replay capacity or unique-image budget violated")
        for entry in self.entries:
            self._check(entry)
        return True

    def sample(self, rng, per_class=20):
        if per_class < 1:
            raise ValueError("Replay minibatch budget must be positive")
        self.assert_label_budget()
        if self.domain == "target":
            # Target labels, even revealed anchors, do not select replay rows.
            return rng.sample(self.entries, min(per_class, len(self.entries)))
        if self.selection == "reservoir":
            return rng.sample(self.entries, min(3 * per_class, len(self.entries)))
        selected = []
        for label in range(3):
            group = [entry for entry in self.entries if self.label(entry) == label]
            selected.extend(rng.sample(group, min(per_class, len(group))))
        return selected

    def tensors(self, entries=None):
        entries = self.entries if entries is None else list(entries)
        if not entries:
            return None
        for entry in entries:
            self._check(entry)
        return (torch.stack([scope.x[index] for scope, index in entries]),
                torch.tensor([self.label(e) for e in entries], dtype=torch.long))

    def loader(self, batch_size=32):
        values = self.tensors()
        if values is None:
            raise ValueError("An empty memory has no precision loader")
        return DataLoader(TensorDataset(*values), batch_size=batch_size, shuffle=False, num_workers=0)

    def manifest(self):
        self.assert_label_budget()
        return [dict(sample_id=_sid(e), cache_key=e[0].key, row=e[1], domain=self.domain,
                     block=int(e[0].block), training_label=self.label(e),
                     label_origin=("labeled_source" if self.domain == "source" else
                                   "original_stage_anchor" if self.label(e) >= 0 else "unlabeled"))
                for e in self.entries]

    @torch.no_grad()
    def _features(self, model, entries, device, batch_size):
        if model is None:
            raise ValueError("Feature-based replay selection requires a selected encoder")
        was_training = model.training
        model.eval()
        pieces = []
        try:
            for start in range(0, len(entries), batch_size):
                x = torch.stack([s.x[i] for s, i in entries[start:start + batch_size]]).to(device)
                z = model.features(x)
                if z.ndim != 2 or not torch.isfinite(z).all():
                    raise ValueError("Invalid replay-selection embedding")
                pieces.append(z.detach().cpu().float())
        finally:
            model.train(was_training)
        return torch.cat(pieces)

    @staticmethod
    def _herd(features, quota):
        """Greedy mean matching, with deterministic index tie resolution."""
        quota = min(quota, len(features))
        if not quota:
            return []
        center = features.mean(0)
        running = torch.zeros_like(center)
        selected, available = [], torch.ones(len(features), dtype=torch.bool)
        for k in range(quota):
            distance = ((running[None] + features) / (k + 1) - center).square().sum(1)
            distance[~available] = torch.inf
            index = int(distance.argmin())
            selected.append(index)
            available[index] = False
            running += features[index]
        return selected

    @staticmethod
    def _diverse(features, quota):
        """Label-free farthest-first target selection in standardized units."""
        quota = min(quota, len(features))
        if not quota:
            return []
        centered = features - features.mean(0)
        scale = features.std(0, unbiased=False).clamp_min(1e-6)
        z = centered / scale
        first = int(z.square().sum(1).argmin())
        selected = [first]
        distance = (z - z[first]).square().sum(1)
        distance[first] = -1
        while len(selected) < quota:
            index = int(distance.argmax())
            selected.append(index)
            distance = torch.minimum(distance, (z - z[index]).square().sum(1))
            distance[selected] = -1
        return selected

    def refresh(self, model, scope, device="cpu", seed=None, batch_size=32):
        """Update from previous bounded memory plus this observed train block."""
        new = [(scope, i) for i in range(len(scope.ids))]
        for entry in new:
            self._check(entry)
        old_ids = {_sid(e) for e in self.entries}
        if old_ids & {_sid(e) for e in new}:
            raise ValueError("A new stream block duplicates an earlier retained image")
        prior_size = len(self.entries)
        self.seen_count += len(new)
        if self.capacity == 0:
            self.entries = []
            self._priorities = {}
            self._original_training_labels = {}
        elif self.selection == "reservoir":
            # Independent uniform priorities are equivalent to uniform reservoir
            # sampling. Unselected past images are neither stored nor revisited.
            pool = list(self.entries) + new
            priorities = dict(self._priorities)
            priorities.update({_sid(e): self._rng.random() for e in new})
            self.entries = sorted(pool, key=lambda e: (-priorities[_sid(e)], _sid(e)))[:self.capacity]
            self._priorities = {_sid(e): priorities[_sid(e)] for e in self.entries}
            self._original_training_labels = {_sid(e): self.label(e) for e in self.entries}
        else:
            pool = list(self.entries) + new
            # Seed changes tie order without consulting target truth.
            rng = random.Random(self.seed if seed is None else seed)
            rng.shuffle(pool)
            features = self._features(model, pool, device, batch_size)
            if self.domain == "source":
                labels = [self.label(e) for e in pool]
                classes = sorted(set(labels))
                chosen = []
                for ordinal, label in enumerate(classes):
                    candidates = [i for i, y in enumerate(labels) if y == label]
                    quota = self.capacity // len(classes) + (ordinal < self.capacity % len(classes))
                    chosen.extend(candidates[j] for j in self._herd(features[candidates], quota))
            else:
                chosen = self._diverse(features, self.capacity)
            self.entries = [pool[j] for j in chosen]
            self._original_training_labels = {_sid(e): self.label(e) for e in self.entries}
        self.assert_label_budget()
        return dict(domain=self.domain, capacity=self.capacity, selection=self.selection,
                    seen_training_rows=self.seen_count, prior_retained=prior_size,
                    new_training_rows=len(new), retained=len(self.entries),
                    target_truth_used=False, memory=self.manifest())


def refresh_memories(memories, scopes, stage, model, device, seed, batch_size=32):
    if set(memories) != {"source", "target"}:
        raise ValueError("Exactly two domain memories are required")
    reports = {}
    for domain, memory in memories.items():
        part = [s for s in scopes if s.split == "train" and s.domain == domain and s.block == stage]
        if len(part) != 1:
            raise ValueError("Expected one current training scope per domain")
        reports[domain] = memory.refresh(model, part[0], device, seed + 1000 * stage, batch_size)
    return reports

"""Generic state/loader utilities; no cluster-refinement implementation."""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

def _evaluation_loader(loader):
    if isinstance(loader, DataLoader) and not isinstance(loader.dataset, IterableDataset):
        return DataLoader(loader.dataset, batch_size=loader.batch_size or 1,
                          shuffle=False, drop_last=False, num_workers=0,
                          collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
                          generator=loader.generator)
    return loader

def _cpu_weights(model):
    return {name: tensor.detach().to(device="cpu", copy=True)
            for name, tensor in model.state_dict().items()}

def _atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)

def _generator_states(loaders):
    states = {}
    for name, loader in loaders.items():
        for slot, holder in (("loader", loader), ("sampler", getattr(loader, "sampler", None))):
            generator = getattr(holder, "generator", None)
            states[f"{name}:{slot}"] = generator.get_state().clone() if generator is not None else None
    return states

def _restore_generators(loaders, states):
    for name, loader in loaders.items():
        for slot, holder in (("loader", loader), ("sampler", getattr(loader, "sampler", None))):
            generator = getattr(holder, "generator", None)
            state = states[f"{name}:{slot}"]
            if (generator is None) != (state is None):
                raise ValueError("Resume loader/sampler generator configuration changed")
            if generator is not None:
                generator.set_state(state)

def _rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}

def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("Resume checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda"])

"""Tests for orchestrator module."""

import importlib
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def _dummy_dataloader():
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def orchestrator_module(fresh_prometheus_registry, monkeypatch):
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def set(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

        def inc(self, *args, **kwargs):
            return None

    import types
    import sys

    aiofiles_stub = types.ModuleType("aiofiles")
    aiofiles_stub.open = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "aiofiles", aiofiles_stub)

    monkeypatch.setattr("prometheus_client.Gauge", DummyMetric)
    monkeypatch.setattr("prometheus_client.Counter", DummyMetric)
    monkeypatch.setattr("prometheus_client.Histogram", DummyMetric)
    return importlib.import_module("orchestrator")


def test_amp_config_merge(tmp_path, orchestrator_module):
    config = {
        "output_dir": str(tmp_path),
        "amp": {"max_grad_accum_steps": 8, "save_every_n_steps": 5},
    }
    orch = orchestrator_module.Orchestrator(config)
    amp_cfg = orch._amp_config()

    assert amp_cfg["max_grad_accum_steps"] == 8
    assert amp_cfg["save_every_n_steps"] == 5
    assert Path(amp_cfg["checkpoint_dir"]).parent == tmp_path


@pytest.mark.unit
def test_run_training_amp_creates_checkpoint(tmp_path, orchestrator_module):
    config = {
        "output_dir": str(tmp_path),
        "mode": "amp",
        "epochs": 1,
        "max_steps_per_epoch": 1,
        "learning_rate": 1e-3,
        "amp": {"activation_checkpointing": False},
    }
    orch = orchestrator_module.Orchestrator(config)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    dataloader = _dummy_dataloader()

    orch.run_training(model, dataloader)

    checkpoint_dir = Path(tmp_path) / "amp"
    files = list(checkpoint_dir.glob("checkpoint_step_*"))
    assert files, "AMP training should create a checkpoint file"

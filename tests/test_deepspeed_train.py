"""Tests for DeepSpeed integration helpers."""

import importlib
import os

import pytest
import torch


@pytest.fixture
def deepspeed_adapter(fresh_prometheus_registry, monkeypatch):
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

    monkeypatch.setattr("prometheus_client.Gauge", DummyMetric)
    monkeypatch.setattr("prometheus_client.Counter", DummyMetric)
    monkeypatch.setattr("prometheus_client.Histogram", DummyMetric)
    return importlib.import_module("distributed.deepspeed_adapter")


def test_build_default_config_merges_overrides(tmp_path, deepspeed_adapter):
    overrides = {"train_batch_size": 64, "nvme_offload_dir": str(tmp_path / "nvme")}
    cfg = deepspeed_adapter.build_default_config(str(tmp_path), overrides=overrides)

    assert cfg["train_batch_size"] == 64
    assert cfg["zero_optimization"]["offload_param"]["nvme_path"] == str(tmp_path / "nvme")
    assert cfg["checkpoint_dir"] == os.path.join(str(tmp_path), "checkpoints", "deepspeed")


def test_deepspeed_trainer_requires_cuda(tmp_path, deepspeed_adapter):
    pytest.importorskip("deepspeed")

    config = deepspeed_adapter.build_default_config(str(tmp_path))
    with pytest.raises(RuntimeError, match="requires CUDA-enabled devices"):
        deepspeed_adapter.DeepSpeedTrainer(model=torch.nn.Linear(4, 4), dataloader=None, config=config)  # type: ignore[arg-type]

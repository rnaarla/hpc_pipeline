"""Tests for recovery manager utilities."""

from pathlib import Path

import pytest
import torch

from fault_tolerance.recovery_manager import RecoveryManager


@pytest.fixture()
def cpu_checkpoint_env(monkeypatch):
    """Force torch.load to map checkpoints to CPU for test purposes."""
    original_load = torch.load

    def _cpu_load(path, map_location=None, *args, **kwargs):
        return original_load(path, map_location="cpu", *args, **kwargs)

    monkeypatch.setattr(torch, "load", _cpu_load)


def _create_manager(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer, RecoveryManager(model, optimizer, output_dir=str(tmp_path))


def test_save_and_load_checkpoint(tmp_path, cpu_checkpoint_env):
    model, optimizer, manager = _create_manager(tmp_path)

    manager.save_checkpoint(step=1)
    state = manager.load_checkpoint(step=1)

    assert state is not None
    assert state["step"] == 1
    assert (Path(tmp_path) / "ckpt_rank0_step1.pt").exists()


def test_load_missing_checkpoint_returns_none(tmp_path, cpu_checkpoint_env):
    _, _, manager = _create_manager(tmp_path)
    assert manager.load_checkpoint(step=999) is None


def test_stitch_checkpoints(tmp_path, cpu_checkpoint_env):
    model, optimizer, manager = _create_manager(tmp_path)
    manager.save_checkpoint(step=5)

    # Simulate another rank checkpoint
    other_path = Path(tmp_path) / "ckpt_rank1_step5.pt"
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "optimizer": optimizer.state_dict(),
            "scaler": None,
            "step": 5,
            "rank": 1,
        },
        other_path,
    )

    stitched = RecoveryManager.stitch_checkpoints(str(tmp_path), 5, world_size=2)
    data = torch.load(stitched, map_location="cpu")

    assert stitched.endswith("global_ckpt.pt")
    assert "rank0" in data["model"]
    assert "rank1" in data["model"]

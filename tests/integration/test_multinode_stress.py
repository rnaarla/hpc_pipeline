"""Integration test for checkpoint stitching under multi-rank conditions."""

from pathlib import Path

import torch

from fault_tolerance.recovery_manager import RecoveryManager


def _build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )


def test_multinode_checkpoint_stitching(tmp_path):
    model0 = _build_model()
    opt0 = torch.optim.AdamW(model0.parameters(), lr=1e-3)
    mgr0 = RecoveryManager(model0, opt0, output_dir=str(tmp_path), rank=0)
    mgr0.save_checkpoint(step=1)

    model1 = _build_model()
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    mgr1 = RecoveryManager(model1, opt1, output_dir=str(tmp_path), rank=1)
    mgr1.save_checkpoint(step=1)

    stitched_path = Path(RecoveryManager.stitch_checkpoints(str(tmp_path), step=1, world_size=2))
    assert stitched_path.exists()

    global_state = torch.load(stitched_path)
    assert global_state["step"] == 1
    assert len(global_state["model"]) == 2
    assert set(global_state["model"].keys()) == {"rank0", "rank1"}

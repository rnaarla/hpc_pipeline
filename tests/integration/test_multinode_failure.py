"""Integration test validating recovery manager failure handling."""

import torch

from fault_tolerance.recovery_manager import RecoveryManager


def test_corrupted_checkpoint_triggers_failure(tmp_path):
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = RecoveryManager(model, optimizer, output_dir=str(tmp_path), rank=0)

    manager.save_checkpoint(step=7)
    ckpt_path = tmp_path / "ckpt_rank0_step7.pt"
    assert ckpt_path.exists()

    # Corrupt checksum file to simulate failure
    with open(f"{ckpt_path}.md5", "w", encoding="utf-8") as handle:
        handle.write("badchecksum")

    restored = manager.load_checkpoint(step=7)
    assert restored is None, "Corrupted checkpoints must not be restored"

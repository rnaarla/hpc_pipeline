"""Integration test exercising the DeepSpeed adapter fallback."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from distributed.deepspeed_adapter import DeepSpeedTrainer


def test_deepspeed_fallback_training(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 5),
    )

    inputs = torch.randn(20, 16)
    targets = torch.randint(0, 5, (20,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)

    config = {
        "optimizer": {"params": {"lr": 1e-3, "weight_decay": 0.0}},
        "checkpoint_dir": str(tmp_path / "ds_checkpoints"),
    }

    trainer = DeepSpeedTrainer(model, dataloader, config)
    avg_loss = trainer.train_epoch(epoch=0, max_steps=3)
    assert avg_loss >= 0.0

    trainer.save_checkpoint(tag="ep0")
    saved = list((tmp_path / "ds_checkpoints").glob("checkpoint_rank0_step_0*.pt"))
    assert saved, "DeepSpeed fallback should emit a checkpoint"

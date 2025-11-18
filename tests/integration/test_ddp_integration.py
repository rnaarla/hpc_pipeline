"""Integration test validating DDP trainer in single-rank mode."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from distributed.ddp_training import DDPTrainer


def test_ddp_trainer_single_rank(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 5),
    )

    config = {
        "checkpoint_dir": str(tmp_path / "ddp"),
        "amp_enabled": False,
        "num_workers": 0,
        "bucket_cap_mb": 4,
    }

    trainer = DDPTrainer(model, config)
    dataset = TensorDataset(torch.randn(24, 16), torch.randint(0, 5, (24,)))
    dataloader = trainer.create_dataloader(dataset, batch_size=6, shuffle=False)

    loss = trainer.train_epoch(epoch=0, dataloader=dataloader, max_steps=3)
    assert loss >= 0.0

    checkpoint = trainer.save_checkpoint(step=1, loss=loss)
    if trainer.rank == 0:
        assert checkpoint.exists()

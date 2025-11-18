"""Smoke tests for DDP training helpers."""

import torch

from distributed.ddp_training import DDPTrainer


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index):
        x = torch.randn(8)
        y = torch.randint(0, 2, (1,)).item()
        return x, y


def _tiny_model():
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )


def test_ddp_trainer_single_rank_cpu(tmp_path, fresh_prometheus_registry):
    """Ensure DDPTrainer operates on CPU-only environments when world_size == 1."""
    config = {
        "amp_enabled": False,
        "checkpoint_dir": str(tmp_path / "ckpts"),
        "checkpoint_every_n_steps": 5,
    }

    trainer = DDPTrainer(_tiny_model(), config)
    assert trainer.device.type == "cpu"

    dataset = _TinyDataset()
    dataloader = trainer.create_dataloader(dataset, batch_size=4, shuffle=False)
    loss = trainer.train_epoch(epoch=0, dataloader=dataloader, max_steps=2)

    assert loss >= 0
    checkpoint = trainer.save_checkpoint(step=1, loss=loss)
    assert checkpoint.exists()

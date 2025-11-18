"""Tests for distributed DDP trainer."""

import types

import pytest
import torch

from distributed.ddp_training import DDPTrainer


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        x = torch.randn(16)
        y = torch.randint(0, 2, (1,)).item()
        return x, y


def _simple_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def test_ddp_trainer_cpu_epoch(tmp_path, fresh_prometheus_registry):
    config = {
        "amp_enabled": False,
        "checkpoint_dir": str(tmp_path / "ckpts"),
        "checkpoint_every_n_steps": 10,
    }

    trainer = DDPTrainer(_simple_model(), config)
    dataset = DummyDataset()
    dataloader = trainer.create_dataloader(dataset, batch_size=4, shuffle=False)

    loss = trainer.train_epoch(epoch=0, dataloader=dataloader, max_steps=2)
    assert loss >= 0


def test_ddp_trainer_save_checkpoint_cpu(tmp_path, fresh_prometheus_registry):
    config = {
        "amp_enabled": False,
        "checkpoint_dir": str(tmp_path / "ckpts"),
        "checkpoint_every_n_steps": 1,
    }

    trainer = DDPTrainer(_simple_model(), config)
    trainer.ddp_model = types.SimpleNamespace(module=trainer.model)
    step_path = trainer.save_checkpoint(step=1, loss=0.5)
    assert step_path.exists()


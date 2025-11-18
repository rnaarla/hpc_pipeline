"""Integration test covering orchestrator data pipeline and AMP training."""

from itertools import islice

import torch
from torch.utils.data import DataLoader, TensorDataset

from orchestrator import Orchestrator


def _build_toy_model(input_dim: int = 16, num_classes: int = 4) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, num_classes),
    )


def test_pipeline_end_to_end(tmp_path):
    output_dir = tmp_path / "runs"
    config = {
        "output_dir": str(output_dir),
        "mode": "amp",
        "epochs": 1,
        "max_steps_per_epoch": 2,
        "learning_rate": 1e-3,
        "enable_data_pipeline": True,
        "enable_training": True,
        "enable_profiler": False,
        "enable_benchmarking": False,
        "enable_recovery": False,
        "enable_observability": False,
        "data_pipeline": {
            "nvme_path": str(tmp_path / "nvme"),
            "remote_path": str(tmp_path / "remote"),
            "prefill_samples": 4,
            "prefetch_queue_size": 2,
            "prefetch_workers": 1,
        },
        "amp": {
            "checkpoint_dir": str(output_dir / "amp"),
            "save_every_n_steps": 1,
            "amp_enabled": False,
        },
    }

    orchestrator = Orchestrator(config)

    # Validate data pipeline yields synthetic samples
    pipeline_loader = orchestrator.run_data_pipeline()
    samples = list(islice(pipeline_loader, 2))
    assert samples, "Data pipeline should yield samples"
    first_sample, first_label = samples[0]
    label_value = first_label.item() if hasattr(first_label, "item") else int(first_label)
    assert isinstance(label_value, int)
    assert first_sample.ndim == 2

    # Train AMP pipeline on a toy dataset (CPU friendly)
    inputs = torch.randn(12, 16)
    targets = torch.randint(0, 4, (12,))
    training_loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = _build_toy_model()
    orchestrator.run_training(model, training_loader)

    checkpoints = list((output_dir / "amp").glob("checkpoint_step_*"))
    assert checkpoints, "AMP training should emit checkpoints"

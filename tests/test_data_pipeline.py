"""Tests for multi-tier data pipeline components."""

import asyncio
import types
import sys
from pathlib import Path

import pytest
import torch

aiofiles_stub = types.ModuleType("aiofiles")
aiofiles_stub.open = lambda *args, **kwargs: None
sys.modules.setdefault("aiofiles", aiofiles_stub)

import data.pipeline as pipeline_mod


@pytest.fixture
def cpu_pipeline(monkeypatch, tmp_path, fresh_prometheus_registry):
    """Create a MultiTierDataPipeline with HBM tier patched for CPU testing."""

    class DummyHBMTier(pipeline_mod.MemoryTier):
        def __init__(self, capacity_gb: float = 80, rank: int = 0):
            super().__init__("hbm", capacity_gb, rank)

        def put(self, key: str, data: torch.Tensor) -> bool:
            return super().put(key, data.clone())

        def get(self, key: str):
            return super().get(key)

    class DummyRemoteTier(pipeline_mod.MemoryTier):
        def __init__(self, base_path: str, capacity_gb: float = 1.0, rank: int = 0):
            super().__init__("remote", capacity_gb, rank)
            self.base_path = Path(base_path)
            self.base_path.mkdir(exist_ok=True)

        async def load_async(self, key: str):
            file_path = self.base_path / f"{key}.pt"
            if not file_path.exists():
                return None
            await asyncio.sleep(0)
            return torch.load(file_path, map_location="cpu")

        def save(self, key: str, data: torch.Tensor):
            file_path = self.base_path / f"{key}.pt"
            torch.save(data, file_path)

    monkeypatch.setattr(pipeline_mod, "HBMTier", DummyHBMTier)
    monkeypatch.setattr(pipeline_mod, "RemoteTier", DummyRemoteTier)

    config = {
        "nvme_path": str(tmp_path / "nvme"),
        "remote_path": str(tmp_path / "remote"),
        "prefetch_queue_size": 2,
        "prefetch_workers": 1,
    }
    return pipeline_mod.MultiTierDataPipeline(config, rank=0)


def test_get_data_promotes_through_tiers(cpu_pipeline):
    cpu_pipeline.put_data("sample", torch.ones(2, 2))

    data = cpu_pipeline.get_data("sample")

    assert torch.allclose(data, torch.ones(2, 2))
    # Data should now live in the top tier (patched HBM)
    cached = cpu_pipeline.hbm.get("sample")
    assert cached is not None
    assert torch.allclose(cached, torch.ones(2, 2))


@pytest.mark.asyncio
async def test_async_prefetcher_fetches_batches(cpu_pipeline):
    cpu_pipeline.remote.save("prefetch_key", torch.arange(4))

    await cpu_pipeline.prefetcher.start_prefetching(
        ["prefetch_key"], pipeline=cpu_pipeline
    )
    key, tensor = await cpu_pipeline.prefetcher.get_prefetched_data()

    assert key == "prefetch_key"
    assert torch.allclose(tensor, torch.arange(4))

    cpu_pipeline.prefetcher.stop()


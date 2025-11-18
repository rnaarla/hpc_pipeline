import os
import io
import time
import tempfile
from pathlib import Path

from data_pipeline.streaming_cache import StreamingCache


def test_stage_and_ram_cache_roundtrip(tmp_path: Path):
    # Prepare a local source file
    src_dir = tmp_path / "src"
    nvme_dir = tmp_path / "nvme"
    src_dir.mkdir()
    nvme_dir.mkdir()
    content = b"hello-streaming-cache" * 1024  # ~22KB
    src_file = src_dir / "data.bin"
    src_file.write_bytes(content)

    url = f"file://{src_file}"

    cache = StreamingCache(nvme_dir=str(nvme_dir), ram_capacity_bytes=10 * 1024 * 1024)
    data1, local1 = cache.get(url)
    assert os.path.exists(local1)
    assert data1 == content

    # Second access should be a RAM cache hit
    data2, local2 = cache.get(url)
    assert local2 == local1
    assert data2 == content


def test_stale_lock_breaker_allows_progress(tmp_path: Path):
    src_file = tmp_path / "data.bin"
    src_file.write_bytes(b"x" * 1024)
    url = f"file://{src_file}"

    cache = StreamingCache(
        nvme_dir=str(tmp_path / "nvme"),
        ram_capacity_bytes=1024 * 1024,
        lock_timeout_s=2,
        lock_poll_s=0.1,
        stale_lock_s=0.2,
    )
    os.makedirs(cache.nvme_dir, exist_ok=True)
    local = cache._nvme_path(url)
    lock_path, _ = cache._lock_paths(local)

    # Create a stale lock
    Path(lock_path).write_text("lock")
    old = time.time() - 10
    os.utime(lock_path, (old, old))

    # Should break the stale lock and complete
    out = cache.stage(url)
    assert out == local
    assert os.path.exists(out)
    assert not os.path.exists(lock_path)

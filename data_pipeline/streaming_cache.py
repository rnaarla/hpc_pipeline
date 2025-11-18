import os
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple, Iterable, List

import fsspec
from concurrent.futures import ThreadPoolExecutor, as_completed
from prometheus_client import Counter, Gauge

CACHE_HIT = Counter("hpc_pipeline_cache_hits_total", "Streaming cache hits")
CACHE_MISS = Counter("hpc_pipeline_cache_misses_total", "Streaming cache misses")
CACHE_ERRORS = Counter("hpc_pipeline_cache_errors_total", "Streaming cache errors")
CACHE_DOWNLOAD_BYTES = Counter(
    "hpc_pipeline_cache_download_bytes_total", "Total bytes downloaded into cache"
)
BYTES_STAGED = Gauge("hpc_pipeline_cache_bytes_staged", "Bytes staged on NVMe")
BYTES_RAM = Gauge("hpc_pipeline_cache_bytes_ram", "Bytes in RAM cache")


def gds_available() -> bool:
    # Lightweight probe for NVIDIA GPUDirect Storage
    return os.path.exists("/dev/nvidia-fs") or os.path.exists("/etc/cufile.json")


class StreamingCache:
    def __init__(
        self,
        nvme_dir: str = "./nvme",
        ram_capacity_bytes: int = 256 * 1024 * 1024,
        prefix: str = "stage",
        max_nvme_bytes: Optional[int] = None,
        lock_timeout_s: int = 600,
        lock_poll_s: float = 0.2,
        stale_lock_s: float = 600.0,
        max_retries: int = 5,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self.nvme_dir = nvme_dir
        self.ram_capacity = ram_capacity_bytes
        self.prefix = prefix
        self.max_nvme_bytes = max_nvme_bytes
        self.lock_timeout_s = lock_timeout_s
        self.lock_poll_s = lock_poll_s
        self.stale_lock_s = stale_lock_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        os.makedirs(self.nvme_dir, exist_ok=True)
        self._ram_lru: "OrderedDict[str, bytes]" = OrderedDict()
        self._ram_size = 0
        self._lock = threading.Lock()
        BYTES_STAGED.set(self._nvme_bytes())
        BYTES_RAM.set(0)

    def _key(self, url: str) -> str:
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        return f"{self.prefix}-{h}"

    def _nvme_path(self, url: str) -> str:
        return os.path.join(self.nvme_dir, self._key(url))

    def _nvme_bytes(self) -> int:
        total = 0
        try:
            for f in os.listdir(self.nvme_dir):
                p = os.path.join(self.nvme_dir, f)
                if os.path.isfile(p) and not f.endswith(".lock") and not f.endswith(".tmp"):
                    total += os.path.getsize(p)
        except FileNotFoundError:
            pass
        return total

    def _lock_paths(self, local: str) -> Tuple[str, str]:
        return f"{local}.lock", f"{local}.tmp"

    def _acquire_download_lock(self, lock_path: str) -> bool:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _release_download_lock(self, lock_path: str) -> None:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            CACHE_ERRORS.inc()

    def _ram_put(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._ram_lru:
                self._ram_size -= len(self._ram_lru[key])
                self._ram_lru.pop(key, None)
            self._ram_lru[key] = data
            self._ram_size += len(data)
            self._evict_ram_if_needed()
            BYTES_RAM.set(self._ram_size)

    def _ram_get(self, key: str) -> Optional[bytes]:
        with self._lock:
            v = self._ram_lru.get(key)
            if v is not None:
                self._ram_lru.move_to_end(key, last=True)
            return v

    def _evict_ram_if_needed(self) -> None:
        while self._ram_size > self.ram_capacity and self._ram_lru:
            _, v = self._ram_lru.popitem(last=False)
            self._ram_size -= len(v)

    def _evict_nvme_if_needed(self, incoming_bytes: int = 0) -> None:
        if self.max_nvme_bytes is None:
            return
        try:
            entries = []
            for f in os.listdir(self.nvme_dir):
                if f.endswith(".lock") or f.endswith(".tmp"):
                    continue
                p = os.path.join(self.nvme_dir, f)
                if os.path.isfile(p):
                    entries.append((os.path.getmtime(p), p, os.path.getsize(p)))
            entries.sort()  # oldest first
            total = sum(size for _, _, size in entries)
            target = self.max_nvme_bytes
            # If adding a new file would exceed capacity, evict oldest until we fit
            while entries and total + incoming_bytes > target:
                _, p, size = entries.pop(0)
                try:
                    os.remove(p)
                    total -= size
                except Exception:
                    CACHE_ERRORS.inc()
            BYTES_STAGED.set(self._nvme_bytes())
        except Exception:
            CACHE_ERRORS.inc()

    def _is_lock_stale(self, lock_path: str) -> bool:
        try:
            if not os.path.exists(lock_path):
                return False
            mtime = os.path.getmtime(lock_path)
            return (time.time() - mtime) > self.stale_lock_s
        except Exception:
            CACHE_ERRORS.inc()
            return False

    def stage(self, url: str, blocksize: int = 8 * 1024 * 1024) -> str:
        """Stage remote URL to NVMe; returns local path."""
        local = self._nvme_path(url)
        if os.path.exists(local):
            return local

        lock_path, tmp_path = self._lock_paths(local)
        start = time.time()

        # Pre-check for stale lock and break it if safe
        if os.path.exists(lock_path) and self._is_lock_stale(lock_path):
            self._release_download_lock(lock_path)

        # Fast path: wait for another process/thread to complete download
        if not self._acquire_download_lock(lock_path):
            while time.time() - start < self.lock_timeout_s:
                # If the lock turns stale while waiting, break and try to acquire again
                if self._is_lock_stale(lock_path):
                    self._release_download_lock(lock_path)
                    if self._acquire_download_lock(lock_path):
                        break
                if os.path.exists(local) and not os.path.exists(lock_path):
                    return local
                time.sleep(self.lock_poll_s)
            # If we reach here and still can't acquire, timeout
            if not os.path.exists(local):
                CACHE_ERRORS.inc()
                raise TimeoutError(f"Timeout while waiting for lock {lock_path}")

        # We own the lock; perform the download
        try:
            # Ensure NVMe capacity if needed (approximate: we don't know size; evict conservatively)
            self._evict_nvme_if_needed()
            fs, path = fsspec.core.url_to_fs(url)

            # Best-effort cleanup before writing
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                CACHE_ERRORS.inc()

            # Retry streaming with exponential backoff
            last_err: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    with fs.open(path, "rb") as src, open(tmp_path, "wb") as dst:
                        while True:
                            buf = src.read(blocksize)
                            if not buf:
                                break
                            dst.write(buf)
                            CACHE_DOWNLOAD_BYTES.inc(len(buf))
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    # Cleanup partial tmp on failure
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        CACHE_ERRORS.inc()
                    if attempt + 1 >= self.max_retries:
                        break
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
            if last_err:
                raise last_err

            # Final capacity check with actual size
            size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            self._evict_nvme_if_needed(incoming_bytes=size)
            os.replace(tmp_path, local)
            BYTES_STAGED.set(self._nvme_bytes())
            return local
        except Exception:
            CACHE_ERRORS.inc()
            # Best-effort cleanup of tmp file on error
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                CACHE_ERRORS.inc()
            raise
        finally:
            self._release_download_lock(lock_path)

    def get(self, url: str) -> Tuple[bytes, str]:
        """Get bytes for URL, staging to NVMe and caching in RAM. Returns (data, local_path)."""
        key = self._key(url)
        v = self._ram_get(key)
        if v is not None:
            CACHE_HIT.inc()
            return v, self._nvme_path(url)
        CACHE_MISS.inc()
        local = self.stage(url)
        with open(local, "rb") as f:
            data = f.read()
        self._ram_put(key, data)
        return data, local

    def prefetch(self, urls: Iterable[str], max_workers: int = 4) -> List[str]:
        """Concurrently stage a list of URLs; returns list of local paths (completed only)."""
        results: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.stage, u): u for u in urls}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception:
                    CACHE_ERRORS.inc()
        return results

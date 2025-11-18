"""Light-weight adapter that exposes a simplified interface to the ZeRO-3 trainer."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .deepspeed_training import DeepSpeedConfig, DeepSpeedZeRO3Trainer
except ImportError:  # pragma: no cover - optional dependency
    DeepSpeedConfig = None  # type: ignore
    DeepSpeedZeRO3Trainer = None  # type: ignore

try:
    import deepspeed  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    deepspeed = None  # type: ignore

import torch
from torch.utils.data import DataLoader


class _FallbackDeepSpeedEngine:
    """Minimal DeepSpeed-like engine for CPU environments."""

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_engine = model.to(self.device)
        optimizer_params = config.get("optimizer", {}).get("params", {})
        lr = float(optimizer_params.get("lr", 1e-4))
        weight_decay = float(optimizer_params.get("weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(self.model_engine.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        self.model_engine.train()
        self.optimizer.zero_grad()
        outputs = self.model_engine(inputs)
        if outputs.dim() > 2:
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        value = float(loss.item())
        return value, {"loss": value}

    def save_sharded_checkpoint(self, step: int, checkpoint_dir: Path) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_rank0_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model_engine.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        return path

logger = logging.getLogger("DeepSpeedAdapter")


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_default_config(output_dir: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a ZeRO-3 configuration ready for orchestration use."""
    if DeepSpeedConfig is None:
        raise RuntimeError("DeepSpeed is not installed. Install deepspeed to enable ZeRO-3 training.")

    overrides = overrides or {}
    nvme_dir = os.path.join(output_dir, "deepspeed_nvme")
    base = DeepSpeedConfig.create_zero3_config(
        train_batch_size=overrides.get("train_batch_size", 32),
        enable_cpu_offload=overrides.get("enable_cpu_offload", True),
        enable_nvme_offload=overrides.get("enable_nvme_offload", True),
        nvme_offload_dir=overrides.get("nvme_offload_dir", nvme_dir),
        activation_checkpointing=overrides.get("activation_checkpointing", True),
    )
    if overrides:
        base = _deep_update(base, overrides)
    base.setdefault("checkpoint_dir", os.path.join(output_dir, "checkpoints", "deepspeed"))
    return base


class DeepSpeedTrainer:
    """Adapter that mirrors the AMP/DDP trainer interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        config: Dict[str, Any],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.dataloader = dataloader

        self._using_fallback = False
        if DeepSpeedZeRO3Trainer is None or deepspeed is None or not torch.cuda.is_available():
            self._using_fallback = True
            if DeepSpeedZeRO3Trainer is None or deepspeed is None:
                logger.warning("DeepSpeed runtime not available. Falling back to CPU implementation.")
            else:
                logger.warning("CUDA not detected. Falling back to CPU DeepSpeed simulation.")
            self.engine = _FallbackDeepSpeedEngine(model, config)
        else:
            self.engine = DeepSpeedZeRO3Trainer(model, config)

        if self.dataloader is None:
            if self._using_fallback:
                raise RuntimeError("DeepSpeed ZeRO-3 requires CUDA-enabled devices.")
            else:
                from .deepspeed_training import create_synthetic_dataset  # lazy import

                dataset = create_synthetic_dataset()
                self.dataloader = DataLoader(
                    dataset,
                    batch_size=config.get("train_micro_batch_size_per_gpu", 1),
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )

    def train_epoch(self, epoch: int, max_steps: Optional[int] = None) -> float:
        if self.dataloader is None:
            raise RuntimeError("DeepSpeed trainer requires a DataLoader instance.")

        running_loss = 0.0
        processed_steps = 0

        if hasattr(self.engine, "device"):
            device = self.engine.device
        elif hasattr(self.engine, "model_engine"):
            try:
                device = next(self.engine.model_engine.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        non_blocking = device.type == "cuda"

        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            loss, _ = self.engine.training_step(inputs, targets)
            running_loss += loss
            processed_steps += 1

            if max_steps and processed_steps >= max_steps:
                break

        avg_loss = running_loss / max(processed_steps, 1)
        logger.info(
            "DeepSpeed epoch %s completed on rank %s | steps=%s | loss=%.4f",
            epoch,
            self.rank,
            processed_steps,
            avg_loss,
        )
        return avg_loss

    def save_checkpoint(self, tag: str) -> None:
        step = int(tag.replace("ep", ""))
        checkpoint_root = self.engine.config.get("checkpoint_dir", "./checkpoints/deepspeed")  # type: ignore[attr-defined]
        checkpoint_dir = Path(checkpoint_root)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.engine.save_sharded_checkpoint(step=step, checkpoint_dir=checkpoint_dir)


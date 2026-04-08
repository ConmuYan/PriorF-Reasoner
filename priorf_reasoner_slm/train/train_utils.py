"""Shared training utilities for PriorF-Reasoner SLM training.

Provides:
- Device detection (cuda/cpu/mps)
- LoRA adapter save/load
- Logging helpers
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the best available device.

    Checks for CUDA, MPS (Apple Silicon), then falls back to CPU.

    Returns:
        A ``torch.device`` object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_adapter(model: PeftModel, output_path: str | Path) -> Path:
    """Save a PEFT LoRA adapter to disk.

    Args:
        model: A PEFT-wrapped model.
        output_path: Directory to save the adapter.

    Returns:
        Path to the saved adapter directory.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    logger.info("LoRA adapter saved to: %s", output_path)
    return output_path


def load_adapter(
    base_model: torch.nn.Module,
    adapter_path: str | Path,
    device: str | torch.device = "cpu",
) -> PeftModel:
    """Load a PEFT LoRA adapter onto a base model.

    Args:
        base_model: The base causal LM (without LoRA).
        adapter_path: Path to the saved LoRA adapter directory.
        device: Target device for the model.

    Returns:
        The base model with the LoRA adapter loaded.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        device_map={"": str(device)},
    )
    logger.info("LoRA adapter loaded from: %s", adapter_path)
    return model


def format_log(value: Any) -> str:
    """Format a metric value for logging.

    Handles floats, ints, tensors, and other types.

    Args:
        value: The value to format.

    Returns:
        A human-readable string.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, int):
        return str(value)
    elif hasattr(value, "item"):
        # Tensor or similar
        return format_log(value.item())
    else:
        return str(value)


def get_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient norm across all parameters.

    Args:
        model: The model whose gradients to measure.

    Returns:
        The total gradient norm as a float.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return float(total_norm)


def log_training_step(
    step: int,
    loss: float,
    lr: float,
    grad_norm: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a training step with structured output.

    Args:
        step: Current training step.
        loss: Loss value for this step.
        lr: Current learning rate.
        grad_norm: Optional gradient norm.
        extra: Optional dict of additional metrics to log.
    """
    parts = [f"step={step}", f"loss={format_log(loss)}", f"lr={format_log(lr)}"]
    if grad_norm is not None:
        parts.append(f"grad_norm={format_log(grad_norm)}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={format_log(v)}")
    logger.info(" | ".join(parts))


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int, float]:
    """Count trainable vs total parameters.

    Args:
        model: The model to analyze.

    Returns:
        Tuple of (trainable_count, total_count, trainable_pct).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct

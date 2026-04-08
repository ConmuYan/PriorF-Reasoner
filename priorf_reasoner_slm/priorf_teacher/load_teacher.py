"""Load a trained PriorF-GNN teacher model from checkpoint.

Supports both v1 (SCRE) and v2 (ASDA) model variants.
"""

from __future__ import annotations

import logging
import inspect
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_teacher_model(
    checkpoint_path: str | Path,
    model_cls: type[nn.Module],
    model_kwargs: dict | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a trained PriorF-GNN model from a checkpoint file.

    Args:
        checkpoint_path: Path to best_model.pt checkpoint.
        model_cls: Model class (LGHGCLNetV2 or LGHGCLNet).
        model_kwargs: kwargs for model constructor (in_dim, hidden dims, etc.).
        device: Target device.

    Returns:
        Loaded model in eval mode with weights from checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    model_kwargs = model_kwargs or {}
    signature = inspect.signature(model_cls.__init__)
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
    ignored_kwargs = sorted(set(model_kwargs) - set(filtered_kwargs))
    if ignored_kwargs:
        logger.info("Ignoring unsupported teacher kwargs for %s: %s", model_cls.__name__, ignored_kwargs)

    model = model_cls(**filtered_kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded teacher model from %s (epoch=%d)",
        checkpoint_path,
        ckpt.get("epoch", -1),
    )
    return model

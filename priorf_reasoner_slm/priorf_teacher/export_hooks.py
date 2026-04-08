"""Export hooks for extracting intermediate outputs from PriorF-GNN.

Registers forward hooks on MLP branch, GNN branch, and ASDA layer
to capture: mlp_logit, gnn_logit, asda_switch (node-level frequency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HookOutputs:
    """Intermediate outputs captured from teacher forward pass."""

    mlp_embedding: torch.Tensor | None = None  # z_mlp (N, out_hidden)
    gnn_embedding: torch.Tensor | None = None  # z_gnn (N, out_hidden)
    asda_switch: torch.Tensor | None = None  # alpha_bar (N, 1)


def _make_hook(storage: dict, key: str):
    """Create a forward hook that stores the output tensor."""

    def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        storage[key] = output.detach()

    return hook


def _make_asda_switch_hook(storage: dict):
    """Hook that captures the ASDA node-level switch (alpha_bar)."""

    def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        # In ASDALayer.forward, alpha_bar is computed from node_switch
        # We capture it by hooking the node_switch sub-module
        pass

    return hook


def register_export_hooks(model: nn.Module) -> tuple[list[torch.utils.hooks.RemovableHook], dict[str, torch.Tensor]]:
    """Register forward hooks on teacher model to capture intermediate outputs.

    Args:
        model: PriorF-GNN model (LGHGCLNetV2 or LGHGCLNet).

    Returns:
        Tuple of (hook_handles, storage_dict).
        Call remove_hooks(handles) when done.
    """
    handles: list[torch.utils.hooks.RemovableHook] = []
    storage: dict[str, torch.Tensor] = {}

    # Hook MLP branch
    if hasattr(model, "mlp"):
        h = model.mlp.register_forward_hook(_make_hook(storage, "mlp_embedding"))
        handles.append(h)
        logger.debug("Registered hook on model.mlp")

    # Hook GNN branch output (after 2nd RGCN + BN + ReLU)
    if hasattr(model, "rgcn2"):
        h = model.rgcn2.register_forward_hook(_make_hook(storage, "rgcn2_output"))
        handles.append(h)
        logger.debug("Registered hook on model.rgcn2")

    # Hook ASDA node switch to capture alpha_bar
    if hasattr(model, "asda") and hasattr(model.asda, "node_switch"):
        h = model.asda.node_switch.register_forward_hook(
            _make_hook(storage, "asda_switch")
        )
        handles.append(h)
        logger.debug("Registered hook on model.asda.node_switch")

    return handles, storage


def remove_hooks(handles: list[torch.utils.hooks.RemovableHook]) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()


def extract_branch_logits(
    model: nn.Module,
    mlp_embedding: torch.Tensor | None,
    gnn_embedding: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    """Compute per-branch pseudo-logits from embeddings.

    Since the teacher model fuses branches before the prediction head,
    we approximate branch-level scores by applying the prediction head
    to each branch independently (zero-padding the other).

    Args:
        model: Teacher model with `out` Linear layer.
        mlp_embedding: MLP branch output (N, out_hidden) or None.
        gnn_embedding: GNN branch output (N, out_hidden) or None.

    Returns:
        Dict with mlp_logit, gnn_logit (N,) tensors.
    """
    out_layer = model.out  # nn.Linear(head_dim, 1)
    head_dim = out_layer.in_features

    # Validate fusion layout assumption: branches are concatenated in [mlp, gnn] order
    if mlp_embedding is not None and gnn_embedding is not None:
        expected = mlp_embedding.size(-1) + gnn_embedding.size(-1)
        assert head_dim == expected, (
            f"Fusion layout mismatch: out.in_features={head_dim}, "
            f"but mlp_dim({mlp_embedding.size(-1)}) + gnn_dim({gnn_embedding.size(-1)}) = {expected}"
        )

    device = next(model.parameters()).device

    mlp_logit = None
    gnn_logit = None

    if mlp_embedding is not None:
        mlp_dim = mlp_embedding.size(-1)
        if hasattr(model, "use_gnn") and model.use_gnn:
            # Zero-pad GNN part
            dummy_gnn = torch.zeros(
                mlp_embedding.size(0), head_dim - mlp_dim,
                device=device, dtype=mlp_embedding.dtype,
            )
            mlp_input = torch.cat([mlp_embedding, dummy_gnn], dim=-1)
        else:
            mlp_input = mlp_embedding
        mlp_logit = out_layer(mlp_input).squeeze(-1)

    if gnn_embedding is not None:
        gnn_dim = gnn_embedding.size(-1)
        if hasattr(model, "use_mlp") and model.use_mlp:
            # Zero-pad MLP part
            dummy_mlp = torch.zeros(
                gnn_embedding.size(0), head_dim - gnn_dim,
                device=device, dtype=gnn_embedding.dtype,
            )
            gnn_input = torch.cat([dummy_mlp, gnn_embedding], dim=-1)
        else:
            gnn_input = gnn_embedding
        gnn_logit = out_layer(gnn_input).squeeze(-1)

    return {"mlp_logit": mlp_logit, "gnn_logit": gnn_logit}

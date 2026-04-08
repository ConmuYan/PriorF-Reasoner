"""Model loading and hidden-state extraction for the Qwen3-4B student.

Wraps AutoModelForCausalLM with optional LoRA (PEFT), and provides
utilities for extracting hidden states used by the classification head.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)

# Default LoRA configuration for the student model
DEFAULT_LORA_CONFIG: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    # Qwen3-4B-Instruct-2507 exposes standard attention projection modules.
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


def load_student_model(
    model_name: str = "Qwen/Qwen3-4B",
    lora_config: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> PreTrainedModel:
    """Load a Qwen3-4B student model with optional LoRA adapters.

    By default, applies LoRA (r=16, alpha=32) targeting the attention
    projection layers. Pass ``lora_config={}`` to skip LoRA entirely
    and load the full model.

    Args:
        model_name: HuggingFace model identifier or local path.
        lora_config: LoRA configuration dict. Pass None for defaults,
            or an empty dict ``{}`` to disable LoRA.
        device: Target device string (``"cpu"``, ``"cuda"``, etc.).
        load_in_4bit: Load model in 4-bit quantization (bitsandbytes).
        load_in_8bit: Load model in 8-bit quantization (bitsandbytes).

    Returns:
        A PreTrainedModel (wrapped with PEFT if LoRA is enabled).
    """
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    elif load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
    else:
        device_str = str(device)
        if device_str.startswith("cuda"):
            kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            kwargs["torch_dtype"] = torch.float32

    logger.info("Loading student model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if not load_in_4bit and not load_in_8bit:
        model = model.to(device)

    # Determine whether to apply LoRA
    apply_lora = True
    if lora_config is not None and len(lora_config) == 0:
        # Empty dict explicitly disables LoRA
        apply_lora = False

    if apply_lora:
        from peft import LoraConfig, get_peft_model

        effective_config = DEFAULT_LORA_CONFIG.copy()
        if lora_config is not None:
            effective_config.update(lora_config)

        peft_config = LoraConfig(**effective_config)
        model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Applied LoRA: trainable=%d / total=%d (%.2f%%)",
            trainable,
            total,
            100.0 * trainable / total,
        )

    model.eval()
    return model


def get_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract last-layer hidden states from the model.

    Runs a forward pass with ``output_hidden_states=True`` and returns
    the hidden states of the final layer.

    Args:
        model: The causal LM model.
        input_ids: Token IDs of shape ``(batch, seq_len)``.
        attention_mask: Optional attention mask of shape ``(batch, seq_len)``.

    Returns:
        Hidden states tensor of shape ``(batch, seq_len, hidden_dim)``.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # Last layer hidden states
    hidden = outputs.hidden_states[-1]
    return hidden


def get_last_token_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract the hidden state of the last non-padding token per sample.

    For each item in the batch, selects the hidden state at the position
    of the last ``1`` in the attention mask. This is the standard
    approach for classification with causal LMs.

    Args:
        hidden_states: Full hidden states ``(batch, seq_len, hidden_dim)``.
        attention_mask: Binary mask ``(batch, seq_len)`` where 1 = valid.

    Returns:
        Last-token hidden states of shape ``(batch, hidden_dim)``.
    """
    # Find the index of the last non-padding token for each sample
    # attention_mask: (batch, seq_len)
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)

    # Gather the hidden state at the last valid position
    last_hidden = hidden_states[batch_idx, seq_lengths]  # (batch, hidden_dim)
    return last_hidden

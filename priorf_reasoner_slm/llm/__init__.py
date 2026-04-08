"""llm: LLM wrapper, classification head, losses, and generation.

Public API for the PriorF-Reasoner student LLM module (Block 4).
"""

from priorf_reasoner_slm.llm.tokenizer_utils import (
    load_tokenizer,
    format_evidence_card,
    apply_chat_template,
)
from priorf_reasoner_slm.llm.model_wrapper import (
    load_student_model,
    get_hidden_states,
    get_last_token_hidden,
    DEFAULT_LORA_CONFIG,
)
from priorf_reasoner_slm.llm.cls_head import BinaryClsHead
from priorf_reasoner_slm.llm.losses import (
    gen_loss,
    cls_loss,
    distill_loss_kl,
    distill_loss_mse,
    PriorFLoss,
)
from priorf_reasoner_slm.llm.collators import (
    SFTDataCollator,
    CoTrainDataCollator,
)
from priorf_reasoner_slm.llm.generation import (
    generate_prediction,
    parse_prediction_output,
    batch_generate,
)
from priorf_reasoner_slm.llm.fusion import (
    fuse_predictions,
    optimize_alpha,
)

__all__ = [
    # Tokenizer
    "load_tokenizer",
    "format_evidence_card",
    "apply_chat_template",
    # Model
    "load_student_model",
    "get_hidden_states",
    "get_last_token_hidden",
    "DEFAULT_LORA_CONFIG",
    # Classification head
    "BinaryClsHead",
    # Losses
    "gen_loss",
    "cls_loss",
    "distill_loss_kl",
    "distill_loss_mse",
    "PriorFLoss",
    # Collators
    "SFTDataCollator",
    "CoTrainDataCollator",
    # Generation
    "generate_prediction",
    "parse_prediction_output",
    "batch_generate",
    # Fusion
    "fuse_predictions",
    "optimize_alpha",
]

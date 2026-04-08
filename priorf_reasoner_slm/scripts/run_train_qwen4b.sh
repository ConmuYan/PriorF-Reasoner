#!/bin/bash
# Full training script for PriorF-Reasoner with Qwen3-4B
set -e

# ============================================================
# PriorF-Reasoner Full Training Pipeline
# Stage 1: SFT with LoRA
# Stage 2: Co-training (gen + cls + distill)
# ============================================================

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-outputs/sft}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-outputs/cotrain}"
TEACHER_CSV="${TEACHER_CSV:-assets/teacher_exports/train_evidence.parquet}"
DATASET_FORMAT="${DATASET_FORMAT:-prompt_completion}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Training hyperparameters
SFT_LR="${SFT_LR:-1e-4}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_BATCH="${SFT_BATCH:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"

COTRAIN_LR="${COTRAIN_LR:-5e-5}"
COTRAIN_EPOCHS="${COTRAIN_EPOCHS:-3}"
COTRAIN_BATCH="${COTRAIN_BATCH:-4}"
COTRAIN_GRAD_ACCUM="${COTRAIN_GRAD_ACCUM:-4}"
LAMBDA_CLS="${LAMBDA_CLS:-0.1}"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.1}"
DISTILL_TEMP="${DISTILL_TEMP:-2.0}"

# ============================================================
echo "===== PriorF-Reasoner Full Training ====="
echo "Model: $MODEL_NAME"
echo "Teacher CSV: $TEACHER_CSV"
echo "Stage 1 output: $STAGE1_OUTPUT"
echo "Stage 2 output: $STAGE2_OUTPUT"
echo ""

# ============================================================
# Stage 1: SFT Training
# ============================================================
echo "===== Stage 1: SFT Training ====="
echo "Learning rate: $SFT_LR"
echo "Epochs: $SFT_EPOCHS"
echo "Batch size: $SFT_BATCH (grad_accum=$SFT_GRAD_ACCUM)"

python - <<'PY'
import logging
import os
import sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from pathlib import Path
import pandas as pd
from priorf_reasoner_slm.train.train_sft import train_sft
from priorf_reasoner_slm.train.train_utils import get_device

teacher_csv = os.getenv("TEACHER_CSV", "assets/teacher_exports/train_evidence.parquet")
stage1_output = os.getenv("STAGE1_OUTPUT", "outputs/sft")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
dataset_format = os.getenv("DATASET_FORMAT", "prompt_completion")
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
lora_r = int(os.getenv("LORA_R", "16"))
lora_alpha = int(os.getenv("LORA_ALPHA", "32"))
sft_lr = float(os.getenv("SFT_LR", "1e-4"))
sft_epochs = int(os.getenv("SFT_EPOCHS", "3"))
sft_batch = int(os.getenv("SFT_BATCH", "4"))
sft_grad_accum = int(os.getenv("SFT_GRAD_ACCUM", "4"))

print(f"Device: {get_device()}")
print(f"Teacher CSV: {Path(teacher_csv).resolve()}")

if not Path(teacher_csv).exists():
    print(f"ERROR: Teacher CSV not found: {teacher_csv}")
    sys.exit(1)

teacher_path = Path(teacher_csv)
if teacher_path.suffix == '.parquet':
    df = pd.read_parquet(teacher_path)
else:
    df = pd.read_csv(teacher_path)
print(f"Loaded {len(df)} teacher samples")

adapter_path = train_sft(
    teacher_df=df,
    output_dir=stage1_output,
    model_name=model_name,
    dataset_format=dataset_format,
    max_seq_length=max_seq_length,
    lora_config={
        'r': lora_r,
        'lora_alpha': lora_alpha,
    },
    training_args={
        'learning_rate': sft_lr,
        'num_train_epochs': sft_epochs,
        'per_device_train_batch_size': sft_batch,
        'gradient_accumulation_steps': sft_grad_accum,
    },
)
print(f"Stage 1 complete. Adapter: {adapter_path}")
PY

echo ""
echo "Stage 1 SFT training complete!"
echo ""

# ============================================================
# Stage 2: Co-training
# ============================================================
echo "===== Stage 2: Co-Training ====="
echo "Learning rate: $COTRAIN_LR"
echo "Epochs: $COTRAIN_EPOCHS"
echo "Batch size: $COTRAIN_BATCH (grad_accum=$COTRAIN_GRAD_ACCUM)"
echo "lambda_cls=$LAMBDA_CLS, lambda_distill=$LAMBDA_DISTILL, distill_temp=$DISTILL_TEMP"

python - <<'PY'
import logging
import os
import sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from pathlib import Path
import pandas as pd
from priorf_reasoner_slm.train.train_cotrain import train_cotrain

teacher_csv = os.getenv("TEACHER_CSV", "assets/teacher_exports/train_evidence.parquet")
stage1_output = os.getenv("STAGE1_OUTPUT", "outputs/sft")
stage2_output = os.getenv("STAGE2_OUTPUT", "outputs/cotrain")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
lambda_cls = float(os.getenv("LAMBDA_CLS", "0.1"))
lambda_distill = float(os.getenv("LAMBDA_DISTILL", "0.1"))
distill_temp = float(os.getenv("DISTILL_TEMP", "2.0"))
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
cotrain_batch = int(os.getenv("COTRAIN_BATCH", "4"))
cotrain_grad_accum = int(os.getenv("COTRAIN_GRAD_ACCUM", "4"))
cotrain_epochs = int(os.getenv("COTRAIN_EPOCHS", "3"))
cotrain_lr = float(os.getenv("COTRAIN_LR", "5e-5"))

adapter_path = Path(stage1_output) / 'final_adapter'
if not adapter_path.exists():
    print(f"ERROR: Stage 1 adapter not found at {adapter_path}")
    sys.exit(1)

teacher_path = Path(teacher_csv)
if teacher_path.suffix == '.parquet':
    df = pd.read_parquet(teacher_path)
else:
    df = pd.read_csv(teacher_path)
print(f"Loaded {len(df)} teacher samples")

checkpoint_path = train_cotrain(
    teacher_df=df,
    adapter_path=str(adapter_path),
    output_dir=stage2_output,
    model_name=model_name,
    lambda_cls=lambda_cls,
    lambda_distill=lambda_distill,
    distill_temp=distill_temp,
    max_seq_length=max_seq_length,
    per_device_batch_size=cotrain_batch,
    gradient_accumulation_steps=cotrain_grad_accum,
    num_train_epochs=cotrain_epochs,
    learning_rate=cotrain_lr,
)
print(f"Stage 2 complete. Checkpoint: {checkpoint_path}")
PY

echo ""
echo "===== Full training pipeline complete ====="
echo "Stage 1: $STAGE1_OUTPUT"
echo "Stage 2: $STAGE2_OUTPUT"

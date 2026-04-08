#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data1/mq/conda_envs/priorfgnn}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-/data1/mq/models/Qwen3-4B-Instruct-2507}"
DATASETS="${DATASETS:-amazon,yelpchi}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

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

GEN_SAMPLE_FRACTION="${GEN_SAMPLE_FRACTION:-0.1}"
HEAD_SAMPLE_FRACTION="${HEAD_SAMPLE_FRACTION:-1.0}"
FAITH_SAMPLE_FRACTION="${FAITH_SAMPLE_FRACTION:-0.1}"

echo "===== PriorF-Reasoner Full Pipeline (per-dataset) ====="
echo "Model: $MODEL_NAME"
echo "Datasets: $DATASETS"
echo "GPU: $GPU_ID"

# ---- Stage 0: Smoke Checks (shared, run once) ----
RUN_NAME="${RUN_NAME:-formal_$(date +%Y%m%d_%H%M%S)}"
LOG_BASE="outputs/pipeline_runs/$RUN_NAME"
mkdir -p "$LOG_BASE"

echo ""
echo "===== Stage 0: Smoke Checks ====="
bash priorf_reasoner_slm/scripts/run_smoke.sh | tee "$LOG_BASE/smoke.log"

# ---- Per-dataset loop ----
IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

for DATASET in "${DATASET_LIST[@]}"; do
    DATASET=$(echo "$DATASET" | xargs)  # trim whitespace
    echo ""
    echo "=========================================="
    echo "  Dataset: $DATASET"
    echo "=========================================="

    TRAIN_EVIDENCE="assets/teacher_exports/${DATASET}_train_evidence.parquet"
    TEST_EVIDENCE="assets/teacher_exports/${DATASET}_test_evidence.parquet"

    if [ ! -f "$TRAIN_EVIDENCE" ]; then
        echo "ERROR: Missing train evidence: $TRAIN_EVIDENCE"
        exit 1
    fi
    if [ ! -f "$TEST_EVIDENCE" ]; then
        echo "ERROR: Missing test evidence: $TEST_EVIDENCE"
        exit 1
    fi

    RUN_DIR="${LOG_BASE}/${DATASET}"
    STAGE1_OUTPUT="${RUN_DIR}/sft"
    STAGE2_OUTPUT="${RUN_DIR}/cotrain"
    EVAL_OUTPUT="${RUN_DIR}/eval"
    mkdir -p "$RUN_DIR" "$EVAL_OUTPUT"

    # ---- Stage 1+2: Training ----
    echo ""
    echo "===== Training: $DATASET ====="
    export MODEL_NAME
    export TEACHER_CSV="$TRAIN_EVIDENCE"
    export STAGE1_OUTPUT
    export STAGE2_OUTPUT
    export MAX_SEQ_LENGTH
    export LORA_R
    export LORA_ALPHA
    export SFT_LR
    export SFT_EPOCHS
    export SFT_BATCH
    export SFT_GRAD_ACCUM
    export COTRAIN_LR
    export COTRAIN_EPOCHS
    export COTRAIN_BATCH
    export COTRAIN_GRAD_ACCUM
    export LAMBDA_CLS
    export LAMBDA_DISTILL
    export DISTILL_TEMP

    bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh | tee "$RUN_DIR/train.log"

    # ---- Stage 3: Evaluation ----
    echo ""
    echo "===== Evaluation: $DATASET ====="
    export ADAPTER_PATH="$STAGE2_OUTPUT/final_cotrain/adapter"
    export CLS_HEAD_PATH="$STAGE2_OUTPUT/final_cotrain/cls_head.pt"
    export EVIDENCE_CSV="$TEST_EVIDENCE"
    export EVAL_OUTPUT
    export GEN_SAMPLE_FRACTION
    export HEAD_SAMPLE_FRACTION
    export FAITH_SAMPLE_FRACTION

    bash priorf_reasoner_slm/scripts/run_eval.sh | tee "$RUN_DIR/eval.log"

    echo ""
    echo "===== Dataset $DATASET complete ====="
done

echo ""
echo "===== Full Pipeline Complete ====="
echo "Results in: $LOG_BASE"
echo ""
echo "Per-dataset evaluation results:"
for DATASET in "${DATASET_LIST[@]}"; do
    DATASET=$(echo "$DATASET" | xargs)
    echo "  $DATASET:"
    echo "    gen_eval:    $LOG_BASE/$DATASET/eval/gen_eval.json"
    echo "    head_eval:   $LOG_BASE/$DATASET/eval/head_eval.json"
    echo "    faithfulness: $LOG_BASE/$DATASET/eval/faithfulness.json"
done

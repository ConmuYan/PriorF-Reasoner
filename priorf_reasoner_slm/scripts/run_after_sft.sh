#!/bin/bash
# Continue pipeline after SFT: CoTrain + Eval for a single dataset
# Usage: DATASET=amazon bash priorf_reasoner_slm/scripts/run_after_sft.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PATH="/data1/mq/conda_envs/priorfgnn/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="${DATASET:-amazon}"
MODEL_NAME="${MODEL_NAME:-/data1/mq/models/Qwen3-4B-Instruct-2507}"
RUN_DIR="outputs/formal_${DATASET}"

TRAIN_EVIDENCE="assets/teacher_exports/${DATASET}_train_evidence.parquet"
TEST_EVIDENCE="assets/teacher_exports/${DATASET}_test_evidence.parquet"
STAGE1_OUTPUT="${RUN_DIR}/sft"
STAGE2_OUTPUT="${RUN_DIR}/cotrain"
EVAL_OUTPUT="${RUN_DIR}/eval"

echo "===== Continue Pipeline: $DATASET ====="
echo "SFT adapter: $STAGE1_OUTPUT/final_adapter"
echo "CoTrain output: $STAGE2_OUTPUT"
echo "Eval output: $EVAL_OUTPUT"

# Verify SFT adapter exists
if [ ! -d "$STAGE1_OUTPUT/final_adapter" ]; then
    echo "ERROR: SFT adapter not found at $STAGE1_OUTPUT/final_adapter"
    echo "Run SFT first with run_train_qwen4b.sh"
    exit 1
fi

# ---- Stage 2: Co-training ----
echo ""
echo "===== Stage 2: Co-Training ====="
export MODEL_NAME
export TEACHER_CSV="$TRAIN_EVIDENCE"
export STAGE1_OUTPUT
export STAGE2_OUTPUT
export MAX_SEQ_LENGTH=2048
export LORA_R=16
export LORA_ALPHA=32
export COTRAIN_LR=5e-5
export COTRAIN_EPOCHS=3
export COTRAIN_BATCH=4
export COTRAIN_GRAD_ACCUM=4
export LAMBDA_CLS=0.1
export LAMBDA_DISTILL=0.1
export DISTILL_TEMP=2.0

bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh 2>&1 | tee "$RUN_DIR/cotrain.log"

# ---- Stage 3: Evaluation ----
echo ""
echo "===== Stage 3: Evaluation ====="
export ADAPTER_PATH="$STAGE2_OUTPUT/final_cotrain/adapter"
export CLS_HEAD_PATH="$STAGE2_OUTPUT/final_cotrain/cls_head.pt"
export EVIDENCE_CSV="$TEST_EVIDENCE"
export EVAL_OUTPUT
export GEN_SAMPLE_FRACTION=0.1
export HEAD_SAMPLE_FRACTION=1.0
export FAITH_SAMPLE_FRACTION=0.1

bash priorf_reasoner_slm/scripts/run_eval.sh 2>&1 | tee "$RUN_DIR/eval.log"

echo ""
echo "===== Pipeline Complete: $DATASET ====="
echo "Results:"
for f in gen_eval.json head_eval.json faithfulness.json; do
    if [ -f "$EVAL_OUTPUT/$f" ]; then
        echo ""
        echo "--- $f ---"
        python -c "import json; print(json.dumps(json.load(open('$EVAL_OUTPUT/$f')), indent=2))"
    fi
done

# PriorF-Reasoner Commands

## Single-GPU Training (RTX 3090)

Prerequisites:

- Teacher evidence already exported to `assets/teacher_exports/`
- Base model downloaded locally: `/data1/mq/models/Qwen3-4B-Instruct-2507`
- Conda env: `/data1/mq/conda_envs/priorfgnn`

### 1. Full pipeline (per-dataset)

```bash
cd /data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner

# Both datasets
GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh

# Single dataset
DATASETS=amazon GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh
```

This runs: smoke test -> SFT -> CoTrain -> eval, independently per dataset.

### 2. Training only

```bash
export MODEL_NAME=/data1/mq/models/Qwen3-4B-Instruct-2507
export TEACHER_CSV=assets/teacher_exports/amazon_train_evidence.parquet
export STAGE1_OUTPUT=outputs/formal_amazon/sft
export STAGE2_OUTPUT=outputs/formal_amazon/cotrain
export MAX_SEQ_LENGTH=2048
export SFT_BATCH=4
export SFT_GRAD_ACCUM=4
export SFT_EPOCHS=3
export COTRAIN_BATCH=4
export COTRAIN_GRAD_ACCUM=4
export COTRAIN_EPOCHS=3
export SFT_LR=1e-4
export COTRAIN_LR=5e-5
export LAMBDA_CLS=0.1
export LAMBDA_DISTILL=0.1
export DISTILL_TEMP=2.0

bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh
```

### 3. Evaluation only

```bash
export ADAPTER_PATH=outputs/formal_amazon/cotrain/final_cotrain/adapter
export CLS_HEAD_PATH=outputs/formal_amazon/cotrain/final_cotrain/cls_head.pt
export EVIDENCE_CSV=assets/teacher_exports/amazon_test_evidence.parquet
export EVAL_OUTPUT=outputs/formal_amazon/eval

bash priorf_reasoner_slm/scripts/run_eval.sh
```

### 4. Continue after SFT (CoTrain + eval)

```bash
DATASET=amazon bash priorf_reasoner_slm/scripts/run_after_sft.sh
```

### Monitor training

```bash
# TensorBoard for SFT
tensorboard --logdir outputs/formal_amazon/sft/runs --port 6006

# GPU usage
watch -n 5 nvidia-smi
```

## Quick smoke check

```bash
bash priorf_reasoner_slm/scripts/run_smoke.sh
```

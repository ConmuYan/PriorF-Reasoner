# PriorF-Reasoner Commands

## 3090 Single-GPU Full Training

Prerequisites:

- Teacher assets are linked via `priorf_reasoner_slm/scripts/setup_assets.sh`
- Base model is downloaded locally, for example:
  - `/data1/mq/models/Qwen3-4B-Instruct-2507`
- CUDA / PyTorch are available in `priorfgnn`

### 1. Export train evidence

```bash
cd /data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner

DATASET=amazon SPLIT=train bash priorf_reasoner_slm/scripts/export_teacher_evidence.sh
DATASET=yelpchi SPLIT=train bash priorf_reasoner_slm/scripts/export_teacher_evidence.sh
```

This produces:

- `assets/teacher_exports/amazon_train_evidence.parquet`
- `assets/teacher_exports/yelpchi_train_evidence.parquet`

### 2. Merge train evidence into one file

`run_train_qwen4b.sh` expects a single training file, so merge the two train exports:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner")
amazon = pd.read_parquet(root / "assets/teacher_exports/amazon_train_evidence.parquet")
yelpchi = pd.read_parquet(root / "assets/teacher_exports/yelpchi_train_evidence.parquet")
train = pd.concat([amazon, yelpchi], ignore_index=True)
out = root / "assets/teacher_exports/train_evidence.parquet"
train.to_parquet(out, index=False)
print(f"saved {len(train)} rows -> {out}")
PY
```

### 3. Train the student LLM

For a single RTX 3090, keep the batch size conservative:

```bash
cd /data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner

export MODEL_NAME=/data1/mq/models/Qwen3-4B-Instruct-2507
export TEACHER_CSV=assets/teacher_exports/train_evidence.parquet
export STAGE1_OUTPUT=outputs/sft_qwen3_4b
export STAGE2_OUTPUT=outputs/cotrain_qwen3_4b
export MAX_SEQ_LENGTH=2048
export SFT_BATCH=1
export SFT_GRAD_ACCUM=8
export SFT_EPOCHS=3
export COTRAIN_BATCH=1
export COTRAIN_GRAD_ACCUM=8
export COTRAIN_EPOCHS=3
export SFT_LR=1e-4
export COTRAIN_LR=5e-5
export LAMBDA_CLS=0.1
export LAMBDA_DISTILL=0.1
export DISTILL_TEMP=2.0

bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh
```

### 4. Evaluation

After training, run:

```bash
cd /data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner

export ADAPTER_PATH=outputs/cotrain_qwen3_4b/final_cotrain/adapter
export CLS_HEAD_PATH=outputs/cotrain_qwen3_4b/final_cotrain/cls_head.pt
export EVIDENCE_CSV=assets/teacher_exports/test_evidence.parquet
export EVAL_OUTPUT=outputs/eval_qwen3_4b

bash priorf_reasoner_slm/scripts/run_eval.sh
```

## Quick smoke check

```bash
cd /data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner
bash priorf_reasoner_slm/scripts/run_smoke.sh
```

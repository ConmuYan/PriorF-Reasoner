#!/bin/bash
# Evaluation script for PriorF-Reasoner
set -e

echo "===== PriorF-Reasoner Evaluation ====="

# Configuration
ADAPTER_PATH="${ADAPTER_PATH:-outputs/cotrain/final_cotrain/adapter}"
CLS_HEAD_PATH="${CLS_HEAD_PATH:-outputs/cotrain/final_cotrain/cls_head.pt}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
EVIDENCE_CSV="${EVIDENCE_CSV:-assets/teacher_exports/test_evidence.parquet}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/eval/results}"
GEN_SAMPLE_FRACTION="${GEN_SAMPLE_FRACTION:-0.1}"
HEAD_SAMPLE_FRACTION="${HEAD_SAMPLE_FRACTION:-0.1}"
FAITH_SAMPLE_FRACTION="${FAITH_SAMPLE_FRACTION:-0.05}"

mkdir -p "$EVAL_OUTPUT"

echo "Adapter: $ADAPTER_PATH"
echo "CLS Head: $CLS_HEAD_PATH"
echo "Evidence CSV: $EVIDENCE_CSV"
echo "Output: $EVAL_OUTPUT"
echo "Gen sample fraction: $GEN_SAMPLE_FRACTION"
echo "Head sample fraction: $HEAD_SAMPLE_FRACTION"
echo "Faithfulness sample fraction: $FAITH_SAMPLE_FRACTION"

# Check files
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$EVIDENCE_CSV" ]; then
    echo "ERROR: Evidence CSV not found at $EVIDENCE_CSV"
    exit 1
fi

# ============================================================
# 1. Generation Quality Evaluation
# ============================================================
echo ""
echo "===== [1/4] Generation Quality Evaluation ====="

python - <<'PY'
import logging
import os
import sys
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from priorf_reasoner_slm.eval.eval_gen_only import evaluate_gen
from priorf_reasoner_slm.llm.model_wrapper import load_student_model
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer
from priorf_reasoner_slm.train.train_utils import get_device, load_adapter

adapter_path = Path(os.getenv("ADAPTER_PATH", "outputs/cotrain/final_cotrain/adapter"))
evidence_csv = Path(os.getenv("EVIDENCE_CSV", "assets/teacher_exports/test_evidence.parquet"))
eval_output = Path(os.getenv("EVAL_OUTPUT", "outputs/eval/results"))
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
gen_sample_fraction = float(os.getenv("GEN_SAMPLE_FRACTION", "0.1"))

device = get_device()
model = load_student_model(model_name=model_name, device=device)
model = load_adapter(model, adapter_path, device=str(device))
tokenizer = load_tokenizer(model_name)

evidence_path = evidence_csv
if evidence_path.suffix == '.parquet':
    df = pd.read_parquet(evidence_path)
else:
    df = pd.read_csv(evidence_path)
print(f"Loaded {len(df)} test samples")

results = evaluate_gen(df, model, tokenizer, sample_fraction=gen_sample_fraction)
print("\nGeneration Quality Results:")
for k, v in results.items():
    print(f"  {k}: {v}")

import json
with open(eval_output / 'gen_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {eval_output / 'gen_eval.json'}")
PY

# ============================================================
# 2. Classification Head Evaluation
# ============================================================
echo ""
echo "===== [2/4] Classification Head Evaluation ====="

if [ -f "$CLS_HEAD_PATH" ]; then
    python - <<'PY'
import logging
import os
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from priorf_reasoner_slm.eval.eval_head_only import evaluate_head_from_checkpoint
from priorf_reasoner_slm.train.train_utils import get_device, load_adapter
from priorf_reasoner_slm.llm.model_wrapper import load_student_model

adapter_path = Path(os.getenv("ADAPTER_PATH", "outputs/cotrain/final_cotrain/adapter"))
cls_head_path = os.getenv("CLS_HEAD_PATH", "outputs/cotrain/final_cotrain/cls_head.pt")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
evidence_csv = Path(os.getenv("EVIDENCE_CSV", "assets/teacher_exports/test_evidence.parquet"))
eval_output = Path(os.getenv("EVAL_OUTPUT", "outputs/eval/results"))
head_sample_fraction = float(os.getenv("HEAD_SAMPLE_FRACTION", "0.1"))

device = get_device()
base_model = load_student_model(model_name=model_name, device=device)
model = load_adapter(base_model, adapter_path, device=str(device))

evidence_path = evidence_csv
if evidence_path.suffix == '.parquet':
    df = pd.read_parquet(evidence_path)
else:
    df = pd.read_csv(evidence_path)
print(f"Loaded {len(df)} test samples")

results = evaluate_head_from_checkpoint(
    df, str(adapter_path), cls_head_path,
    model_name=model_name, sample_fraction=head_sample_fraction
)
print("\nHead Evaluation Results:")
for k, v in results.items():
    print(f"  {k}: {v}")

import json
with open(eval_output / 'head_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {eval_output / 'head_eval.json'}")
PY
else
    echo "SKIP: cls_head.pt not found at $CLS_HEAD_PATH"
fi

# ============================================================
# 3. Fusion Evaluation
# ============================================================
echo ""
echo "===== [3/4] Fusion Evaluation ====="

python - <<'PY'
import logging
import os
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from priorf_reasoner_slm.eval.eval_fusion import evaluate_fusion_with_optimization

evidence_path = Path(os.getenv("EVIDENCE_CSV", "assets/teacher_exports/test_evidence.parquet"))
eval_output = Path(os.getenv("EVAL_OUTPUT", "outputs/eval/results"))
if evidence_path.suffix == '.parquet':
    df = pd.read_parquet(evidence_path)
else:
    df = pd.read_csv(evidence_path)
print(f"Loaded {len(df)} test samples")

if 'cls_prob' not in df.columns or 'teacher_prob' not in df.columns:
    print("SKIP: CSV missing cls_prob or teacher_prob columns for fusion eval")
else:
    results = evaluate_fusion_with_optimization(df, df['cls_prob'], df['teacher_prob'])
    print("\nFusion Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    import json
    with open(eval_output / 'fusion_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {eval_output / 'fusion_eval.json'}")
PY

# ============================================================
# 4. Faithfulness Evaluation
# ============================================================
echo ""
echo "===== [4/4] Faithfulness Evaluation ====="

if [ -f "$CLS_HEAD_PATH" ]; then
    python - <<'PY'
import logging
import os
import pandas as pd
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from priorf_reasoner_slm.eval.faithfulness import (
    evaluate_sufficiency,
    evaluate_comprehensiveness,
)
from priorf_reasoner_slm.train.train_utils import get_device, load_adapter
from priorf_reasoner_slm.llm.model_wrapper import load_student_model
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer

adapter_path = Path(os.getenv("ADAPTER_PATH", "outputs/cotrain/final_cotrain/adapter"))
cls_head_path = os.getenv("CLS_HEAD_PATH", "outputs/cotrain/final_cotrain/cls_head.pt")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
evidence_csv = Path(os.getenv("EVIDENCE_CSV", "assets/teacher_exports/test_evidence.parquet"))
eval_output = Path(os.getenv("EVAL_OUTPUT", "outputs/eval/results"))
faith_sample_fraction = float(os.getenv("FAITH_SAMPLE_FRACTION", "0.05"))

device = get_device()
base_model = load_student_model(model_name=model_name, device=device)
model = load_adapter(base_model, adapter_path, device=str(device))
tokenizer = load_tokenizer(model_name)

evidence_path = evidence_csv
if evidence_path.suffix == '.parquet':
    df = pd.read_parquet(evidence_path)
else:
    df = pd.read_csv(evidence_path)
print(f"Loaded {len(df)} test samples")

# Sufficiency
print("\n--- Sufficiency ---")
suff_results = evaluate_sufficiency(df, model, tokenizer, sample_fraction=faith_sample_fraction)
print(f"Sufficiency: {suff_results['sufficiency']:.4f}")

# Comprehensiveness
print("\n--- Comprehensiveness ---")
cls_head_weights = torch.load(cls_head_path, map_location=device)
comp_results = evaluate_comprehensiveness(
    df, model, cls_head_weights, tokenizer, sample_fraction=faith_sample_fraction
)
print(f"Comprehensiveness: {comp_results['comprehensiveness']:.4f}")

import json
faithfulness_results = {
    "sufficiency": suff_results["sufficiency"],
    "comprehensiveness": comp_results["comprehensiveness"],
    "num_samples": suff_results["num_samples"],
}
with open(eval_output / 'faithfulness.json', 'w') as f:
    json.dump(faithfulness_results, f, indent=2)
print(f"Saved to {eval_output / 'faithfulness.json'}")
PY
else
    echo "SKIP: cls_head.pt not found"
fi

echo ""
echo "===== Evaluation complete ====="
echo "Results saved to: $EVAL_OUTPUT"

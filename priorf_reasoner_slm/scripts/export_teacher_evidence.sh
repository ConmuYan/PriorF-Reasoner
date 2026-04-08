#!/bin/bash
# Export teacher evidence cards for Amazon / YelpChi from PriorF-GNN assets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GNN_DIR="${GNN_DIR:-/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-GNN}"
DATASET="${DATASET:-amazon}"
SPLIT="${SPLIT:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/assets/teacher_exports}"

case "$DATASET" in
  amazon)
    MAT_PATH="${MAT_PATH:-$ROOT_DIR/assets/data/Amazon.mat}"
    TEACHER_DIR="${TEACHER_DIR:-$ROOT_DIR/assets/teacher/amazon}"
    PROC_PATH="${PROC_PATH:-$GNN_DIR/processed_data/amazon/seed_42/data.pt}"
    REL_KEYS=("upu" "usu" "uvu")
    REL_NAMES=("UPU" "USU" "UVU")
    ;;
  yelpchi)
    MAT_PATH="${MAT_PATH:-$ROOT_DIR/assets/data/YelpChi.mat}"
    TEACHER_DIR="${TEACHER_DIR:-$ROOT_DIR/assets/teacher/yelpchi}"
    PROC_PATH="${PROC_PATH:-$GNN_DIR/processed_data/yelpchi/seed_42/data.pt}"
    REL_KEYS=("rur" "rtr" "rsr")
    REL_NAMES=("RUR" "RTR" "RSR")
    ;;
  *)
    echo "ERROR: unsupported DATASET=$DATASET (expected amazon or yelpchi)" >&2
    exit 1
    ;;
esac

PYTHONPATH="$GNN_DIR:$ROOT_DIR:${PYTHONPATH:-}" \
python - "$DATASET" "$SPLIT" "$MAT_PATH" "$TEACHER_DIR" "$PROC_PATH" "$OUTPUT_DIR" "${REL_KEYS[*]}" "${REL_NAMES[*]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from priorf_reasoner_slm.priorf_teacher.export_scores import export_teacher_evidence, save_evidence
from priorf_reasoner_slm.priorf_teacher.load_teacher import load_teacher_model
from lghgcl.models.lg_hgcl_v2 import LGHGCLNetV2

dataset, split, mat_path, teacher_dir, proc_path, output_dir, rel_keys_str, rel_names_str = sys.argv[1:]
mat_path = Path(mat_path)
teacher_dir = Path(teacher_dir)
proc_path = Path(proc_path)
output_dir = Path(output_dir)
rel_keys = rel_keys_str.split()
rel_names = rel_names_str.split()

teacher_ckpt = teacher_dir / "best_model.pt"
summary_path = teacher_dir / "model_summary.json"
if not teacher_ckpt.exists():
    raise FileNotFoundError(f"teacher checkpoint not found: {teacher_ckpt}")
if not summary_path.exists():
    raise FileNotFoundError(f"teacher summary not found: {summary_path}")
if not proc_path.exists():
    raise FileNotFoundError(f"processed data not found: {proc_path}")

summary = json.loads(summary_path.read_text())
model_cfg = dict(summary.get("config", {}).get("model", {}))

data = torch.load(proc_path, map_location="cpu", weights_only=False)
_ = mat_path  # kept as a source of truth for the asset layout

x_full = torch.cat([data.x, data.hsd.unsqueeze(1)], dim=1)
model_cfg["in_dim"] = x_full.size(1)
teacher = load_teacher_model(teacher_ckpt, LGHGCLNetV2, model_cfg, device="cpu")

relations = {}
edge_type_parts = []
edge_tensors = []
for idx, (key, name) in enumerate(zip(rel_keys, rel_names)):
    edge_index = data.edge_index_dict[key]
    relations[name] = {"edge_index": edge_index}
    edge_tensors.append(edge_index)
    edge_type_parts.append(torch.full((edge_index.size(1),), idx, dtype=torch.long))

edge_index = torch.cat(edge_tensors, dim=1)
edge_type = torch.cat(edge_type_parts, dim=0)

if split == "train":
    train_mask, val_mask, test_mask = data.train_mask, torch.zeros_like(data.train_mask), torch.zeros_like(data.train_mask)
elif split == "val":
    train_mask, val_mask, test_mask = torch.zeros_like(data.train_mask), data.val_mask, torch.zeros_like(data.train_mask)
elif split == "test":
    train_mask, val_mask, test_mask = torch.zeros_like(data.train_mask), torch.zeros_like(data.train_mask), data.test_mask
elif split == "all":
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
else:
    raise ValueError(f"Unsupported split: {split}")

df = export_teacher_evidence(
    model=teacher,
    x=x_full,
    edge_index=edge_index,
    edge_type=edge_type,
    hsd=data.hsd,
    y=data.y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    relations=relations,
    dataset_name=dataset,
    device="cpu",
)

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{dataset}_{split}_evidence.parquet"
save_evidence(df, output_path)
print(output_path)
PY

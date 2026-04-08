#!/bin/bash
# Create the local asset layout expected by PriorF-Reasoner.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GNN_DIR="${GNN_DIR:-/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-GNN}"

AMAZON_MAT="${AMAZON_MAT:-$GNN_DIR/datasets/Amazon.mat}"
YELPCHI_MAT="${YELPCHI_MAT:-$GNN_DIR/datasets/YelpChi.mat}"
AMAZON_TEACHER="${AMAZON_TEACHER:-$GNN_DIR/runs/rerun_20260404/amazon_full/seed_42/best_model.pt}"
AMAZON_SUMMARY="${AMAZON_SUMMARY:-$GNN_DIR/runs/rerun_20260404/amazon_full/seed_42/model_summary.json}"
YELPCHI_TEACHER="${YELPCHI_TEACHER:-$GNN_DIR/runs/rerun_20260404/yelpchi_full/seed_42/best_model.pt}"
YELPCHI_SUMMARY="${YELPCHI_SUMMARY:-$GNN_DIR/runs/rerun_20260404/yelpchi_full/seed_42/model_summary.json}"

mkdir -p \
  "$ROOT_DIR/assets/data" \
  "$ROOT_DIR/assets/teacher/amazon" \
  "$ROOT_DIR/assets/teacher/yelpchi" \
  "$ROOT_DIR/assets/teacher_exports"

link_file() {
  local src="$1"
  local dst="$2"
  if [ ! -f "$src" ]; then
    echo "ERROR: source file not found: $src" >&2
    exit 1
  fi
  ln -sfn "$src" "$dst"
  echo "linked: $dst -> $src"
}

link_file "$AMAZON_MAT" "$ROOT_DIR/assets/data/Amazon.mat"
link_file "$YELPCHI_MAT" "$ROOT_DIR/assets/data/YelpChi.mat"
link_file "$AMAZON_TEACHER" "$ROOT_DIR/assets/teacher/amazon/best_model.pt"
link_file "$AMAZON_SUMMARY" "$ROOT_DIR/assets/teacher/amazon/model_summary.json"
link_file "$YELPCHI_TEACHER" "$ROOT_DIR/assets/teacher/yelpchi/best_model.pt"
link_file "$YELPCHI_SUMMARY" "$ROOT_DIR/assets/teacher/yelpchi/model_summary.json"

echo
echo "Asset layout created under: $ROOT_DIR/assets"

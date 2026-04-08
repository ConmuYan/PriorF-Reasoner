# PriorF-Reasoner — Project Context

## Overview

**PriorF-Reasoner** trains a small LLM (Qwen3-4B) as a student model for graph fraud detection, using PriorF-GNN as the teacher and structural evidence cards as the knowledge bridge.

**Spec**: See `read.md` for the full design document.

## Architecture

```
PriorF-GNN (teacher, frozen)
    → export_scores.py → per-node evidence (parquet)
    → serializer.py → Evidence Cards (JSON)
    → dataset_builder.py → SFT dataset (JSONL)
Qwen3-4B (student, LoRA)
    Stage 1: SFT (L_gen only)
    Stage 2: Co-train (L_gen + λ₁·L_cls + λ₂·L_distill)
    → eval: gen_only / head_only / fusion
```

## Directory Layout

```
priorf_reasoner_slm/          # Main package
  graph_data/       mat_loader, split_manager, validators, adjacency_builder
  priorf_teacher/   load_teacher, export_hooks, export_scores
  evidence/         feature_extractors, evidence_schema, serializer,
                    rationale_templates, dataset_builder
  llm/              model_wrapper, tokenizer_utils, cls_head, losses,
                    collators, generation, fusion
  train/            train_sft, train_cotrain, train_utils
  eval/             metrics, eval_gen_only, eval_head_only, eval_fusion, faithfulness
  configs/train/    sft_config.yaml, cotrain_config.yaml
  scripts/          run_smoke.sh, run_train_qwen4b.sh, run_eval.sh, run_full_pipeline.sh
  tests/            183 tests (all passing)

datasets/           Amazon.mat, YelpChi.mat
assets/             teacher_exports/ (parquet per dataset+split)
outputs/            SFT adapters, co-train adapters, eval results
meetings/           Codex review records
```

## Key Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `/data1/mq/models/Qwen3-4B-Instruct-2507` (local) |
| LoRA rank | 16, alpha 32, target: q/k/v/o_proj |
| SFT LR | 1e-4, 3 epochs, batch 4, grad_accum 4 |
| CoTrain LR | 5e-5, 3 epochs, λ_cls=0.1, λ_distill=0.1, temp=2.0 |
| Max seq length | 2048 |
| BF16 | true |
| Conda env | `/data1/mq/conda_envs/priorfgnn` |

## Training Mode: Per-Dataset

Each dataset (Amazon, YelpChi) is trained **independently** — no cross-dataset merge. Each dataset has its own relation schema:
- Amazon: UPU, USU, UVU
- YelpChi: RUR, RTR, RSR

The `feature_extractors.discover_row_relations()` function auto-detects which relations are valid per row, skipping None/NaN columns.

## Environment Setup

```bash
export PATH="/data1/mq/conda_envs/priorfgnn/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Commands (Quick Reference)

### Run tests
```bash
python -m pytest priorf_reasoner_slm/tests/ -q
```

### Smoke test (<1 min)
```bash
bash priorf_reasoner_slm/scripts/run_smoke.sh
```

### Full pipeline (per-dataset, smoke → train → eval)
```bash
GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh
# Single dataset only:
DATASETS=amazon GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh
```

### Training only (single dataset)
```bash
MODEL_NAME=/data1/mq/models/Qwen3-4B-Instruct-2507 \
TEACHER_CSV=assets/teacher_exports/amazon_train_evidence.parquet \
STAGE1_OUTPUT=outputs/formal_amazon/sft \
STAGE2_OUTPUT=outputs/formal_amazon/cotrain \
bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh
```

### Evaluation only
```bash
ADAPTER_PATH=outputs/formal_amazon/cotrain/final_cotrain/adapter \
CLS_HEAD_PATH=outputs/formal_amazon/cotrain/final_cotrain/cls_head.pt \
EVIDENCE_CSV=assets/teacher_exports/amazon_test_evidence.parquet \
EVAL_OUTPUT=outputs/formal_amazon/eval \
bash priorf_reasoner_slm/scripts/run_eval.sh
```

## Datasets

| Dataset | Nodes | Features | Relations | Anomaly% |
|---------|-------|----------|-----------|----------|
| Amazon  | 11,944| 25       | UPU, USU, UVU | ~14% |
| YelpChi | 45,954| 32       | RUR, RTR, RSR | ~14% |

## Loss Function

```
L = L_gen + λ₁·L_cls + λ₂·L_distill
```
- L_gen: cross-entropy on generated JSON tokens
- L_cls: BCE on last-token hidden state → linear classifier
- L_distill: KL divergence aligning student score with teacher_prob

## Evaluation Modes

1. **gen_only**: Parse JSON `score` from generated text → AUROC
2. **head_only**: Classification head probability → AUROC
3. **fusion**: `p = α·p_cls + (1-α)·p_teacher`, α optimized on validation set

## Faithfulness Metrics

- **Sufficiency**: How much does keeping only evidence improve over no evidence?
- **Comprehensiveness**: How much does removing evidence hurt?
- **Evidence-ablation**: Per-feature impact analysis

## Acceptance Criteria (from read.md)

1. Teacher results align with PriorF-GNN baseline (Amazon ~0.982/0.901, YelpChi ~0.952/0.836)
2. gen_only significantly better than random/majority
3. head_only at least approaches teacher, no collapse
4. fusion not weaker than teacher on at least one metric
5. JSON parse rate ≥95%
6. Faithfulness metrics output normally

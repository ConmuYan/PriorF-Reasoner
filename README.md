<div align="center">

# PriorF-Reasoner

**Prior-guided LLM Reasoner for Graph Fraud Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Qwen3](https://img.shields.io/badge/model-Qwen3--4B-green.svg)](https://huggingface.co/Qwen/Qwen3-4B)
[![Tests](https://img.shields.io/badge/tests-183%20passing-brightgreen.svg)]()

</div>

---

## What is PriorF-Reasoner?

PriorF-Reasoner trains a small LLM (Qwen3-4B) as a **student** to perform generative reasoning for graph fraud detection, using PriorF-GNN as a **teacher** and structured Evidence Cards as the knowledge bridge.

> Core idea: let the GNN do what it's good at (graph structure modeling), let the LLM do what it's good at (reasoning and explanation). They are teacher and student, not competitors.

Each dataset (Amazon, YelpChi) is trained **independently** — no cross-dataset merge. Each has its own relation schema:
- Amazon: UPU, USU, UVU
- YelpChi: RUR, RTR, RSR

---

## Pipeline Overview

```
datasets/Amazon.mat
       │
       ▼
[Step 0] Data Loading & Teacher Export (offline, already done)
       │
       ▼
assets/teacher_exports/amazon_{train,test}_evidence.parquet
  (per-node: teacher_prob, hsd, asda_switch, branch_gap, disc_level_*, ...)
       │
       ▼
[Step 1] Evidence Card + SFT Dataset Construction (automatic)
       │
       ▼
HuggingFace Dataset (prompt=card JSON, completion=M3 JSON)
       │
       ▼
[Step 2] Stage 1: SFT Training (L_gen only, Qwen3-4B + LoRA)
       │
       ▼
outputs/formal_amazon/sft/final_adapter/  (LoRA weights)
       │
       ▼
[Step 3] Stage 2: CoTrain (L_gen + L_cls + L_distill)
       │
       ▼
outputs/formal_amazon/cotrain/final_cotrain/  (LoRA + cls_head.pt)
       │
       ▼
[Step 4] Evaluation (gen_only / head_only / fusion / faithfulness)
       │
       ▼
outputs/formal_amazon/eval/{gen_eval,head_eval,faithfulness}.json
```

---

## Pipeline Details

### Step 0: Data Loading & Teacher Export (offline)

Already completed. Raw `.mat` files are loaded by `mat_loader.py`, split 70/10/20 by `split_manager.py`, then PriorF-GNN runs forward with hooks (`export_hooks.py`) to extract per-node evidence. The output is a flat parquet file with ~29 columns per node.

```
Input:  datasets/Amazon.mat  (CARE-GNN benchmark format)
        PriorF-GNN trained checkpoint
Output: assets/teacher_exports/amazon_train_evidence.parquet  (11,944 rows)
        assets/teacher_exports/amazon_test_evidence.parquet
Columns: node_id, dataset, split, label,
         teacher_prob, teacher_logit, hsd, hsd_quantile,
         asda_switch, mlp_logit, gnn_logit, branch_gap,
         high_hsd_flag, disc_level_UPU, disc_level_USU, disc_level_UVU,
         suspicious_neighbor_ratio, topk_neighbors, ...
```

### Step 1: Evidence Card Construction (automatic, per training run)

For each node row, the pipeline:

1. **`feature_extractors.py`** — `discover_row_relations()` detects which relation columns are non-null for this row's dataset (e.g. Amazon gets UPU/USU/UVU, YelpChi gets RUR/RTR/RSR)
2. **`serializer.py`** — converts the flat row into a nested `EvidenceCard` Pydantic model
3. **`rationale_templates.py`** — rule-based template generates the M3 prediction output (no external API)
4. **`dataset_builder.py`** — packages into `{"prompt": card_json, "completion": m3_json}` samples

**Evidence Card example (Amazon):**
```json
{
  "dataset": "amazon",
  "node_id": 1832,
  "teacher_summary": {
    "teacher_prob": 0.914,
    "branch_gap": 0.231
  },
  "structure_evidence": {
    "hsd_quantile": "top_5_percent",
    "asda_switch": 0.81,
    "high_hsd_flag": true,
    "relation_profile": [
      {"relation": "UPU", "discrepancy_level": "low"},
      {"relation": "USU", "discrepancy_level": "high"},
      {"relation": "UVU", "discrepancy_level": "high"}
    ],
    "neighbor_stats": {
      "suspicious_neighbor_ratio": 0.67,
      "topk_neighbors": 6
    }
  },
  "task": "Predict fraud label and explain using only the provided structural evidence."
}
```

**M3 output the student must generate:**
```json
{
  "label": "fraud",
  "score": 0.93,
  "pattern_hint": "camouflage",
  "evidence": [
    "high HSD quantile",
    "high routing switch",
    "relation discrepancy concentrated on USU and UVU"
  ],
  "rationale": "The node deviates strongly from its neighborhood and the routing mechanism preserves residual discrepancy instead of smoothing it away."
}
```

### Step 2: Stage 1 — SFT Training

```
Input:  HF Dataset (e.g. 11,944 samples for Amazon)
        Qwen3-4B base model (local path)
Trains: Qwen3-4B + LoRA (r=16, alpha=32, target: q/k/v/o_proj)
        Trainable params: 11.8M / 4.03B (0.29%)
Config: LR=1e-4, 3 epochs, batch=4, grad_accum=4, bf16
        Effective batch size = 16, cosine LR schedule with 10% warmup
Loss:   L_gen only (cross-entropy on completion tokens; prompt tokens masked to -100)
Output: outputs/formal_<dataset>/sft/final_adapter/
```

SFT uses TRL's `SFTTrainer` with `DataCollatorForLanguageModeling(completion_only_loss=True)`. The loss only applies to the completion (M3 JSON) tokens, not the evidence card prompt.

### Step 3: Stage 2 — CoTrain

```
Input:  Same teacher parquet + Stage 1 LoRA adapter
Adds:   BinaryClsHead (hidden_dim -> 1 logit, with dropout=0.1)
Config: LR=5e-5, 3 epochs, batch=4, grad_accum=4
Loss:   L = L_gen + 0.1 * L_cls + 0.1 * L_distill
```

Each sample carries 4 tensors:
- `input_ids` / `labels` — tokenized prompt+completion (prompt masked to -100)
- `teacher_probs` — float, from the teacher's `teacher_prob` column
- `cls_targets` — int, ground-truth label (0/1)

**Forward pass:**
1. LLM forward produces `lm_logits` for generation loss
2. Last valid token's hidden state feeds into `cls_head` producing `cls_logits`
3. Tri-loss combines all three

**Loss components:**
- `L_gen`: cross-entropy on completion tokens (same as SFT)
- `L_cls`: BCEWithLogits on cls_logits vs cls_targets
- `L_distill`: KL divergence between student score and teacher_prob (temperature=2.0)

This is a manual training loop (not Trainer), with gradient clipping at max_norm=1.0.

Output: `outputs/formal_<dataset>/cotrain/final_cotrain/adapter/` + `cls_head.pt`

### Step 4: Evaluation

Three evaluation modes plus faithfulness metrics, all run against the **test** evidence parquet:

| Mode | How it works | Output |
|------|-------------|--------|
| `gen_only` | Model generates text, parse JSON `score` field | AUROC, AUPRC, F1, parse_rate, format_correct_rate |
| `head_only` | LLM forward -> last hidden -> cls_head -> sigmoid | AUROC, AUPRC, F1 |
| `fusion` | `p = alpha * p_cls + (1-alpha) * p_teacher`, alpha optimized on validation | AUROC, AUPRC, F1, best_alpha |

Alpha is selected once on validation data, then fixed for test. Never tuned on test.

**Faithfulness metrics:**
- **Sufficiency**: how much better does the model predict with the full evidence card vs no evidence?
- **Comprehensiveness**: how much does prediction quality drop when evidence is removed?
- **Evidence-ablation**: per-feature impact analysis

---

## Loss Function

$$\mathcal{L} = \mathcal{L}_{gen} + \lambda_1 \cdot \mathcal{L}_{cls} + \lambda_2 \cdot \mathcal{L}_{distill}$$

| Component | What | Default Weight |
|-----------|------|---------------|
| `L_gen` | CE on generated JSON tokens | 1.0 |
| `L_cls` | BCE from cls_head on last-token hidden state | 0.1 |
| `L_distill` | KL divergence vs teacher soft score (temp=2.0) | 0.1 |

Stage 1 (SFT) uses only `L_gen`. Stage 2 (CoTrain) uses all three.

---

## Quick Start

### Environment

```bash
# Using conda (recommended)
conda activate /data1/mq/conda_envs/priorfgnn
# Or: export PATH="/data1/mq/conda_envs/priorfgnn/bin:$PATH"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Smoke Test (< 1 min)

```bash
bash priorf_reasoner_slm/scripts/run_smoke.sh
```

Checks: Python env, module imports, config files, data collators, eval metrics.

### Full Pipeline (per-dataset)

```bash
# Both datasets
GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh

# Single dataset only
DATASETS=amazon GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh
```

This runs: smoke test -> SFT -> CoTrain -> eval, per dataset independently.

### Training Only

```bash
export MODEL_NAME=/data1/mq/models/Qwen3-4B-Instruct-2507
export TEACHER_CSV=assets/teacher_exports/amazon_train_evidence.parquet
export STAGE1_OUTPUT=outputs/formal_amazon/sft
export STAGE2_OUTPUT=outputs/formal_amazon/cotrain

bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh
```

### Evaluation Only

```bash
export ADAPTER_PATH=outputs/formal_amazon/cotrain/final_cotrain/adapter
export CLS_HEAD_PATH=outputs/formal_amazon/cotrain/final_cotrain/cls_head.pt
export EVIDENCE_CSV=assets/teacher_exports/amazon_test_evidence.parquet
export EVAL_OUTPUT=outputs/formal_amazon/eval

bash priorf_reasoner_slm/scripts/run_eval.sh
```

### Continue After SFT

If SFT has completed and you want to run CoTrain + eval:

```bash
DATASET=amazon bash priorf_reasoner_slm/scripts/run_after_sft.sh
```

---

## Monitoring Training

### TensorBoard (SFT stage)

SFT logs to TensorBoard automatically (`report_to: [tensorboard]`):

```bash
tensorboard --logdir outputs/formal_amazon/sft/runs --port 6006
```

Logged: `train/loss` every 10 steps, `train/learning_rate`.

### CoTrain stage

CoTrain uses a manual training loop, prints to stdout every 10 steps:

```
Step 100 | loss=2.4521 | gen=2.1234 | cls=0.6812 | dist=0.0475
Epoch 1/3 | loss=2.3100 | gen=2.0100 | cls=0.6200 | dist=0.0400
```

### GPU monitoring

```bash
watch -n 5 nvidia-smi
```

---

## Project Structure

```
PriorF-Reasoner/
├── priorf_reasoner_slm/
│   ├── graph_data/           # mat_loader, split_manager, validators
│   │   ├── mat_loader.py     # Load .mat → standard format (x, y, relations)
│   │   ├── split_manager.py  # Stratified 70/10/20 split
│   │   └── validators.py     # NaN/shape/symmetry/label checks
│   ├── priorf_teacher/       # PriorF-GNN integration
│   │   ├── load_teacher.py   # Load trained PriorF-GNN checkpoint
│   │   ├── export_hooks.py   # Forward hooks for MLP/GNN branch, ASDA switch
│   │   └── export_scores.py  # Per-node evidence export → parquet
│   ├── evidence/             # Evidence card construction
│   │   ├── evidence_schema.py     # Pydantic: EvidenceCard, PredictionOutput
│   │   ├── feature_extractors.py  # Row → structured features, relation discovery
│   │   ├── serializer.py          # Flat row → nested EvidenceCard
│   │   ├── rationale_templates.py # Rule-based M3 generation (no API)
│   │   └── dataset_builder.py     # → HuggingFace Dataset (prompt/completion)
│   ├── llm/                  # Student model components
│   │   ├── model_wrapper.py  # Qwen3-4B loading + LoRA
│   │   ├── tokenizer_utils.py # Tokenizer + chat template
│   │   ├── cls_head.py       # Binary classification head
│   │   ├── losses.py         # PriorFLoss (gen + cls + distill)
│   │   ├── collators.py      # SFTDataCollator, CoTrainDataCollator
│   │   ├── generation.py     # Batch generation + JSON parsing
│   │   └── fusion.py         # Student-teacher score fusion
│   ├── train/
│   │   ├── train_sft.py      # Stage 1: SFT with TRL SFTTrainer
│   │   ├── train_cotrain.py  # Stage 2: CoTrain manual loop
│   │   └── train_utils.py    # Device detection, LoRA save/load
│   ├── eval/
│   │   ├── metrics.py        # AUROC, AUPRC, F1, G-means
│   │   ├── eval_gen_only.py  # Generation-mode evaluation
│   │   ├── eval_head_only.py # Classification head evaluation
│   │   ├── eval_fusion.py    # Fused evaluation with alpha optimization
│   │   └── faithfulness.py   # Sufficiency, comprehensiveness, ablation
│   ├── configs/train/
│   │   ├── sft_config.yaml
│   │   └── cotrain_config.yaml
│   ├── scripts/
│   │   ├── run_smoke.sh           # Smoke test (<1 min)
│   │   ├── run_full_pipeline.sh   # Per-dataset: smoke → SFT → CoTrain → eval
│   │   ├── run_train_qwen4b.sh    # SFT + CoTrain only
│   │   ├── run_eval.sh            # Evaluation only
│   │   ├── run_after_sft.sh       # CoTrain + eval (after SFT done)
│   │   ├── export_teacher_evidence.sh
│   │   └── setup_assets.sh
│   └── tests/                # 183 unit tests (all passing)
├── datasets/                 # Amazon.mat, YelpChi.mat
├── assets/
│   └── teacher_exports/      # Per-dataset parquet files
├── outputs/                  # Training outputs, adapters, eval results
├── meetings/                 # Code review records
├── CLAUDE.md                 # Agent context file
└── read.md                   # Design specification (Chinese)
```

---

## Datasets

| Dataset | Nodes | Features | Relations | Anomaly Rate |
|---------|-------|----------|-----------|-------------|
| Amazon  | 11,944| 25       | UPU, USU, UVU | ~6.9% |
| YelpChi | 45,954| 32       | RUR, RTR, RSR | ~14.5% |

---

## Key Configuration

| Parameter | Default |
|-----------|---------|
| Base model | Qwen3-4B (local) |
| LoRA rank / alpha | 16 / 32 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 11.8M / 4.03B (0.29%) |
| SFT LR | 1e-4 |
| CoTrain LR | 5e-5 |
| Epochs | 3 (both stages) |
| Batch / grad_accum | 4 / 4 |
| Max seq length | 2048 |
| Precision | bf16 |
| lambda_cls | 0.1 |
| lambda_distill | 0.1 |
| Distill temperature | 2.0 |

---

## Acceptance Criteria

1. Teacher export aligns with PriorF-GNN baseline (Amazon ~0.982/0.901, YelpChi ~0.952/0.836)
2. `gen_only` significantly better than random/majority class
3. `head_only` at least approaches teacher, no collapse
4. `fusion` not weaker than teacher on at least one main metric, no significant degradation on the other
5. JSON parse rate >= 95%
6. Faithfulness metrics (sufficiency, comprehensiveness) output normally

---

## Installation

```bash
# Using conda (recommended)
conda create -n priorfgnn python=3.10
conda activate priorfgnn

# PyTorch (match your CUDA version from pytorch.org)
pip install torch torchvision torchaudio

# Core dependencies
pip install -U transformers accelerate datasets peft trl safetensors
pip install -U scipy scikit-learn pandas pyarrow pydantic

# PyG
pip install -U torch_geometric

# Optional: FlashAttention (CUDA 12+, Ampere/Ada/Hopper GPU)
pip install flash-attn --no-build-isolation
```

---

## Tests

```bash
python -m pytest priorf_reasoner_slm/tests/ -q
```

183 tests covering: mat_loader, teacher export, evidence schema, dataset builder, tokenization, losses, collators, model forward, eval metrics.

---

## Citation

```bibtex
@inproceedings{priorf2026,
  title={PriorF-Reasoner: Prior-guided LLM Reasoning for Interpretable Graph Fraud Detection},
  author={},
  booktitle={},
  year={2026}
}
```

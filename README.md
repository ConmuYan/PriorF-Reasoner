<div align="center">

# PriorF-Reasoner

**用于图欺诈检测的先验引导 LLM 推理器**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Qwen3](https://img.shields.io/badge/model-Qwen3--4B-green.svg)](https://huggingface.co/Qwen/Qwen3-4B)
[![Tests](https://img.shields.io/badge/tests-183%20passing-brightgreen.svg)]()

</div>

---

## 什么是 PriorF-Reasoner？

PriorF-Reasoner 使用一个小型 LLM（Qwen3-4B）作为**学生模型**，在图欺诈检测任务上进行生成式推理；以 PriorF-GNN 作为**教师模型**，并通过结构化 Evidence Card 作为知识桥梁。

> 核心思想：让 GNN 做它擅长的事（图结构建模），让 LLM 做它擅长的事（推理与解释）。二者是师生关系，而不是竞争关系。

每个数据集（Amazon、YelpChi）都**独立训练**——不进行跨数据集合并。每个数据集都有自己的关系模式：
- Amazon：UPU、USU、UVU
- YelpChi：RUR、RTR、RSR

---

## 流程总览

```
datasets/Amazon.mat
        │
        ▼
[步骤 0] 数据加载与教师导出（离线，已完成）
       │
       ▼
assets/teacher_exports/amazon_{train,test}_evidence.parquet
  (per-node: teacher_prob, hsd, asda_switch, branch_gap, disc_level_*, ...)
        │
        ▼
[步骤 1] Evidence Card + SFT 数据集构建（自动）
       │
       ▼
HuggingFace Dataset (prompt=card JSON, completion=M3 JSON)
        │
        ▼
[步骤 2] 阶段 1：SFT 训练（仅 L_gen，Qwen3-4B + LoRA）
       │
       ▼
outputs/formal_amazon/sft/final_adapter/  (LoRA weights)
        │
        ▼
[步骤 3] 阶段 2：CoTrain（L_gen + L_cls + L_distill）
       │
       ▼
outputs/formal_amazon/cotrain/final_cotrain/  (LoRA + cls_head.pt)
        │
        ▼
[步骤 4] 评估（gen_only / head_only / fusion / faithfulness）
       │
       ▼
outputs/formal_amazon/eval/{gen_eval,head_eval,faithfulness}.json
```

---

## 流程细节

### 步骤 0：数据加载与教师导出（离线）

已完成。原始 `.mat` 文件由 `mat_loader.py` 加载，使用 `split_manager.py` 按 70/10/20 划分，然后 PriorF-GNN 通过 hook（`export_hooks.py`）前向运行，提取每个节点的证据。输出是一个扁平的 parquet 文件，每个节点约 29 列。

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

### 步骤 1：Evidence Card 构建（自动，每次训练运行）

对于每一行节点，流程如下：

1. **`feature_extractors.py`** —— `discover_row_relations()` 检测该数据集这一行中哪些关系列非空（例如 Amazon 使用 UPU/USU/UVU，YelpChi 使用 RUR/RTR/RSR）
2. **`serializer.py`** —— 将扁平行转换为嵌套的 `EvidenceCard` Pydantic 模型
3. **`rationale_templates.py`** —— 基于规则的模板生成 M3 预测输出（无外部 API）
4. **`dataset_builder.py`** —— 封装为 `{"prompt": card_json, "completion": m3_json}` 样本

**Evidence Card 示例（Amazon）：**
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

**学生模型必须生成的 M3 输出：**
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

### 步骤 2：阶段 1 —— SFT 训练

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

SFT 使用 TRL 的 `SFTTrainer` 和 `DataCollatorForLanguageModeling(completion_only_loss=True)`。损失只作用于 completion（M3 JSON）token，而不作用于 evidence card prompt。

### 步骤 3：阶段 2 —— CoTrain

```
Input:  Same teacher parquet + Stage 1 LoRA adapter
Adds:   BinaryClsHead (hidden_dim -> 1 logit, with dropout=0.1)
Config: LR=5e-5, 3 epochs, batch=4, grad_accum=4
Loss:   L = L_gen + 0.1 * L_cls + 0.1 * L_distill
```

每个样本包含 4 个张量：
- `input_ids` / `labels` — tokenized prompt+completion (prompt masked to -100)
- `teacher_probs` — float, from the teacher's `teacher_prob` column
- `cls_targets` — int, ground-truth label (0/1)

**前向过程：**
1. LLM 前向输出 `lm_logits`，用于生成损失
2. 最后一个有效 token 的 hidden state 输入 `cls_head`，得到 `cls_logits`
3. 三重损失合并这三部分

**损失组件：**
- `L_gen`：completion token 上的交叉熵（与 SFT 相同）
- `L_cls`：`cls_logits` 与 `cls_targets` 的 BCEWithLogits
- `L_distill`：学生分数与 `teacher_prob` 之间的 KL 散度（temperature=2.0）

这是一个手动训练循环（不是 Trainer），并使用 `max_norm=1.0` 做梯度裁剪。

Output: `outputs/formal_<dataset>/cotrain/final_cotrain/adapter/` + `cls_head.pt`

### 步骤 4：评估

三种评估模式加上 faithfulness 指标，全部针对 **test** evidence parquet 运行：

| 模式 | 工作方式 | 输出 |
|------|-------------|--------|
| `gen_only` | Model generates text, parse JSON `score` field | AUROC, AUPRC, F1, parse_rate, format_correct_rate |
| `head_only` | LLM forward -> last hidden -> cls_head -> sigmoid | AUROC, AUPRC, F1 |
| `fusion` | `p = alpha * p_cls + (1-alpha) * p_teacher`, alpha optimized on validation | AUROC, AUPRC, F1, best_alpha |

Alpha 只在验证集上选择一次，然后在测试集上固定使用。绝不在测试集上调参。

**Faithfulness 指标：**
- **Sufficiency**：使用完整 evidence card 与不使用 evidence 时，模型预测提升了多少？
- **Comprehensiveness**：移除 evidence 后，预测质量下降了多少？
- **Evidence-ablation**：按特征分析影响

---

## 损失函数

$$\mathcal{L} = \mathcal{L}_{gen} + \lambda_1 \cdot \mathcal{L}_{cls} + \lambda_2 \cdot \mathcal{L}_{distill}$$

| 组件 | 含义 | 默认权重 |
|-----------|------|---------------|
| `L_gen` | CE on generated JSON tokens | 1.0 |
| `L_cls` | BCE from cls_head on last-token hidden state | 0.1 |
| `L_distill` | KL divergence vs teacher soft score (temp=2.0) | 0.1 |

阶段 1（SFT）只使用 `L_gen`。阶段 2（CoTrain）使用全部三项。

---

## 快速开始

### 环境

```bash
# 使用 conda（推荐）
conda activate /data1/mq/conda_envs/priorfgnn
# 或：export PATH="/data1/mq/conda_envs/priorfgnn/bin:$PATH"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 烟雾测试（< 1 分钟）

```bash
bash priorf_reasoner_slm/scripts/run_smoke.sh
```

检查项：Python 环境、模块导入、配置文件、data collator、评估指标。

### 完整流水线（按数据集）

```bash
# 两个数据集都跑
GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh

# 只跑单个数据集
DATASETS=amazon GPU_ID=1 bash priorf_reasoner_slm/scripts/run_full_pipeline.sh
```

执行顺序：烟雾测试 -> SFT -> CoTrain -> eval，且按数据集独立进行。

### 仅训练

```bash
export MODEL_NAME=/data1/mq/models/Qwen3-4B-Instruct-2507
export TEACHER_CSV=assets/teacher_exports/amazon_train_evidence.parquet
export STAGE1_OUTPUT=outputs/formal_amazon/sft
export STAGE2_OUTPUT=outputs/formal_amazon/cotrain

bash priorf_reasoner_slm/scripts/run_train_qwen4b.sh
```

### 仅评估

```bash
export ADAPTER_PATH=outputs/formal_amazon/cotrain/final_cotrain/adapter
export CLS_HEAD_PATH=outputs/formal_amazon/cotrain/final_cotrain/cls_head.pt
export EVIDENCE_CSV=assets/teacher_exports/amazon_test_evidence.parquet
export EVAL_OUTPUT=outputs/formal_amazon/eval

bash priorf_reasoner_slm/scripts/run_eval.sh
```

### SFT 后继续运行

如果 SFT 已完成，并且你想运行 CoTrain + eval：

```bash
DATASET=amazon bash priorf_reasoner_slm/scripts/run_after_sft.sh
```

---

## 训练监控

### TensorBoard（SFT 阶段）

SFT logs to TensorBoard automatically (`report_to: [tensorboard]`):

```bash
tensorboard --logdir outputs/formal_amazon/sft/runs --port 6006
```

记录内容：每 10 步记录一次 `train/loss` 和 `train/learning_rate`。

### CoTrain 阶段

CoTrain uses a manual training loop, prints to stdout every 10 steps:

```
Step 100 | loss=2.4521 | gen=2.1234 | cls=0.6812 | dist=0.0475
Epoch 1/3 | loss=2.3100 | gen=2.0100 | cls=0.6200 | dist=0.0400
```

### GPU 监控

```bash
watch -n 5 nvidia-smi
```

---

## 项目结构

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

## 数据集

| 数据集 | 节点数 | 特征数 | 关系 | 异常率 |
|---------|-------|----------|-----------|-------------|
| Amazon  | 11,944| 25       | UPU, USU, UVU | ~6.9% |
| YelpChi | 45,954| 32       | RUR, RTR, RSR | ~14.5% |

---

## 关键配置

| 参数 | 默认值 |
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

## 验收标准

1. 教师导出结果与 PriorF-GNN 基线一致（Amazon 约 0.982/0.901，YelpChi 约 0.952/0.836）
2. `gen_only` 明显优于随机/多数类
3. `head_only` 至少接近教师模型，且不崩溃
4. `fusion` 在至少一个主指标上不弱于教师模型，另一个指标也无明显退化
5. JSON 解析率 >= 95%
6. Faithfulness 指标（sufficiency、comprehensiveness）可正常输出

---

## 安装

```bash
# 使用 conda（推荐）
conda create -n priorfgnn python=3.10
conda activate priorfgnn

# PyTorch（请根据 pytorch.org 匹配你的 CUDA 版本）
pip install torch torchvision torchaudio

# 核心依赖
pip install -U transformers accelerate datasets peft trl safetensors
pip install -U scipy scikit-learn pandas pyarrow pydantic

# PyG
pip install -U torch_geometric

# 可选：FlashAttention（CUDA 12+，Ampere/Ada/Hopper GPU）
pip install flash-attn --no-build-isolation
```

---

## 测试

```bash
python -m pytest priorf_reasoner_slm/tests/ -q
```

183 个测试覆盖：mat_loader、teacher export、evidence schema、dataset builder、tokenization、losses、collators、模型前向、评估指标。

---

## 引用

```bibtex
@inproceedings{priorf2026,
  title={PriorF-Reasoner: Prior-guided LLM Reasoning for Interpretable Graph Fraud Detection},
  author={},
  booktitle={},
  year={2026}
}
```

## 项目定义

项目名先定为：

**PriorF-Reasoner**
*Prior-guided LLM Reasoner for Graph Fraud Detection*

项目主张：

1. PriorF-GNN 是 teacher 和结构证据提取器，不替换。
2. 小参数 LLM 是 student，只读“结构证据卡”，不直接生啃原始图。
3. v1 不依赖原始评论文本，只依赖标准 `.mat` 数据和 PriorF 中间产物。
4. v1 的主结果以 **分类头 / 融合结果** 为准，生成文本主要承担推理与解释。
5. 训练主栈采用 **Hugging Face Transformers + PEFT + TRL + Accelerate**；vLLM 只做后期批量推理加速，不进入第一版训练闭环。Qwen3 已在 Transformers 中加入并支持 thinking / non-thinking 模式切换；Gemma 4 也已可在 Hugging Face Transformers 中运行，Google 官方文档同时给出 Hugging Face 推理与 LoRA 调优路径。([Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen3))

## 模型选择与平台决策

**主模型：Qwen3-4B**
原因很简单：Qwen3-4B 是 4.0B 参数，原生 32,768 上下文，支持 YaRN 扩展到 131,072，上手快，Transformers / vLLM / llama.cpp / Ollama 等生态都已覆盖；而且 Qwen3 的默认 thinking 能力可以显式关闭，这对你这种稳定输出 JSON 的任务非常重要。([Hugging Face](https://huggingface.co/Qwen/Qwen3-4B))

**强化模型：Qwen3-8B**
Qwen3-8B 是 8.2B 参数，同样原生 32,768 上下文，作为冲榜版本更合适；但第一版不建议直接上 8B。([Hugging Face](https://huggingface.co/Qwen/Qwen3-8B))

**备选模型：Gemma 4 E4B-it**
Gemma 4 是 2026 年 4 月的新系列，E4B / E4B-it 明确面向 edge / consumer GPU 场景，Google 官方也给出了 Hugging Face 运行说明和 LoRA 调优说明。它适合做 cross-model 验证，但不适合替代 Qwen3 成为第一主线，因为你现在更需要成熟度高、资料齐的训练闭环。([Hugging Face](https://huggingface.co/google/gemma-4-E4B))

**训练框架决策**
第一选择是 **Hugging Face 原生栈**，原因是：

- Transformers 直接支持 Qwen3，Gemma 也有官方 HF 使用文档。([Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen3))
- PEFT 的 LoRA 接入简单，TRL 的 `SFTTrainer` 直接支持 prompt-completion 和 conversational dataset。([Hugging Face](https://huggingface.co/docs/peft/developer_guides/lora))
- Chat template 能用 `apply_chat_template` 统一处理模型输入格式。([Hugging Face](https://huggingface.co/docs/transformers/chat_templating))

第二选择是 **LLaMA-Factory**，作为 fallback 或快速对照实验框架。它已经明确支持 Qwen3、Gemma、LoRA、QLoRA、FlashAttention-2、Unsloth 与 vLLM。([GitHub](https://github.com/hiyouga/LLaMAFactory))

## 交给 Codex/Claude Code 的工程目标

Codex/Claude Code 需要按下面顺序推进，不要跳步：

### Phase 0：冻结问题定义

只解决这一个问题：

> 在 Amazon / YelpChi 标准 `.mat` benchmark 上，利用 PriorF-GNN 中间产物构造结构证据卡，训练一个小参数 LLM 做生成式推理，并通过辅助分类头与蒸馏提升最终判别性能。

v1 明确不做：

- 原始评论文本引入
- 新图结构重建
- 多模型 RLHF
- agent/tool calling
- 多模态图文输入

## 代码仓库结构

Codex/Claude Code 直接按这个目录建：

```text
priorf_reasoner_slm/
  configs/
    data/
    model/
    train/
    eval/
  data/
    raw/
    processed/
    evidence/
    sft/
  priorf_teacher/
    load_teacher.py
    export_hooks.py
    export_scores.py
  graph_data/
    mat_loader.py
    split_manager.py
    adjacency_builder.py
    validators.py
  evidence/
    feature_extractors.py
    evidence_schema.py
    serializer.py
    rationale_templates.py
    dataset_builder.py
  llm/
    tokenizer_utils.py
    model_wrapper.py
    cls_head.py
    losses.py
    collators.py
    generation.py
    fusion.py
  train/
    train_sft.py
    train_cotrain.py
    train_utils.py
  eval/
    metrics.py
    eval_gen_only.py
    eval_head_only.py
    eval_fusion.py
    faithfulness.py
  tests/
    test_mat_loader.py
    test_teacher_export.py
    test_evidence_schema.py
    test_dataset_builder.py
    test_tokenization.py
    test_model_forward.py
    test_loss.py
    test_eval.py
    test_smoke_pipeline.py
  scripts/
    prepare_data.py
    export_teacher_evidence.py
    build_sft_dataset.py
    run_smoke.sh
    run_train_qwen4b.sh
    run_train_qwen8b.sh
    run_eval.sh
  README.md
```

## 方法设计，先做最稳版

### M1. Teacher 导出

输入是你已经训练好的 PriorF-GNN。Codex/Claude Code 要导出每个节点的：

- `node_id`
- `split`
- `label`
- `teacher_prob`
- `teacher_logit`
- `hsd`
- `hsd_quantile`
- `asda_switch`
- `mlp_logit`
- `gnn_logit`
- `branch_gap = |mlp_prob - gnn_prob|`
- `top_relations`
- `neighbor_summary`
- `high_hsd_flag`

这里的 `neighbor_summary` 不要上来就做复杂图解释器。v1 只做统计摘要：

- 每种 relation 的度数
- top-k 关系上的邻域差异均值
- suspicious-neighbor ratio
- same-relation discrepancy rank

### M2. 结构证据卡

Codex/Claude Code 需要把每个节点转成一张 Evidence Card，格式固定。

推荐 schema：

```json
{
  "dataset": "YelpChi",
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
      {"relation": "RUR", "discrepancy_level": "low"},
      {"relation": "RTR", "discrepancy_level": "high"},
      {"relation": "RSR", "discrepancy_level": "high"}
    ],
    "neighbor_stats": {
      "suspicious_neighbor_ratio": 0.67,
      "topk_neighbors": 6
    }
  },
  "task": "Predict fraud label and explain using only the provided structural evidence."
}
```

### M3. 训练输出格式

统一输出 JSON，不要自由文本：

```json
{
  "label": "fraud",
  "score": 0.93,
  "pattern_hint": "camouflage",
  "evidence": [
    "high HSD quantile",
    "high routing switch",
    "relation discrepancy concentrated on RTR and RSR"
  ],
  "rationale": "The node deviates strongly from its neighborhood and the routing mechanism preserves residual discrepancy instead of smoothing it away."
}
```

注意这里叫 `pattern_hint`，不是 `pattern_label`。因为 camouflage / co-attack 在标准 benchmark 上不是人工标注真值，只能来自结构规则弱标签。

### M4. 损失函数

v1 的主训练目标写死成：

[
L = L_{gen} + \lambda_1 L_{cls} + \lambda_2 L_{distill}
]

其中：

- `L_gen`：生成 JSON 的 token loss
- `L_cls`：最后一个有效 token 的 hidden state 接线性层做二分类
- `L_distill`：对齐 teacher 的 soft score，优先用 KL 或 MSE

v2 再加 `L_rank`，只给 hard cases：

- `branch_gap` 大
- `teacher_prob` 接近阈值
- `high_hsd_flag = true`

### M5. 输出仲裁

Codex/Claude Code 不要临场“挑最好”。规则固定：

- `gen_only`：解析 JSON 的 `score`
- `head_only`：分类头 `p_cls`
- `fusion`：`p_final = alpha * p_cls + (1 - alpha) * p_teacher`

`alpha` 只在 validation 上选一次，测试集固定。

## 数据验证任务

Codex/Claude Code 先做数据验证，再写模型。

### 验证项

1. `.mat` key 是否完整。CARE-GNN 的官方处理脚本给出了标准键名。([GitHub](https://github.com/YingtongDou/CARE-GNN/blob/master/data_process.py))
2. 节点特征和标签维度是否与论文设定一致：Amazon 25 维、YelpChi 32 维，关系均为 3 类。
3. split 是否固定复现为标准 70/10/20；低标签实验单独做 nested subsampling，不与主结果混用。
4. teacher 导出前后节点顺序不能变化。
5. 任何 evidence 卡不得泄漏测试标签。

### 数据文件产出

- `processed/amazon_base.pkl`
- `processed/yelpchi_base.pkl`
- `evidence/amazon_teacher_export.parquet`
- `evidence/yelpchi_teacher_export.parquet`
- `sft/amazon_train.jsonl`
- `sft/yelpchi_train.jsonl`

## 最小单元编写

Codex/Claude Code 先写这些最小单元，每个都要有测试：

1. `mat_loader.py`
   读取 `.mat`，标准化为：
   - `x`
   - `y`
   - `relations`
   - `homo`
   - `node_ids`
2. `validators.py`
   检查 NaN、shape、稀疏图对称性、label 比例。
3. `export_hooks.py`
   从 PriorF-GNN forward 中抓出 HSD、ASDA switch、branch logit。
4. `serializer.py`
   把结构数值变成 Evidence Card。
5. `rationale_templates.py`
   用规则模板生成训练解释，v1 不接外部闭源 API。
6. `dataset_builder.py`
   生成 Hugging Face Dataset 需要的 `messages` 或 `prompt/completion`。
7. `tokenizer_utils.py`
   用 `apply_chat_template` 统一格式化 Qwen3 / Gemma 的输入。([Hugging Face](https://huggingface.co/docs/transformers/chat_templating))
8. `model_wrapper.py`
   包装 `AutoModelForCausalLM`，暴露 hidden state。
9. `cls_head.py`
   分类头。
10. `losses.py`
    `gen + cls + distill`。

## 最小单元集成

完成上面 10 个最小单元后，Codex/Claude Code 做 4 个集成检查：

1. **Teacher export 集成**
   单个 batch 导出证据并落盘。
2. **Dataset 集成**
   单条 Evidence Card 能变成合法 chat sample。
3. **Model forward 集成**
   一条 sample 能完成 tokenization、forward、生成和分类头输出。
4. **Train-step 集成**
   一个 batch 能完成 loss backward，无 NaN，无 shape error。

## 完整单元编写

完成最小单元后，再补完整单元：

- `eval_gen_only.py`
- `eval_head_only.py`
- `eval_fusion.py`
- `faithfulness.py`
- `run_smoke.sh`
- `run_train_qwen4b.sh`
- `run_train_qwen8b.sh`

### `faithfulness.py` 至少实现三件事

- sufficiency
- comprehensiveness
- evidence-ablation impact

这一步很重要。SEFraud 已经把“解释性”推进到模型机制层面了，你这里必须把“生成解释”至少做成可验证解释，而不是只给几段好看的文本。([Multimedia Semantic Analytics Lab](https://msalab-pku.github.io/publication/sefraud-2024/))

## 完整模型集成

完整系统按这个顺序集成：

1. 先只跑 `Qwen3-4B + evidence card + L_gen`
2. 再加 `cls head`
3. 再加 `distill`
4. 最后加 `fusion`

不要一开始就把 soft prompt、pseudo-label、Qwen3-8B、Gemma 4 全部混进去。

DGP 的经验很明确：图信息进入 LLM 时，**最先要解决的是 token budget 和信息组织**，不是模型越大越好。([arXiv](https://arxiv.org/abs/2507.21653))

## 训练 pipeline

建议 Codex/Claude Code 做两条 pipeline：

### Pipeline A：研究主线

- Teacher 复现 / 加载
- Teacher 证据导出
- 构建 SFT 数据
- SFT 训练
- Co-training 训练
- Eval 三种模式
- 出报告

### Pipeline B：工程验收线

- 数据校验
- 单元测试
- smoke test
- 迷你训练
- 全量训练
- 结果归档

## 冒烟测试

smoke test 只取每个数据集 128 个训练节点、64 个验证节点、64 个测试节点，要求：

- 全流程 30 分钟内跑完
- JSON 可解析率 ≥ 95%
- `train_step` / `eval_step` 均无 NaN
- 至少产出 AUROC / AUPRC
- 单元测试全绿

## 完整训练

完整训练按三层推进：

### Stage 1

`Qwen3-4B`, non-thinking, LoRA, `L_gen` only

### Stage 2

`Qwen3-4B`, non-thinking, LoRA, `L_gen + L_cls + L_distill`

### Stage 3

`Qwen3-8B`, same config, 仅在 Stage 2 稳定后再跑

Gemma 4 E4B-it 只做对照验证，不先做主线。Qwen3 官方模型卡和文档都明确支持 thinking / non-thinking 切换；默认 thinking 打开，而你这条任务更适合 non-thinking 的稳定 JSON 输出。([Hugging Face](https://huggingface.co/Qwen/Qwen3-8B-AWQ))

## 测试验收标准

验收按下面 6 条：

1. Teacher 结果与当前 PriorF-GNN 基准对齐，误差在可接受范围内。目标对齐你的论文级结果：Amazon 约 0.982/0.901，YelpChi 约 0.952/0.836。
2. `gen_only` 能明显优于随机和多数类。
3. `head_only` 至少逼近 teacher，不出现大幅崩溃。
4. `fusion` 在至少一个主指标上不弱于 teacher，且另一个主指标不出现明显退化。
5. JSON 解析稳定，错误率低。
6. faithfulness 三项指标能正常输出。

## 环境配置建议

主建议是 **Linux + NVIDIA GPU + pip venv / uv**。
原因是：

- PyTorch 官方本地安装页建议按机器 CUDA 版本选择 pip 安装命令，且最新稳定版要求 Python 3.10+。([PyTorch](https://pytorch.org/get-started/locally/))
- PyG 最新文档明确写了：对于 `torch > 2.5.0`，conda 包已经不再提供，建议走 pip。([PyG 文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))
- FlashAttention 官方仍然最推荐 Linux / NVIDIA PyTorch 容器环境，CUDA 12.0+，支持 Ampere/Ada/Hopper。([GitHub](https://github.com/Dao-AILab/flash-attention))

### 推荐环境

- Ubuntu 22.04 或 24.04
- Python 3.11
- 最新稳定 PyTorch，按本机 CUDA 版本从官方 selector 生成安装命令
- `transformers` 最新稳定版，至少要满足 Qwen3 与 Gemma 4 支持。Qwen3 的文档说明它已加入 Transformers，且较旧版本会出现 `KeyError: 'qwen3'`；Gemma 官方 Hugging Face 教程也要求使用支持 Gemma 的新版本 Transformers。([Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen3))

### 推荐安装命令

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# 按 pytorch.org selector 选择与你机器匹配的 stable torch 安装命令
# 安装完成后再装以下包
pip install -U transformers accelerate datasets peft trl safetensors sentencepiece
pip install -U bitsandbytes scipy scikit-learn pandas pyarrow pydantic wandb
pip install -U torch_geometric
```

### 可选加速

```bash
pip install flash-attn --no-build-isolation
```

只在满足 CUDA 12+ 且 GPU 属于 Ampere/Ada/Hopper 时启用。([GitHub](https://github.com/Dao-AILab/flash-attention))

### Gemma 4 特别说明

Gemma 4 可以通过 Hugging Face Transformers 使用，但需要先在 Hugging Face 上接受对应模型许可，然后用 HF token 登录。Google 官方 Gemma 文档明确给出了这条流程。([Google AI for Developers](https://ai.google.dev/gemma/docs/core/huggingface_inference?hl=zh-cn))

### vLLM 的位置

vLLM 不作为首发训练工具，只作为后期批量推理加速。Qwen3 现有官方 recipes 和 vLLM 文档都支持该系列。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.html))

## 模型配置建议

### Qwen3-4B 首发配置

- base model: `Qwen/Qwen3-4B`
- mode: non-thinking
- tuning: LoRA 或 4-bit QLoRA
- context: 4k 足够，先不碰 32k / 128k
- output: JSON only

### Qwen3-8B 强化配置

- base model: `Qwen/Qwen3-8B`
- 仅在 Qwen3-4B 主线完全稳定后再跑

### Gemma 4 E4B-it 对照配置

- base model: `google/gemma-4-E4B-it`
- 同样走 LoRA / QLoRA
- 作为 cross-model 验证，不承担主冲榜任务

Qwen3 官方模型卡已经给出 4B 和 8B 的参数规模、32,768 原生上下文和 131,072 的 YaRN 扩展；Gemma 4 E4B 则明确定位于边缘设备、移动端到消费级 GPU 场景。([Hugging Face](https://huggingface.co/Qwen/Qwen3-4B))

## 训练数据格式

Codex/Claude Code 应该生成两种数据：

### 格式 1：prompt-completion

适合 `SFTTrainer`

```json
{"prompt": "...evidence card...", "completion": "{\"label\":\"fraud\", ... }"}
```

### 格式 2：conversational

适合 chat template

```json
{
  "messages": [
    {"role": "system", "content": "You are a fraud-reasoning assistant..."},
    {"role": "user", "content": "...evidence card..."},
    {"role": "assistant", "content": "{\"label\":\"fraud\", ... }"}
  ]
}
```

TRL 的 `SFTTrainer` 兼容这两种格式，conversational 数据会自动应用 chat template。([Hugging Face](https://huggingface.co/docs/trl/v0.25.0/en/sft_trainer))

## 不该做的事

Codex/Claude Code 在第一版不要做这些：

- 不要用 GPT-4 / Claude 批量写 rationale 作为主监督源
- 不要一开始就加 soft prompt token
- 不要一开始就训练 Gemma 和 Qwen 两条主线
- 不要直接用自由文本 label 做最终主结果
- 不要在 test 集上调 `alpha`
- 不要混入你之后自建 text-rich Yelp 扩展数据

## 风险与修正

最大的风险有四个：

第一，**teacher-student 泄漏**。
修正：teacher 只用训练标签训练，test 只导证据，不参与任何模板标注。

第二，**生成解释太花，判别性能不稳**。
修正：主结果看 `head_only` / `fusion`，生成文本只做 reasoning layer。

第三，**模型输出 JSON 不稳定**。
修正：Qwen3 先关 thinking，严格用 chat template 和 schema 校验。([Hugging Face](https://huggingface.co/Qwen/Qwen3-8B-AWQ))

第四，**一开始就想超过 PriorF-GNN**。
修正：先达成 teacher 对齐，再让 `fusion` 追平或小幅超过。要想稳定超过，后面再加 hard-case reweighting、pseudo-label augmentation 和 `L_rank`。
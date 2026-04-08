# PriorF-Reasoner Migration Guide

在新服务器上从零部署 PriorF-Reasoner 的完整前置准备清单。

---

## 1. 拉取代码

```bash
git clone https://github.com/ConmuYan/PriorF-Reasoner.git
cd PriorF-Reasoner
```

## 2. 目录结构总览

以下目录/文件**不在 Git 跟踪中**（被 `.gitignore` 排除），需要手动准备：

| 路径 | 大小 | 说明 | 来源 |
|------|------|------|------|
| `datasets/Amazon.mat` | ~213 MB | Amazon 图欺诈数据集 | 见下方下载 |
| `datasets/YelpChi.mat` | ~199 MB | YelpChi 图欺诈数据集 | 见下方下载 |
| `assets/data/Amazon.mat` | ~213 MB | 同上（供 setup_assets.sh 使用） | 符号链接 |
| `assets/data/YelpChi.mat` | ~199 MB | 同上 | 符号链接 |
| `assets/teacher/amazon/best_model.pt` | ~552 KB | Amazon PriorF-GNN 教师模型 | PriorF-GNN 仓库 |
| `assets/teacher/amazon/model_summary.json` | <1 KB | Amazon 教师模型配置 | PriorF-GNN 仓库 |
| `assets/teacher/yelpchi/best_model.pt` | ~1.5 MB | YelpChi PriorF-GNN 教师模型 | PriorF-GNN 仓库 |
| `assets/teacher/yelpchi/model_summary.json` | <1 KB | YelpChi 教师模型配置 | PriorF-GNN 仓库 |
| `assets/teacher_exports/*.parquet` | ~42 MB (共6个) | 教师导出的节点证据 | 需本地导出 |
| Qwen3-4B-Instruct 模型 | ~7.6 GB | 学生 LLM 基座模型 | HuggingFace |
| `outputs/` | 不定 | 训练产物（适配器、评估结果） | 训练时生成 |

**总需下载/准备的外部数据 ≈ 8.3 GB**（不含训练产物）

## 3. 环境配置

### 3.1 硬件要求

- **GPU**: NVIDIA GPU，显存 ≥ 24 GB（RTX 3090 / A5000 / V100 / A100 等）
- **CUDA**: 12.4+（当前环境使用 CUDA 12.4）
- **磁盘**: ≥ 30 GB 可用空间（模型 + 数据 + 训练产物）
- **内存**: ≥ 32 GB RAM

### 3.2 Conda 环境

```bash
# 创建 conda 环境（Python 3.10）
conda create -n priorfgnn python=3.10 -y
conda activate priorfgnn
```

### 3.3 安装依赖

**按以下顺序安装**，PyG 相关包需要匹配 CUDA 版本：

```bash
# Step 1: PyTorch（CUDA 12.4）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: PyTorch Geometric 及扩展（需匹配 PyTorch 2.6 + CUDA 12.4）
pip install torch_geometric==2.7.0
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Step 3: DGL（用于 PriorF-GNN 教师模型加载）
pip install dgl==2.1.0

# Step 4: HuggingFace 生态
pip install -U transformers==5.5.0 accelerate==1.13.0 \
    datasets==4.8.4 peft==0.18.1 trl==1.0.0 safetensors==0.7.0

# Step 5: 科学计算 & 数据处理
pip install -U scipy==1.15.3 scikit-learn==1.7.2 \
    pandas pyarrow==23.0.1 pydantic==2.12.5

# Step 6: 其他
pip install pytest matplotlib networkx

# Step 7（可选）: Flash Attention（需要 Ampere/Ada/Hopper 架构 GPU + CUDA 12+）
pip install flash-attn --no-build-isolation
```

> **如果 CUDA 版本不是 12.4**: 请到 https://pytorch.org 和 https://data.pyg.org/whl/ 查找对应的 wheel。核心原则是 **PyTorch 版本、CUDA 版本、PyG 扩展版本三者必须一致**。

### 3.4 完整依赖版本参考

以下为当前验证环境的精确版本，可直接复现：

```
python                    3.10.20
torch                     2.6.0+cu124
torch-geometric           2.7.0
torch-scatter             2.1.2+pt26cu124
torch-sparse              0.6.18+pt26cu124
dgl                       2.1.0
transformers              5.5.0
accelerate                1.13.0
datasets                  4.8.4
peft                      0.18.1
trl                       1.0.0
safetensors               0.7.0
scipy                     1.15.3
scikit-learn              1.7.2
pyarrow                   23.0.1
pydantic                  2.12.5
numpy                     2.2.6
```

## 4. 数据集准备

### 4.1 Amazon & YelpChi

这两个数据集来自 CARE-GNN 论文（图欺诈检测标准 benchmark）：

```bash
mkdir -p datasets assets/data

# 方式 A: 从原始来源下载
# Amazon (~213 MB)
wget -O datasets/Amazon.mat "https://github.com/Jhy1993/CARE-GNN/raw/master/data/Amazon.mat"
# YelpChi (~199 MB)
wget -O datasets/YelpChi.mat "https://github.com/Jhy1993/CARE-GNN/raw/master/data/YelpChi.mat"

# 符号链接到 assets/data（供 setup_assets.sh 使用）
ln -sfn "$(pwd)/datasets/Amazon.mat" assets/data/Amazon.mat
ln -sfn "$(pwd)/datasets/YelpChi.mat" assets/data/YelpChi.mat
```

> **备选方式**: 如果上述 GitHub raw 链接不可用，可搜索 "CARE-GNN dataset" 或 "Amazon fraud detection mat" 获取镜像。数据集格式为 MATLAB `.mat` 文件，包含 `features` (节点特征)、`label` (0/1 标签)、`homo`/`net_upu`/`net_usu`/`net_uvu` (邻接矩阵) 等字段。

### 4.2 验证数据集

```bash
python -c "
from priorf_reasoner_slm.graph_data.mat_loader import load_mat
data = load_mat('datasets/Amazon.mat')
print(f'Amazon: {data[\"x\"].shape[0]} nodes, {data[\"x\"].shape[1]} features')
print(f'Labels: {data[\"y\"].sum()}/{len(data[\"y\"])} positive ({data[\"y\"].mean()*100:.1f}%)')
print(f'Relations: {list(data[\"relations\"].keys())}')
"
```

预期输出：
```
Amazon: 11944 nodes, 25 features
Labels: .../11944 positive (~6.9%)
Relations: ['UPU', 'USU', 'UVU']
```

## 5. Qwen3-4B 模型下载

学生模型的基座是 Qwen3-4B-Instruct：

```bash
# 方式 A: 从 HuggingFace 下载（推荐，支持断点续传）
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir /path/to/Qwen3-4B-Instruct-2507 \
    --local-dir-use-symlinks False

# 方式 B: 使用 modelscope（国内网络推荐）
pip install modelscope
modelscope download --model Qwen/Qwen3-4B --local_dir /path/to/Qwen3-4B-Instruct-2507
```

下载完成后确认文件完整：
```bash
ls /path/to/Qwen3-4B-Instruct-2507/
# 应包含: config.json, model-00001-of-00003.safetensors, tokenizer.json, ...
# 总大小 ≈ 7.6 GB
```

> **注意**: 记住你放置模型的路径，训练时通过 `MODEL_NAME` 环境变量指定。当前代码默认路径为 `/data1/mq/models/Qwen3-4B-Instruct-2507`。

## 6. PriorF-GNN 教师模型准备

### 6.1 获取 PriorF-GNN 仓库

教师模型和预处理数据来自 PriorF-GNN 项目：

```bash
# 在 PriorF-Reasoner 同级目录克隆
cd ..
git clone https://github.com/ConmuYan/PriorF-GNN.git  # 或项目实际地址
cd PriorF-Reasoner
```

### 6.2 所需教师资产

从 PriorF-GNN 训练产出中需要以下文件：

```
PriorF-GNN/
  runs/rerun_20260404/
    amazon_full/seed_42/
      best_model.pt          # ~552 KB, Amazon 教师权重
      model_summary.json     # 模型配置摘要
    yelpchi_full/seed_42/
      best_model.pt          # ~1.5 MB, YelpChi 教师权重
      model_summary.json
  processed_data/
    amazon/seed_42/data.pt   # 预处理图数据（用于导出 evidence）
    yelpchi/seed_42/data.pt
```

> **如果无法获取 PriorF-GNN 训练产物**: 需要先运行 PriorF-GNN 的训练流程生成 `best_model.pt` 和 `processed_data/`。教师模型体积很小（<2 MB），也可从原服务器直接拷贝。

### 6.3 使用 setup_assets.sh 建立链接

```bash
# 设置 PriorF-GNN 路径（默认 ../PriorF-GNN）
export GNN_DIR=/path/to/PriorF-GNN

# 运行 setup 脚本创建符号链接
bash priorf_reasoner_slm/scripts/setup_assets.sh
```

如果 PriorF-GNN 路径不同，可逐个覆盖：

```bash
export AMAZON_MAT=/path/to/Amazon.mat
export YELPCHI_MAT=/path/to/YelpChi.mat
export AMAZON_TEACHER=/path/to/amazon_best_model.pt
export YELPCHI_TEACHER=/path/to/yelpchi_best_model.pt
bash priorf_reasoner_slm/scripts/setup_assets.sh
```

## 7. 导出教师 Evidence（teacher_exports）

这是训练前**必须完成**的步骤，从教师模型提取每个节点的结构化证据：

```bash
export GNN_DIR=/path/to/PriorF-GNN

# 导出 Amazon
DATASET=amazon SPLIT=train bash priorf_reasoner_slm/scripts/export_teacher_evidence.sh
DATASET=amazon SPLIT=test  bash priorf_reasoner_slm/scripts/export_teacher_evidence.sh

# 导出 YelpChi
DATASET=yelpchi SPLIT=train bash priorf_reasoner_slm/scripts/export_teacher_evidence.sh
DATASET=yelpchi SPLIT=test  bash priorf_reasoner_slm/scripts/export_teacher_evidence.sh
```

导出完成后验证：

```bash
ls -lh assets/teacher_exports/
# 预期文件:
#   amazon_train_evidence.parquet   (~XX MB)
#   amazon_test_evidence.parquet    (~XX MB)
#   yelpchi_train_evidence.parquet  (~XX MB)
#   yelpchi_test_evidence.parquet   (~XX MB)
```

> **备选**: 如果无法运行导出脚本（缺少 PriorF-GNN 预处理数据），可直接从原服务器拷贝 `assets/teacher_exports/` 目录（约 42 MB）。

## 8. 环境变量配置

训练和评估依赖以下环境变量，建议写入 `~/.bashrc` 或 `setup_env.sh`：

```bash
# GPU 选择
export CUDA_VISIBLE_DEVICES=0          # 根据 GPU 空闲情况设置

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 模型路径（修改为你实际的路径）
export MODEL_NAME=/path/to/Qwen3-4B-Instruct-2507

# PriorF-GNN 路径
export GNN_DIR=/path/to/PriorF-GNN

# conda 环境
conda activate priorfgnn
```

## 9. 验证安装

运行 smoke test 确认一切就绪：

```bash
bash priorf_reasoner_slm/scripts/run_smoke.sh
```

运行单元测试：

```bash
python -m pytest priorf_reasoner_slm/tests/ -q
# 预期: 183 tests passing
```

## 10. 快速迁移 Checklist

在新服务器上依次执行：

```
[ ] 1. git clone PriorF-Reasoner
[ ] 2. conda create -n priorfgnn python=3.10 && conda activate priorfgnn
[ ] 3. pip install torch==2.6.0+cu124, PyG, DGL, transformers, peft, trl, ...
[ ] 4. 下载 Amazon.mat + YelpChi.mat → datasets/
[ ] 5. 下载 Qwen3-4B → /path/to/models/
[ ] 6. 获取 PriorF-GNN 教师模型 (best_model.pt + model_summary.json)
[ ] 7. 获取 PriorF-GNN 预处理数据 (processed_data/)
[ ] 8. bash setup_assets.sh（建立符号链接）
[ ] 9. 导出 teacher evidence（或从原服务器拷贝）
[ ] 10. 运行 smoke test 验证
[ ] 11. python -m pytest priorf_reasoner_slm/tests/ -q（183 pass）
```

## 11. 可能遇到的问题

### Q: `torch_scatter` / `torch_sparse` 安装失败
确保 CUDA 版本和 PyTorch 版本匹配：
```bash
python -c "import torch; print(torch.version.cuda)"
# 输出应与安装时的 cuXXX 一致
```

### Q: 显存不足 (OOM)
- SFT/CoTrain 在 24 GB 显卡上 batch_size=4 可正常运行
- 如果仍然 OOM，减小 batch_size 到 2 并增加 `gradient_accumulation_steps` 到 8
- 确保 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已设置

### Q: `export_teacher_evidence.sh` 报错缺少 `lghgcl` 模块
该模块来自 PriorF-GNN 仓库，确保 `GNN_DIR` 指向 PriorF-GNN 根目录且其中包含 `lghgcl/` 包。

### Q: 找不到 `processed_data/*/seed_42/data.pt`
需要先运行 PriorF-GNN 的数据处理流程生成此文件。或从原服务器直接拷贝 `processed_data/` 目录（约 2.6 GB）。

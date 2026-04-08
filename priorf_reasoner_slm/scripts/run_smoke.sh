#!/bin/bash
# Minimal smoke test for PriorF-Reasoner training pipeline (no GPU, tiny model)
set -e

echo "===== PriorF-Reasoner Smoke Test ====="

# Check environment
echo "[1/5] Checking Python environment..."
python -c "import torch; print(f'  torch: {torch.__version__}, device: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import peft; print(f'  peft: {peft.__version__}')"
python -c "import trl; print(f'  trl: {trl.__version__}')"
echo "  PASS"

# Check imports
echo "[2/5] Checking module imports..."
python -c "from priorf_reasoner_slm.train.train_utils import get_device, format_log; print('  train_utils: OK')"
python -c "from priorf_reasoner_slm.train.train_sft import SFT_CONFIG, LORA_CONFIG; print('  train_sft: OK')"
python -c "from priorf_reasoner_slm.train.train_cotrain import COTRAIN_CONFIG, CoTrainer; print('  train_cotrain: OK')"
python -c "from priorf_reasoner_slm.eval.metrics import compute_auroc, compute_auprc; print('  eval.metrics: OK')"
python -c "from priorf_reasoner_slm.eval.eval_gen_only import evaluate_gen; print('  eval_gen_only: OK')"
python -c "from priorf_reasoner_slm.eval.eval_head_only import evaluate_head; print('  eval_head_only: OK')"
python -c "from priorf_reasoner_slm.eval.eval_fusion import evaluate_fusion, optimize_alpha; print('  eval_fusion: OK')"
python -c "from priorf_reasoner_slm.eval.faithfulness import sufficiency, comprehensiveness; print('  faithfulness: OK')"
echo "  PASS"

# Check config files
echo "[3/5] Checking config files..."
test -f priorf_reasoner_slm/configs/train/sft_config.yaml && echo "  sft_config.yaml: OK"
test -f priorf_reasoner_slm/configs/train/cotrain_config.yaml && echo "  cotrain_config.yaml: OK"
echo "  PASS"

# Check collators
echo "[4/5] Checking data collators..."
python -c "
from priorf_reasoner_slm.llm.collators import SFTDataCollator, CoTrainDataCollator
import torch

sft = SFTDataCollator(pad_token_id=0, ignore_index=-100)
cotrain = CoTrainDataCollator(pad_token_id=0, ignore_index=-100)

# SFT collator test
sft_batch = sft([{'input_ids': [1,2,3], 'labels': [-100,-100,4]}])
assert sft_batch['input_ids'].shape[0] == 1
assert sft_batch['labels'][0][0] == -100
print('  SFTDataCollator: OK')

# CoTrain collator test
cotrain_batch = cotrain([{'input_ids': [1,2,3], 'labels': [-100,-100,4], 'teacher_probs': 0.7, 'cls_targets': 1}])
assert 'teacher_probs' in cotrain_batch
assert 'cls_targets' in cotrain_batch
print('  CoTrainDataCollator: OK')
"
echo "  PASS"

# Check metrics
echo "[5/5] Checking eval metrics..."
python -c "
from priorf_reasoner_slm.eval.metrics import compute_auroc, compute_auprc, compute_f1, compute_gmeans
import numpy as np

y_true = [0,0,1,1]
y_score = [0.1, 0.4, 0.35, 0.8]
auroc = compute_auroc(y_true, y_score)
auprc = compute_auprc(y_true, y_score)
f1 = compute_f1(y_true, y_score)
gm = compute_gmeans(y_true, y_score)
print(f'  AUROC: {auroc:.4f}')
print(f'  AUPRC: {auprc:.4f}')
print(f'  F1: {f1:.4f}')
print(f'  G-means: {gm:.4f}')
"
echo "  PASS"

echo ""
echo "===== All smoke tests passed ====="

# Block 4 (llm/) Code Review

**Reviewer:** Codex Assistant (GPT-5.4)
**Date:** 2026-04-07
**Files:** tokenizer_utils.py, model_wrapper.py, cls_head.py, losses.py, collators.py, generation.py, fusion.py
**Tests:** test_llm_tokenizer.py (11), test_llm_losses.py (21), test_llm_collators.py (16) - 151 tests pass

---

## Summary

The llm/ block implements a tri-loss training pipeline for a Qwen3-4B student model with LoRA adapters, a binary classification head, and teacher distillation. Most components are correct, but **one correctness bug in training-mode label computation** was identified, plus a **likely-fatal LoRA target_modules mismatch** for Qwen3-4B that would prevent training from starting.

---

## 1. Tri-Loss Correctness

### gen_loss (losses.py:19-45) - CORRECT

The causal LM shift is properly implemented:
- `shift_logits = logits[:, :-1, :]` and `shift_labels = labels[:, 1:]` correctly align for next-token prediction.
- `ignore_index=-100` is used throughout. No issues found.

### cls_loss (losses.py:48-62) - CORRECT

- `BCEWithLogitsLoss` is used correctly with float targets.
- No issues found.

### distill_loss_kl (losses.py:65-102) - CORRECT with minor concern

The KL divergence direction is `F.kl_div(log_student, teacher_dist)` which computes KL(teacher || student) - the teacher is the target, student is the source. This is the standard distillation direction.

The temperature-squared scaling (`T**2` factor at line 102) follows established distillation practice (DistilBERT, etc.). Correct.

**Minor concern:** `torch.sigmoid(student_logits / temperature)` is used to build the 2-class student distribution. This works because `cls_logits` is (batch,) - a single logit per sample for binary classification. However, the sigmoid and softmax paths are inconsistent: `cls_loss` uses raw logits with `BCEWithLogitsLoss` (which applies sigmoid internally), while `distill_loss_kl` applies sigmoid manually before temperature scaling. This is not a bug, but worth documenting as the two losses operate on the same logits with different transformations.

---

## 2. Fusion Alpha Logic (fusion.py:15-75) - CORRECT

The fusion formula `alpha * cls_prob + (1-alpha) * teacher_prob` is correctly implemented.

**`optimize_alpha`** (lines 38-75) performs a grid search over [0, 1] maximizing accuracy. The logic is sound:
- Converts all inputs to tensors, iterates `n_steps+1` values.
- Thresholds fused probabilities at 0.5 and compares to binary labels.
- Returns optimal alpha and best accuracy.

---

## 3. LoRA Config Compatibility - BUG (CRITICAL)

**Issue:** The default LoRA `target_modules` in `model_wrapper.py:22` and `train_sft.py:55` are:

```python
"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
```

These are **NOT** the correct module names for Qwen3-4B. Qwen3-4B uses **Grouped Query Attention (GQA)** with separate A/B matrices for each projection. The actual module names are:

```
q_a, q_b, k_a, k_b, v_a, v_b, o_a, o_b
```

The string names `q_proj`, `k_proj`, etc. do not exist in Qwen3-4B's attention layers. When PEFT's `get_peft_model()` is called with non-existent target modules, it **silently proceeds** (in older PEFT versions) or raises a warning but continues (in newer versions), meaning **LoRA adapters are never actually applied to any layers**.

**Impact:** If running with default config, the model trains as a full fine-tune (all parameters trainable), not with LoRA. This defeats the purpose of LoRA and dramatically increases memory usage and training time.

**Detection:** Check `train_sft.py:88-95`:
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info("Applied LoRA: trainable=%d / total=%d (%.2f%%)", trainable, total, ...)
```

If trainable/total ratio is ~100% rather than the expected ~0.1-1% for LoRA, this confirms the bug.

**Fix:** Change `target_modules` to Qwen3-4B's actual attention projection names:
```python
"target_modules": ["q_a", "q_b", "k_a", "k_b", "v_a", "v_b", "o_a", "o_b"]
```

Note: `cotrain_config.yaml` and `sft_config.yaml` also have the same incorrect module names at lines 26-29 and 36-39.

---

## 4. Chat Template Correctness - BUG (MODERATE)

**Issue:** In `tokenizer_utils.py:142-174`, the training-mode label computation has a tokenization mismatch bug.

```python
# Line 148-152: Tokenize prompt with add_generation_prompt=True
prompt_text = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True,
)
# Line 154-158: Tokenize full text with add_generation_prompt=False
full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

# Line 160-161: Encode both texts
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
full_ids = tokenizer.encode(full_text, add_special_tokens=False)

# Line 164: Compute labels
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

**Problem:** `prompt_text` is the prompt WITH the generation prompt appended (`add_generation_prompt=True`). `full_text` is the complete text with assistant message but WITHOUT the generation prompt. These two texts have **different special token structures**, so `len(prompt_ids)` does not equal the prefix length of `full_ids`.

Example:
- `prompt_text` = ` "<|im_start|>system\n...\n<|im_start|>user\n...\n<|im_start|>assistant\n"` (ends with assistant start)
- `full_text` = `"<|im_start|>system\n...\n<|im_start|>user\n...\n<|im_start|>assistant\n/no_think\n{full_completion}<|im_end|>"`

When these are encoded, `len(prompt_ids)` does not equal the number of prompt tokens in `full_ids`. The slicing `full_ids[len(prompt_ids):]` could include or exclude wrong tokens.

**Correct approach:** Apply chat template ONCE on the full messages list with `tokenize=True` to get proper tokenization with labels, then manually set prompt tokens to -100:

```python
# Single tokenization with labels
full_encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,  # Returns dict with input_ids and labels
    add_generation_prompt=False,
)
input_ids = full_encoded["input_ids"]
labels = full_encoded["labels"]

# Now mask prompt tokens
prompt_messages = [m for m in messages if m["role"] != "assistant"]
prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
labels[:prompt_len] = [-100] * prompt_len
```

**Test coverage gap:** `test_tokenization.py:test_prompt_labels_are_masked` (line 121) only checks that `labels[0] == -100` and that "some" tokens are masked. It does NOT verify that the unmasked completion tokens actually correspond to the assistant message content. The bug would not be caught by this weak assertion.

---

## 5. Other Observations

### cls_head.py - CORRECT
`BinaryClsHead` is a simple, correct implementation. Takes last-token hidden state, applies dropout and linear projection to single logit. `predict_proba` correctly applies sigmoid.

### collators.py - CORRECT
Both `SFTDataCollator` and `CoTrainDataCollator` correctly pad sequences and build attention masks. Teacher probs and cls targets are properly handled as scalars.

### generation.py - CORRECT
`generate_prediction` correctly extracts generated tokens by slicing after `input_ids.shape[1]`. `parse_prediction_output` handles markdown code fences and partial JSON robustly. `batch_generate` is a straightforward mini-batch wrapper.

### fusion.py optimize_alpha - CORRECT with minor efficiency note
The grid search from 0 to 1 with `n_steps+1` iterations is O(n_steps). For 100 steps this is fine. Could be optimized with binary search but not necessary at this scale.

---

## Test Coverage Assessment

| Component | Test File | Tests | Coverage Quality |
|-----------|-----------|-------|------------------|
| losses.py | test_llm_losses.py | 21 | Good - covers gen_masking, BCE, KL, weighted sum |
| collators.py | test_llm_collators.py | 16 | Good - padding, masking, scalar fields |
| tokenizer_utils.py | test_llm_tokenizer.py | 11 | Weak for label alignment - see Section 4 |
| tokenizer (real) | test_tokenization.py | 7+ | Weak - only checks first/last tokens |

---

## Action Items

1. **[CRITICAL]** Fix LoRA `target_modules` for Qwen3-4B. Change from `["q_proj", "k_proj", "v_proj", "o_proj"]` to `["q_a", "q_b", "k_a", "k_b", "v_a", "v_b", "o_a", "o_b"]` in:
   - `priorf_reasoner_slm/llm/model_wrapper.py:22`
   - `priorf_reasoner_slm/train/train_sft.py:55`
   - `priorf_reasoner_slm/configs/train/sft_config.yaml:25-29`
   - `priorf_reasoner_slm/configs/train/cotrain_config.yaml:35-39`

2. **[MODERATE]** Fix `apply_chat_template` training label computation in `priorf_reasoner_slm/llm/tokenizer_utils.py:142-174` to use single-pass tokenization with proper prompt masking.

3. **[LOW]** Strengthen tokenizer tests to verify completion token alignment with ground truth assistant message.

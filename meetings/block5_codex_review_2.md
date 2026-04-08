# Block 5 Code Review: PriorF-Reasoner `train/` Directory

**Reviewer:** Codex (GPT-5.4)
**Files reviewed:**
- `priorf_reasoner_slm/train/train_sft.py`
- `priorf_reasoner_slm/train/train_cotrain.py`
- `priorf_reasoner_slm/train/train_utils.py`
- `priorf_reasoner_slm/configs/train/sft_config.yaml`
- `priorf_reasoner_slm/configs/train/cotrain_config.yaml`
- `priorf_reasoner_slm/scripts/run_smoke.sh`
- `priorf_reasoner_slm/scripts/run_train_qwen4b.sh`
- `priorf_reasoner_slm/llm/model_wrapper.py`
- `priorf_reasoner_slm/llm/losses.py`
- `priorf_reasoner_slm/llm/collators.py`
- `priorf_reasoner_slm/llm/cls_head.py`

---

## 1. LoRA Config — CRITICAL BUG

**Status: FAIL**

Both `train_sft.py` (line 55) and `model_wrapper.py` (line 22) define:

```python
"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**This is wrong for Qwen3-4B.** Qwen3-4B uses **Grouped Query Attention (GQA)**, where each attention layer has separate `q_a`/`q_b` (query grouped), `k_a`/`k_b` (key grouped), `v_a`/`v_b` (value grouped), and `o_a`/`o_b` (output) projection modules. The correct `target_modules` must be:

```python
"target_modules": ["q_a", "q_b", "k_a", "k_b", "v_a", "v_b", "o_a", "o_b"]
```

The current config targets the wrong module names (`q_proj`, etc.), so **no LoRA adapter will actually be applied** to Qwen3-4B's attention layers. The model will behave as a full fine-tune, defeating the purpose of LoRA and causing massive memory overhead.

The fix must be applied in three places:
- `priorf_reasoner_slm/llm/model_wrapper.py` line 22 (`DEFAULT_LORA_CONFIG`)
- `priorf_reasoner_slm/train/train_sft.py` line 55 (`LORA_CONFIG`)
- Both YAML config files: `sft_config.yaml` and `cotrain_config.yaml` (lines 26-29, 36-39)

**Severity: Critical** — LoRA is effectively bypassed.

---

## 2. Tri-Loss Weights lambda1=0.1, lambda2=0.1

**Status: PASS**

Tri-loss weights are correctly implemented and consistent across all locations:

| Location | lambda_cls | lambda_distill |
|---|---|---|
| `train_cotrain.py` lines 60-61 | 0.1 | 0.1 |
| `cotrain_config.yaml` lines 22-23 | 0.1 | 0.1 |
| `run_train_qwen4b.sh` lines 31-32 | 0.1 | 0.1 |
| `losses.py` `PriorFLoss.__init__` default | 1.0 (but overridden at call site) | 1.0 (but overridden at call site) |
| `CoTrainer.__init__` | passed from params | passed from params |

The `PriorFLoss` class defaults to 1.0 (lines 136-137), but the `CoTrainer` passes `lambda_cls=0.1` and `lambda_distill=0.1` at construction (lines 91-95, 264-269). This is correct.

The `train_cotrain()` function also accepts `lambda_cls` and `lambda_distill` params and forwards them correctly (lines 215-216, 267-268).

---

## 3. Gradient Accumulation Correctness

**Status: PASS (with minor logging improvement opportunity)**

The gradient accumulation implementation in `train_cotrain.py` is correct (lines 335-346):

```python
loss = losses["loss"] / gradient_accumulation_steps   # line 335
loss.backward()                                          # line 336

if (step + 1) % gradient_accumulation_steps == 0:      # line 338
    torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)    # line 339
    optimizer.step()                                     # line 343
    scheduler.step()                                     # line 344
    optimizer.zero_grad()                                # line 345
    global_step += 1                                     # line 346
```

- Loss is correctly scaled by `1/gradient_accumulation_steps` before backprop.
- Optimizer step fires exactly every `gradient_accumulation_steps`.
- Gradient clipping is applied before the step (correct order).
- Scheduler step is correctly aligned with optimizer steps.

**Minor note:** `total_steps` computation at line 304 is correct:
```python
total_steps = len(dataloader) * num_train_epochs // gradient_accumulation_steps
```
Integer division is appropriate here.

---

## 4. Checkpoint Saving

**Status: FAIL (SFT) / PARTIAL (Co-Training)**

### SFT (`train_sft.py`)
Uses TRL's `SFTTrainer` which handles checkpointing via `TrainingArguments`:
- `save_steps: 500` — intermediate checkpoints saved every 500 steps
- `save_total_limit: 2` — keeps only 2 checkpoints
- Final adapter saved via `trainer.save_model(str(final_path))` (line 181)

This is correct.

### Co-Training (`train_cotrain.py`)
**No intermediate checkpointing during training.** The training loop (lines 315-361) has no `save_steps` logic — only the final checkpoint is saved at the end (lines 373-386):

```python
final_path = output_dir / "final_cotrain"
final_path.mkdir(parents=True, exist_ok=True)

if isinstance(model, PeftModel):
    model.save_pretrained(str(final_path / "adapter"))
else:
    model.save_pretrained(str(final_path / "adapter"))

torch.save(cls_head.state_dict(), str(final_path / "cls_head.pt"))
```

This means if training crashes or is interrupted, **up to an entire epoch of work is lost**. Unlike SFT which uses `save_steps=500`, the co-training loop has no periodic checkpoint mechanism.

**Recommendation:** Add periodic checkpoint saving inside the epoch loop:
```python
if global_step > 0 and global_step % save_steps == 0:
    ckpt_path = output_dir / f"checkpoint-{global_step}"
    # save adapter + cls_head
```

---

## 5. Additional Findings

### HIDDEN_DIM Hardcoded to 2048
`train_cotrain.py` line 66 hardcodes `HIDDEN_DIM: int = 2048` for Qwen3-4B. Qwen3-4B's actual hidden dimension is 2048, so this is correct for the current model. However, this is not validated against the actual model config and would silently break if a different model variant were used.

### `model.eval()` After LoRA Wrapping
`model_wrapper.py` line 97 sets the model to `eval()` mode after applying LoRA. For training, this would suppress dropout and other training-specific behaviors. The `train_sft.py` correctly relies on SFTTrainer's internal mode management. However, `train_cotrain.py` calls `model.train()` at line 312, which is correct for the co-training loop.

### LoRA `bias` Setting
Both configs set `"bias": "none"`. This is standard and correct.

### `load_adapter` Device Handling
`train_utils.py` lines 74-78 uses `device_map={"": str(device)}` which is the PEFT-recommended way to load adapters onto a specific device.

---

## Summary

| Check | Status | Severity |
|---|---|---|
| LoRA config matches Qwen3 GQA | **FAIL** | Critical |
| Tri-loss weights lambda1=0.1, lambda2=0.1 | PASS | — |
| Gradient accumulation correctness | PASS | — |
| Checkpoint saving (SFT) | PASS | — |
| Checkpoint saving (Co-Training) | **FAIL** | Medium |

**Action Required:** Fix LoRA `target_modules` to use `["q_a", "q_b", "k_a", "k_b", "v_a", "v_b", "o_a", "o_b"]` before any training run. Add periodic checkpointing to the co-training loop.

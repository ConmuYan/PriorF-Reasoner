# Pending Debug

Last checked: 2026-04-07 (session continuation)

## All Critical Bugs Fixed (this session — verified with tests)

### Fix 1 — eval/metrics.py sufficiency_score (HIGH — Codex Block 6 Review 2)
- **Bug:** `sufficiency_score` used `mean(|original - ablated|)` — identical to comprehensiveness, measuring prediction shift (necessity semantics)
- **Fix:** `mean(max(min(original, ablated) - 0.5, 0)) * 2` — measures evidence sufficiency above chance level (0.5). Evidence at chance has no discriminative power regardless of prediction level.
- **Also updated:** `test_sufficiency_high` test case now uses `ablated = [0.8, 0.7, 0.6, 0.5]` (above chance) to test sufficiency correctly

### Fix 2 — eval/faithfulness.py evaluate_comprehensiveness (HIGH — Codex Block 6 Review 2)
- **Bug:** `fused_orig = teacher_prob * prob_orig + (1 - teacher_prob) * teacher_prob` always simplified to `teacher_prob`; `fused_ablated = teacher_prob`; comprehension = 0
- **Fix:** `fused_with = alpha * prob_cls + (1 - alpha) * teacher_prob; fused_without = alpha * prob_cls`
- **Also:** Added `cls_head: Any | None = None` parameter for API consistency (via linter)

### Fix 3 — eval/eval_fusion.py alpha optimization leakage (MEDIUM — Codex Block 6 Review 2)
- **Bug:** `optimize_alpha` and metric evaluation operated on same data
- **Fix:** Stratified 50/50 holdout split; `best_holdout_accuracy` for alpha optimization, metrics on held-out eval subset
- **Also updated:** `test_fusion_optimization` assertions now check `best_holdout_accuracy`

## Previously Resolved (prior session)

- Qwen3 GQA LoRA targets (`q_a/q_b/...`) in model_wrapper.py and configs
- graph_data validators, mat_loader union deduplication, split_manager CUDA-safe
- evidence_schema enums, feature_extractors fail-fast
- Teacher export hooks fusion layout assertion
- Co-training checkpoint saving (via linter)
- eval_head_only fail-fast (via linter)
- eval_gen_only failure counting (via linter)
- faithfulness API consistency with cls_head instance (via linter)

## Current Test Status

```
176 passed in ~10.6s
```

## Remaining Minor Items (lower priority)

1. **Block 4 tokenization:** The string-based `common_prefix` approach is the most robust method for real tokenizers. Token-level boundary finding fails on Qwen3 due to role metadata token placement differences between isolated vs conversation formats.
2. **Co-training checkpointing:** Already addressed in prior session per linter update.
3. **eval_gen_only parse fallback:** Already addressed in prior session per linter update.
4. **evaluate_head random cls_head:** Already addressed in prior session per linter update.

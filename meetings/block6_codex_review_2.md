# Block 6 Code Review: eval/ (170 tests pass)

**Reviewer:** Codex (GPT-5.4)
**Scope:** `priorf_reasoner_slm/eval/metrics.py`, `eval/eval_gen_only.py`, `eval/eval_head_only.py`, `eval/eval_fusion.py`, `eval/faithfulness.py`, `scripts/run_eval.sh`, `tests/test_eval.py`

---

## 1. Faithfulness Metric Formulas

### 1.1 Sufficiency — `metrics.sufficiency_score` (metrics.py:275-296)

```python
def sufficiency_score(original_scores, ablated_scores):
    return float(np.mean(np.abs(original - ablated)))
```

**Formula:** `score = mean(|original - ablated|)`

**Semantics:** Higher score = evidence is more sufficient (removing evidence causes larger prediction change). Lower score = evidence was NOT sufficient (removal barely changes prediction).

**Review finding — Semantic inversion bug:**
The docstring and the calling code in `eval/faithfulness.py` state the opposite relationship from the formula name. A *sufficient* piece of evidence should make the prediction **stable** when removed — if the evidence is truly driving the prediction, removing it should *not* change the prediction much. Yet the implementation returns a *higher* score when removal *does* change the prediction more (i.e., when the evidence was *necessary*, not sufficient).

- High `sufficiency_score` (large |orig - ablated|) means evidence removal *changes* predictions — evidence was *necessary*, not sufficient.
- Low `sufficiency_score` (small |orig - ablated|) means evidence removal *does not change* predictions — evidence was *sufficient*.

This is the **inverse** of what the metric name and docstring claim.

The academic sufficiency metric (from Whyy and Lies et al.) is defined as: does the explanation (evidence) sufficiently explain the prediction? If removing evidence *barely changes* the prediction, the original evidence was sufficient. So the correct formula should produce a **higher** score when `|orig - ablated|` is **small**. Either the formula needs to be inverted (e.g., `1 - mean(|orig - ablated|)`) or the semantics need to be clarified and the docstring rewritten to match the current implementation.

### 1.2 Comprehensiveness — `metrics.comprehensiveness_score` (metrics.py:299-320)

```python
def comprehensiveness_score(original_scores, ablated_scores):
    return float(np.mean(np.abs(original - ablated)))
```

**Formula:** `score = mean(|original - ablated|)`

**Same formula as sufficiency** — the two metrics are currently **identical** in implementation, differing only in which scores are passed in.

**Review finding — Formula is correct for comprehensiveness:**
The semantic claim is: higher score = teacher signal was more comprehensive (needed for prediction). If removing teacher signal *changes* predictions significantly (large |orig - ablated|), then the teacher signal was indeed comprehensive (necessary). This matches the formula. So comprehensiveness is implemented correctly, but sufficiency is not.

### 1.3 `evaluate_comprehensiveness` Bug (faithfulness.py:263-269)

```python
# original_scores: include teacher signal
prob_orig = cls_head.predict_proba(last_hidden).item()
fused_orig = teacher_prob * prob_orig + (1 - teacher_prob) * teacher_prob   # BUG
fused_ablated = 0.0 * prob_orig + (1 - 0.0) * teacher_prob  # = teacher_prob
```

The `fused_orig` formula is mathematically wrong. It computes:
```
fused_orig = teacher_prob * prob_orig + (1 - teacher_prob) * teacher_prob
           = teacher_prob * prob_orig + teacher_prob - teacher_prob^2
           = teacher_prob * (prob_orig + 1 - teacher_prob)
```

This does not correspond to any sensible fusion of cls_head and teacher. The correct fused prediction using the standard formula `p_final = alpha * p_cls + (1 - alpha) * p_teacher` would be `alpha * prob_orig + (1 - alpha) * teacher_prob`, but there is no alpha here — the code seems to be conflating the teacher's probability with the teacher's *weight* in the fusion.

Additionally, `fused_ablated = teacher_prob` is just the raw teacher probability, not a proper ablated version of the fused score. The entire computation from line 263-269 appears to be a mistaken reimplementation that doesn't measure comprehensiveness correctly.

### 1.4 `faithfulness_impact` Returns Redundant Keys (metrics.py:346-350)

```python
return {
    "sufficiency": sufficiency_score(original_scores, ablated_scores),
    "comprehensiveness": comprehensiveness_score(original_scores, ablated_scores),
    "prediction_flip_rate": flip_rate,
}
```

This function is called from `evaluate_sufficiency` (faithfulness.py:160) but the return dict overwrites the `sufficiency` and `comprehensiveness` keys — which are **identical formulas** (see 1.1 and 1.2). The `evaluate_sufficiency` function also separately calls `sufficiency()` which calls `sufficiency_score`, so `impact["sufficiency"]` is redundant. The same issue occurs in `evaluate_comprehensiveness`.

---

## 2. Alpha Optimization Correctness

### 2.1 `optimize_alpha` — Accuracy Metric (fusion.py:38-75)

```python
for step in range(n_steps + 1):
    alpha = step / n_steps
    fused = fuse_predictions(cls_probs, teacher_probs, alpha=alpha)
    preds = (fused >= 0.5).float()
    acc = (preds == val_labels).float().mean().item()
    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha
```

**Review finding — Fixed 0.5 threshold for accuracy is reasonable:**
Using accuracy (fraction of correct predictions at threshold=0.5) as the optimization objective is a valid choice for alpha selection. However:
- The `find_optimal_threshold` function (metrics.py:77-103) searches 100 thresholds from 0 to 1 for F1 maximization, but `optimize_alpha` uses a hardcoded 0.5 threshold. This means alpha is optimized for accuracy at threshold=0.5, while evaluation uses the F1-optimal threshold. There is a potential mismatch — alpha selected for 0.5-threshold accuracy may not be optimal for F1.
- No tie-breaking or epsilon smoothing: if multiple alphas yield the same best accuracy, the first one found is returned.

### 2.2 `evaluate_fusion_with_optimization` Uses Optimized Alpha for Final Metrics (eval_fusion.py:119-122)

```python
optimal_alpha, best_acc = optimize_alpha(...)
fused_probs = optimal_alpha * cls_probs_arr + (1.0 - optimal_alpha) * teacher_probs_arr
threshold = find_optimal_threshold(y_true, fused_probs.tolist())
metrics = compute_all_metrics(y_true, fused_probs.tolist(), threshold)
```

**The evaluation is sound.** The alpha is optimized on the same data used for evaluation (no held-out set for alpha optimization vs. metric reporting). In a proper pipeline, alpha should be optimized on a validation split and evaluated on a test split. The current code optimizes and evaluates on the same `teacher_df`, which means the reported metrics are optimistically biased. This is a **train-test leakage** issue in the alpha optimization pipeline.

### 2.3 `evaluate_fusion` Fixed Alpha — Threshold Optimization (eval_fusion.py:59-62)

```python
fused_probs = alpha * cls_probs_arr + (1.0 - alpha) * teacher_probs_arr
if threshold is None:
    threshold = find_optimal_threshold(y_true, fused_probs.tolist())
metrics = compute_all_metrics(y_true, fused_probs.tolist(), threshold)
```

**Correct.** The threshold is found on the same data as metrics, which is standard practice when reporting metrics post-hoc (not for model selection).

---

## 3. Eval Pipeline Completeness

### 3.1 Pipeline Stages (run_eval.sh)

The script runs 4 sequential stages:
1. **Generation quality** (`eval_gen_only`) — parse rate, format correctness, gen_auroc/auprc/f1
2. **Classification head** (`eval_head_only`) — head-only AUROC/AUPRC/F1 from hidden states
3. **Fusion** (`eval_fusion`) — fused AUROC/AUPRC/F1 with alpha optimization
4. **Faithfulness** (`eval/faithfulness.py`) — sufficiency and comprehensiveness

**Coverage is broad** — generation, head-only, fusion, and faithfulness are all evaluated.

### 3.2 Missing: No Joint Evaluation of All Metrics

The pipeline evaluates each component in isolation:
- Generation quality does not report the fusion metrics
- Head evaluation does not measure the generative component
- There is no single end-to-end evaluation that combines gen + head + fusion into one results dict

### 3.3 Missing: No Ground-Truth Label Consistency Check

None of the eval functions verify that the `label` column values are consistently 0/1 binary before computing metrics. If labels are strings ("fraud"/"benign") or floats (0.0/1.0), silent type coercion may occur via `int(row.get("label", 0))`.

### 3.4 `evaluate_gen` — Fallback to 0.5 for Failed Parsing (eval_gen_only.py:101-116)

```python
if parsed is None:
    generated_scores.append(0.5)  # neutral score
    continue
```

Using 0.5 for failed parses is a reasonable neutral fallback, but it means a model that always fails to parse gets a "neutral" score. This is masked from the parse_rate metric — but if the downstream fusion uses these scores, failed generations silently contribute 0.5. This should be logged more explicitly or flagged as a separate "gen_failure" category.

### 3.5 `evaluate_head` — Random cls_head If No State Dict (eval_head_only.py:67-70)

```python
cls_head = BinaryClsHead(hidden_dim=hidden_dim, dropout=0.0).to(device)
if cls_head_state_dict is not None:
    cls_head.load_state_dict(cls_head_state_dict)
```

If `cls_head_state_dict` is None, a **randomly initialized** cls_head is used. This makes the evaluation meaningless — it would evaluate a random head. The code should either raise an error or clearly document that a random head is being used for sanity checking only.

### 3.6 `evaluate_comprehensiveness` — API Inconsistency (faithfulness.py:176-210)

The function signature requires `cls_head_weights: dict[str, Any]` (a state dict), but `evaluate_sufficiency` does not need it — it only needs the model and tokenizer. The comprehensiveness evaluation also needs a loaded `BinaryClsHead` instance, creating an inconsistent API between the two main faithfulness functions.

### 3.7 Missing: No Tests for `evaluate_sufficiency` or `evaluate_comprehensiveness`

The test file (`tests/test_eval.py`) covers metrics computation, fusion evaluation, and CSV-based generation evaluation, but does **not** include tests for the sufficiency or comprehensiveness evaluation functions (`evaluate_sufficiency`, `evaluate_comprehensiveness`). These are the most complex functions in the faithfulness module with known bugs (Section 1.3).

---

## Summary of Issues

| Severity | Location | Issue |
|---|---|---|
| **HIGH** | faithfulness.py:263-269 | `fused_orig` formula is mathematically incorrect; comprehensiveness evaluation is broken |
| **HIGH** | metrics.py:275-296 | `sufficiency_score` formula is semantically inverted — high score means evidence was *necessary*, not *sufficient* |
| **MEDIUM** | eval_fusion.py:113-122 | Alpha optimization and metric evaluation on same data (train-test leakage) |
| **MEDIUM** | faithfulness.py:346-350 | `faithfulness_impact` returns identical `sufficiency` and `comprehensiveness` values (same formula) |
| **MEDIUM** | eval_head_only.py:67-70 | Random cls_head used when no state dict provided — silently evaluates random weights |
| **MEDIUM** | tests/test_eval.py | No tests for `evaluate_sufficiency`, `evaluate_comprehensiveness`, `evaluate_head`, or `evaluate_gen` |
| **LOW** | eval_gen_only.py:101-116 | Silent 0.5 fallback for failed generations masks gen failures in fusion input |
| **LOW** | faithfulness.py | `evaluate_sufficiency` and `evaluate_comprehensiveness` have inconsistent APIs |

---

## What Works Well

- `compute_all_metrics` is comprehensive and well-structured (AUROC, AUPRC, F1, precision, recall, specificity, G-means, threshold).
- `find_optimal_threshold` uses F1 maximization over 100 thresholds, which is a solid approach.
- The 4-stage eval pipeline in `run_eval.sh` is well-organized with clear step numbering and JSON output.
- `evaluate_fusion_from_csv` provides a convenient CSV-based entry point.
- `evaluate_alpha_sweep` is useful for analyzing the student-teacher tradeoff.
- The test suite has good coverage of metric correctness, fusion logic, and edge cases.
- `evidence_ablation_study` is a thoughtful addition for understanding per-field evidence impact.

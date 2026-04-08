# Block 4 Review: LLM Module Tests

**Date:** 2026-04-07
**Block:** 4 - LLM Module (tokenizer_utils, losses, collators)
**Status:** COMPLETE

## Summary

Successfully wrote and verified tests for all llm/ module files. All 151 tests pass.

## Tests Created

### 1. `test_llm_tokenizer.py` (11 tests)
- `TestLoadTokenizer`: 2 tests with mocked AutoTokenizer (no actual model download)
- `TestFormatEvidenceCard`: 5 tests for card formatting (inference/training modes, dict input)
- `TestApplyChatTemplate`: 5 tests for template application (input_ids/labels keys, masking, truncation)

### 2. `test_llm_losses.py` (21 tests)
- `TestGenLoss`: 4 tests (masked labels, all-valid, all-masked, shift correctness)
- `TestClsLoss`: 3 tests (BCE computation, perfect prediction, random)
- `TestDistillLossKL`: 4 tests (KL positivity, temperature scaling, clamping)
- `TestDistillLossMSE`: 3 tests (MSE positivity, identical vectors, batched)
- `TestPriorFLoss`: 7 tests (combined loss, weighted sum, default weights, component modes)

### 3. `test_llm_collators.py` (16 tests)
- `TestSFTDataCollator`: 7 tests (padding behavior, masking, attention, single sample)
- `TestCoTrainDataCollator`: 9 tests (teacher_probs, cls_targets, padding, type handling)

## Pytest Results

```
151 passed in 7.44s
```

All existing tests continue to pass (124 pre-existing + 27 new).

## Key Findings

1. **Mocking strategy**: `load_tokenizer` uses `@patch("priorf_reasoner_slm.llm.tokenizer_utils.AutoTokenizer")` to avoid actual model downloads during tests.

2. **Label masking**: `gen_loss` correctly uses `ignore_index=-100` and shifts logits/labels by 1 for causal LM.

3. **PriorFLoss**: Combined tri-loss properly returns dict with `loss`, `gen_loss`, `cls_loss`, `distill_loss`.

4. **Collators**: Both `SFTDataCollator` and `CoTrainDataCollator` correctly pad sequences and handle `teacher_probs`/`cls_targets`.

## Files Modified

- Created: `priorf_reasoner_slm/tests/test_llm_tokenizer.py`
- Created: `priorf_reasoner_slm/tests/test_llm_losses.py`
- Created: `priorf_reasoner_slm/tests/test_llm_collators.py`

## Next Steps

- Block 5: Training pipeline and configurations (train/)

# Block 2 Review: `priorf_reasoner_slm/priorf_teacher`

## Findings

1. High: The exported teacher table does not match the Evidence Card schema in `read.md`, so Block 2 is not producing the contract the rest of the pipeline expects. `read.md:156-180` defines a nested card with `dataset`, `teacher_summary`, `structure_evidence.relation_profile`, `structure_evidence.neighbor_stats`, and `task`. In contrast, `priorf_reasoner_slm/priorf_teacher/export_scores.py:185-202` emits a flat row with `label`, `teacher_logit`, `hsd`, `asda_switch`, raw `degree_*`/`disc_*` columns, and several extra neighbor statistics, but no `dataset`, no nested `teacher_summary`, and no `relation_profile` mapping from raw discrepancy values to low/high levels. A downstream serializer would need an extra adapter layer before the data can be used as written.

2. High: Hook cleanup is not exception-safe, so a failed export can leave hooks attached and cause duplicate captures on the next run. `priorf_reasoner_slm/priorf_teacher/export_scores.py:96-206` registers hooks, runs the model, builds neighbor stats, and only then calls `remove_hooks(handles)`. There is no `try/finally`, so any exception in the forward pass, quantile computation, or DataFrame assembly leaves the model instrumented. That is a correctness and memory-safety issue for repeated exports.

3. Medium: `compute_hsd_quantile` recomputes full-tensor quantiles for every node, which makes full YelpChi export much slower than necessary. In `priorf_reasoner_slm/priorf_teacher/export_scores.py:48-61`, each call does up to three `torch.quantile(hsd.float(), ...)` reductions. `export_teacher_evidence()` then calls it inside the per-node loop at `export_scores.py:179-183`, and it also recomputes the 95th percentile again for `high_hsd_flag`. For ~45k nodes this turns a simple threshold lookup into tens of thousands of whole-array reductions plus repeated host/device sync points. The output is likely correct, but the implementation is not scalable for the full-dataset export target.

4. Medium: Branch-logit extraction relies on an unverified fusion-layout assumption, so the pseudo-logits can silently be wrong on any teacher variant that does not concatenate branches in the exact expected order. `priorf_reasoner_slm/priorf_teacher/export_hooks.py:108-139` zero-pads one branch and feeds the other through `model.out`, assuming the head was trained on `[mlp_embedding, gnn_embedding]` with dimensions summing to `out.in_features`. There is no assertion that the model actually uses that ordering or that the embeddings already represent the post-BN/post-ReLU tensors the head saw during training. The current tests only cover a mock model with that exact layout, so the approximation is not yet validated against real teacher variants.

## Residual Risks

- `register_export_hooks()` only attaches to exact top-level attributes named `mlp`, `rgcn2`, and `asda.node_switch`. If a teacher checkpoint wraps those blocks differently, the export will silently miss evidence rather than failing fast.
- The current export path also serializes the raw `label` for every node. That may be intended for supervised training artifacts, but it should not flow into any prompt-time Evidence Card for test nodes.

## Verification

- I reviewed the implementation and schema text directly. I did not rerun the test suite in this pass.

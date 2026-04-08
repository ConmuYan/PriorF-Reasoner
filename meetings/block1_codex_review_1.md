# Block 1 Review: `priorf_reasoner_slm/graph_data`

## Findings

1. High: `validate_data` is not actually strict for graph symmetry, and it can crash on empty relations. In `priorf_reasoner_slm/graph_data/validators.py:93-123`, `ei.max()` is called unconditionally, so a relation with zero edges raises before validation can finish. Separately, the low-symmetry path only logs a warning when `strict=True` instead of raising `ValidationError`, so malformed graphs can still pass the validator in the mode that is supposed to be enforcing correctness.

2. Medium: `build_neighbor_stats` advertises metrics it does not compute, and its output shape is expensive for full-dataset evidence generation. In `priorf_reasoner_slm/graph_data/adjacency_builder.py:1-100`, the docstring promises top-k neighbor HSD discrepancy mean and same-relation discrepancy rank, but the implementation only returns degree, suspicious-neighbor ratio, per-relation discrepancy, and `topk_neighbors = min(top_k, degree)`. The implementation also materializes full NumPy arrays plus one nested Python dict per node, which will be unnecessarily heavy when generating Evidence Cards for all YelpChi/Amazon nodes.

3. Medium: `split_data` is a fragile API surface for anything beyond the current benchmark path. In `priorf_reasoner_slm/graph_data/split_manager.py:36-89`, `y.numpy()` assumes a CPU tensor, so callers passing a CUDA tensor will fail. The fallback paths also drop stratification entirely and can yield degenerate splits on tiny or highly imbalanced label distributions, which is exactly the kind of edge case a reusable data utility should guard more carefully.

## Compatibility Notes

- The new graph-data API is not drop-in compatible with PriorF-GNN’s loader conventions. PriorF-GNN uses `yelp` rather than `yelpchi`, lowercase relation names (`rur/rtr/rsr`), and appends HSD into `x` instead of keeping it separate. That is fine if this module is intentionally normalized for the student pipeline, but it should be treated as an adapter boundary rather than a direct mirror of the teacher pipeline.

## Verification

- `pytest -q priorf_reasoner_slm/tests/test_mat_loader.py` → 11 passed.

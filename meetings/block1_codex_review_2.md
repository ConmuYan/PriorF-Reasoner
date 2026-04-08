# Block 1 Review 2: `priorf_reasoner_slm/graph_data`

## Findings

1. High: `load_mat_dataset` still does not build a true union graph, so overlapping relations are counted multiple times and bias HSD. In [`priorf_reasoner_slm/graph_data/mat_loader.py:135-139`](/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner/priorf_reasoner_slm/graph_data/mat_loader.py#L135), `homo` is created with `torch.cat(all_edges, dim=1)`, which preserves duplicates across relations. The code and docstring describe `homo` as the union of all edges, but on any overlapping YelpChi/Amazon relation pair the same edge is counted two or three times in `compute_hsd`, changing the downstream score distribution.

2. Medium: `split_data` can still return an empty validation set on tiny inputs, so the new minimum-node assertion does not actually guarantee a three-way split. In [`priorf_reasoner_slm/graph_data/split_manager.py:35-69`](/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-Reasoner/priorf_reasoner_slm/graph_data/split_manager.py#L35), the fallback path clamps `val_size` to `len(remainder_idx) - 1`. For `num_nodes=3` the remainder has size 1, so `val_size` becomes 0 and `val_mask` stays empty. This is a silent regression for the stated API contract and for any small synthetic or filtered dataset use.

## Verified Fixes

- The empty-edge crash in `validators.py` is fixed, and strict symmetry violations now raise through `_check()`.
- `adjacency_builder.py` now uses batched tensor outputs instead of per-node Python dicts, which matches the updated docstring and removes the previous materialization bottleneck.
- `split_manager.py` now calls `.cpu()` before `.numpy()`, so CUDA labels no longer crash the split path.

## Verification

- The provided tests still pass: `11 passed`.

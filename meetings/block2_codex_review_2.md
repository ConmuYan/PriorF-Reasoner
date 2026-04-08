# Block 2 Review 2

## Findings

1. High: The export is still not schema-complete for the Evidence Card contract. `read.md:154-180` expects a nested card with `teacher_summary`, `structure_evidence.relation_profile`, `structure_evidence.neighbor_stats`, and `task`, plus the per-node export contract earlier in the doc also calls for `top_relations` and `neighbor_summary`. `export_teacher_evidence()` still emits a flat dataframe with scalar columns and per-relation `degree_*` / `disc_*` / `disc_level_*` fields at `priorf_reasoner_slm/priorf_teacher/export_scores.py:263-291`. The new `dataset` and `disc_level_*` columns are useful, but this block still cannot be consumed as the spec describes without another adapter layer.

2. Medium: The new discrepancy-threshold code is not device-robust. `build_neighbor_stats()` explicitly places its outputs on `x.device` (`priorf_reasoner_slm/graph_data/adjacency_builder.py:58-105`), but `export_teacher_evidence()` immediately calls `.numpy()` on `neighbor_stats.relation_discrepancy[rel_name]` while computing thresholds (`priorf_reasoner_slm/priorf_teacher/export_scores.py:163-170`). That will fail if the caller passes CUDA tensors, and it also breaks if those tensors still require grad. A detached CPU conversion is needed before percentile computation.

## Coverage Note

- The current tests validate helpers and hook behavior, but they still do not exercise `export_teacher_evidence()` end-to-end (`priorf_reasoner_slm/tests/test_teacher_export.py:80-148`). That leaves the schema shape, label masking, and cleanup path unverified.

# PriorF-Reasoner Assets

This directory holds the local assets needed by the Reasoner pipeline.

Expected layout:

```text
assets/
  data/
    Amazon.mat
    YelpChi.mat
  teacher/
    amazon/
      best_model.pt
      model_summary.json
    yelpchi/
      best_model.pt
      model_summary.json
  teacher_exports/
    amazon_train_evidence.parquet
    amazon_test_evidence.parquet
    yelpchi_train_evidence.parquet
    yelpchi_test_evidence.parquet
```

Use `priorf_reasoner_slm/scripts/setup_assets.sh` to create the recommended
symlinks from the existing `PriorF-GNN` outputs.

Use `priorf_reasoner_slm/scripts/export_teacher_evidence.sh` to generate
`assets/teacher_exports/*.parquet` from the linked teacher checkpoints and
`PriorF-GNN/processed_data/*/seed_42/data.pt` splits.

# Block 3 Review: `priorf_reasoner_slm/evidence`

## Findings

1. Medium: the schema does not actually enforce the categorical contract from `read.md`.
   - `RelationEntry.discrepancy_level`, `StructureEvidence.hsd_quantile`, `PredictionOutput.label`, and `PredictionOutput.pattern_hint` are plain `str` fields with no enum or literal validation.
   - In practice, invalid values are accepted by Pydantic, which means malformed teacher exports or template regressions can flow into SFT data without failing fast.
   - This weakens the JSON contract the student is supposed to learn and makes corruption harder to detect.
   - References: [`priorf_reasoner_slm/evidence/evidence_schema.py`](../priorf_reasoner_slm/evidence/evidence_schema.py)

2. Medium-high: relation extraction silently fabricates `low` discrepancy values when data is missing or malformed.
   - `extract_relation_profile()` defaults missing `disc_level_*` columns to `"low"` and also coerces any unknown value to `"low"`.
   - That means a partially exported or corrupted teacher table can still produce a valid-looking Evidence Card, but with evidence that is materially wrong.
   - For a training pipeline, this is a bad failure mode because it contaminates supervision instead of stopping the build.
   - References: [`priorf_reasoner_slm/evidence/feature_extractors.py`](../priorf_reasoner_slm/evidence/feature_extractors.py)

3. Low-medium: the rationale generator computes `high_disc_rels` but never uses it, so the generated rationale is less evidence-specific than the schema implies.
   - `generate_prediction()` collects the high-discrepancy relations, passes them into `build_rationale()`, and `build_rationale()` ignores the argument.
   - As a result, the text for camouflage/co-attack is generic and cannot mention the actual relations that were surfaced in the evidence list.
   - That is not a hard schema bug, but it is a faithfulness gap versus the M3 example in `read.md`.
   - References: [`priorf_reasoner_slm/evidence/rationale_templates.py`](../priorf_reasoner_slm/evidence/rationale_templates.py)

## Checks

- The existing evidence tests pass: `75 passed`.
- The current tests do cover empty relation profiles and round-trips, but they do not appear to cover malformed category values, missing `disc_level_*` columns, or NaN-like export rows.

## Data Leakage

- I did not find direct label leakage into the Evidence Card or completion JSON in the reviewed code paths.
- The main leakage risk is indirect: if the teacher export itself is contaminated or split filtering is skipped, the builders will happily serialize whatever rows they are given.

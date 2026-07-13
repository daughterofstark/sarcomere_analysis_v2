# Expert Feature Audit

This module audits whether existing automated patch features correspond to Natalia's blinded manual striation visibility, organisation score, and confidence annotations.

It is exploratory feature audit only. It does not tune thresholds, select a new production endpoint, modify feature tables, train a model, or make clinical claims.

## Inputs

- `results/validation/expert_annotation_validation/expert_annotation_validation_matched.csv`
- `results/tables/features_per_patch.csv`

Optional image-level feature tables may be inspected later, but the current audit is patch-level.

## Command

```bash
../sarcgraph-env/bin/python scripts/audit_expert_feature_relationships.py --config configs/default.yaml
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/audit_expert_feature_relationships.py \
  --config configs/default.yaml \
  --min-n 10 \
  --min-confidence 3
```

## What Is Audited

For the 75 expert-annotated patches, the module joins Natalia's manual endpoints to all available numeric or boolean automated patch descriptors.

Manual endpoints:

- `striations_visible`
- `organisation_score`
- `confidence_score`

Excluded from automated feature ranking:

- identifiers
- donor/image/patch IDs
- health or diagnosis fields
- manual/expert columns
- notes
- labels

Boolean automated flags are converted to 0/1.

## Outputs

```text
results/validation/expert_feature_audit/
```

Files:

- `expert_feature_audit_feature_table.csv`
- `expert_feature_audit_visibility_summary.csv`
- `expert_feature_audit_organisation_summary.csv`
- `expert_feature_audit_confidence_summary.csv`
- `expert_feature_audit_summary.json`
- `expert_feature_audit_summary.txt`

## Analyses

Visibility endpoint:

- median feature value for `yes`, `unclear`, and `no`
- simple `median_yes - median_no` separation
- Kruskal-Wallis p-value when group sizes are sufficient

Organisation endpoint:

- Spearman correlation between each automated feature and `organisation_score`
- confidence-filtered Spearman for rows with `confidence_score >= 3`

Confidence endpoint:

- Spearman correlation between each feature and `confidence_score`

## Interpretation

This audit helps answer whether OOP failed alone but another existing automated descriptor tracks expert judgement better.

It does not establish a new validated endpoint. Any candidate feature identified here still requires follow-up validation and clear biological interpretation.

Dominant orientation remains excluded as a primary validation endpoint because the expert reported ambiguity in how that field was interpreted.

Spacing remains `exploratory_low_yield` and is not validated here.

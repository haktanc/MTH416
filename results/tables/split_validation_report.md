# Stage 1 Split Validation

**Status: provisional. Identifiable exact-content and verified-group leakage checks pass; complete physical-part independence is not established.**

Training: {'total': 1040, 'class_counts': {'def_front': 625, 'ok_front': 415}, 'class_proportions': {'def_front': 0.6009615384615384, 'ok_front': 0.39903846153846156}}
Testing: {'total': 260, 'class_counts': {'def_front': 156, 'ok_front': 104}, 'class_proportions': {'def_front': 0.6, 'ok_front': 0.4}}
Holdout method: `stratified_image_split`; target 80/20; seed 42.
Five-fold method: `StratifiedKFold`; shuffle=True; random_state=42.
For each fold, indices refer only to train.csv rows. Validation membership is recorded in cv_folds.csv; the fit set is the remaining training rows. Every training image validates exactly once; no held-out test image is used.

Suspicious pairs: 3485; crossing holdout: 1019; crossing training CV folds: 1892.
These counts flag unresolved similarities, not confirmed leakage. See near_duplicate_partition_review.csv and near_duplicate_review.html.

Groups are connected components of exact byte/pixel matches plus any explicitly supplied verified source relationships. Generated group IDs are bookkeeping only. Group isolation takes priority over exact sizes or class proportions. When groups exist, 1,000 seeded group-shuffle candidates are ranked first by test-size error, then class-count error; no model outcomes are considered. Realized counts/proportions appear in stage1_validation.json. No image is dropped to improve balance.

## Validation

- complete_primary_coverage: True
- valid_manifest_paths: True
- disjoint_image_identities: True
- no_exact_content_crosses_holdout: True
- verified_groups_isolated: True
- no_exact_content_crosses_cv: True
- training_only_cv: True
- split_regeneration_identical: True
- cv_regeneration_identical: True
- Primary file membership and SHA-256 contents unchanged: True.

## Limitations and required review

- Unknown physical-part relationships are not ruled out; the split is provisional.
- Similarity candidates are retained and not automatically grouped or discarded.
- Shared filename counters are not verified physical-part identifiers.
- Resolve cross-boundary candidates before interpreting final test performance.
- Fit future scaling and feature selection inside training/CV pipelines only.
- Do not select models, features, thresholds, or hyperparameters using test outcomes.

Obtain acquisition records or supplier confirmation for source/part relationships. Record substantiated group assignments with evidence, then regenerate both holdout and CV assignments. Do not infer shared parts solely from similar impeller structure, hash distance, or numeric filenames. This stage does not verify visual class labels.

## Reproduction

Run `python -m src.prepare_splits` to regenerate and `python -m src.prepare_splits --validate-only` to re-audit and verify saved manifest/fold bytes against deterministic regeneration.

References: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

# Limited Visual Review Notes

This is a limited human-readable review record, separate from the generated
audit tables. It is not a completed review of all 3,485 candidate pairs.

## Pair inspected

- `data/raw/casting_512x512/ok_front/cast_ok_0_2816.jpeg`
- `data/raw/casting_512x512/ok_front/cast_ok_0_2989.jpeg`

Both original images were inspected visually. The automated screen reports
pHash distance 0/63 with image B mirrored, normalized 64 x 64 RMSE 0.031278,
and correlation 0.991225. Neither the file SHA-256 nor the decoded-RGB SHA-256
matches. These observations demonstrate why a pHash match is not an exact
duplicate finding.

The views share strong global geometry and similar visible structure. This
limited visual comparison does not establish whether they show the same
physical component, different captures, transformed source imagery, or distinct
similar components. Their lineage remains **unresolved**. No group assignment,
image exclusion, or visual class-label verification resulted from this review.

## Metadata and filename evidence

Across all 1,300 images, Pillow reports only the non-EXIF information keys
`jfif,jfif_density,jfif_unit,jfif_version`; no EXIF entries were found. No supplied
mapping connects these header fields or the numeric filenames to physical parts.

## Required follow-up

Use `near_duplicate_partition_review.html` to prioritize the 1,019 candidate
pairs across the holdout boundary and the 1,892 pairs across training CV folds.
These pair counts do not establish duplication or the number of distinct
physical components. Similar outlines alone cannot resolve these relationships.
Obtain acquisition logs or confirmation from the dataset provider where possible.
Record only substantiated source-group mappings, then regenerate both partitions
and folds. If provenance remains unavailable, retain the provisional designation
and state the independence assumption in any subsequent scientific report.

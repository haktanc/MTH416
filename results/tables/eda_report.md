# Stage 2: Exploratory Data Analysis and Preprocessing

Primary images: 1300; training: 1040; test: 260.
Images processed: 1300; test descriptive images: 260.

## Scope and methodology

All image examples, intensity comparison figures, and class effect sizes use training images only. The class-count plot uses manifest labels. Optional test pixel summaries are marked test_descriptive_only and never enter class comparisons or training figures. No preprocessing configuration was selected using test descriptors or model outcomes.

Grayscale decoding produces uint8 single-channel arrays in [0, 255]; it is not intensity normalization. The float32 processing input keeps native 512 x 512 dimensions by default and is not resampled. Only an explicit different target invokes resizing: area interpolation when both dimensions shrink, otherwise bilinear interpolation. Constant inputs are preserved exactly to avoid amplifying interpolation noise. The legacy CSV stage key 'resized' denotes this processing input, even when no spatial resampling is applied.

Configured size: 512 x 512. Spatial resizing applied: False. For each processing input r, m = min(r), M = max(r), and z = (r - m) / (M - m) when M > m; otherwise z = 0. Normalization preserves r's dimensions. The output is float32 in [0, 1]. These extrema belong to that image alone; there is no dataset-fitted scaler or transfer of test statistics.

Per-image pixel SD uses ddof=0. Pooled class SD includes within-image and between-image variation and is distinct from the average of within-image SDs. Between-image mean SD uses ddof=1. Histograms use 256 bins and probabilities summing to one; images contribute equal weight within each stage because stage dimensions are fixed.

Class mean/median differences and Cliff's delta use one observation per image. Delta = P(Defective > OK) - P(Defective < OK); ties contribute zero. Values are descriptive, with no p-values, confidence intervals, or claims of independent physical components.

## Manifest class distribution

| Subset | Class | Images | Percentage |
| --- | --- | ---: | ---: |
| all | ok_front | 519 | 39.9231% |
| all | def_front | 781 | 60.0769% |
| train | ok_front | 415 | 39.9038% |
| train | def_front | 625 | 60.0962% |
| test | ok_front | 104 | 40.0000% |
| test | def_front | 156 | 60.0000% |

## Training pixel summaries

| Stage | Class | Images | Pooled mean | Pooled SD | Minimum | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original | ok_front | 415 | 148.843053 | 63.130976 | 0.000000 | 255.000000 |
| original | def_front | 625 | 138.834889 | 60.762945 | 0.000000 | 255.000000 |
| resized | ok_front | 415 | 148.843053 | 63.130976 | 0.000000 | 255.000000 |
| resized | def_front | 625 | 138.834889 | 60.762945 | 0.000000 | 255.000000 |
| normalized | ok_front | 415 | 0.579783 | 0.250633 | 0.000000 | 1.000000 |
| normalized | def_front | 625 | 0.542676 | 0.239455 | 0.000000 | 1.000000 |

See eda_class_comparisons.csv for image-level differences and effects; eda_image_statistics.csv and eda_intensity_histograms.csv contain the underlying computed summaries. Test rows, when present, are separate.

## Reproducibility and validation

Sample seed: 42; samples per class: 3. Selection uses sorted training lists and a local NumPy generator; sample identities are stored in eda_selected_samples.csv.
All 1300 preprocessing checks passed: True. Constant original images: 0; constant resized images: 0.
Raw membership/content and manifest SHA-256 fingerprints were checked before/after processing. The saved split files were not regenerated. PNG figures are exported at 300 DPI with fixed display ranges for original/resized [0, 255] and normalized [0, 1] images.

## Remaining limitations

- Class comparisons are descriptive and use image-level observations, not independent pixels.
- Unknown physical-part relationships and Stage 1 provisional-split limitations remain.
- Default processing preserves native 512 x 512 dimensions; a different processing size must be requested explicitly.
- Test descriptors, if requested, must not inform feature or model selection.
- No HOG, GLCM, model training, or hyperparameter optimization was run.

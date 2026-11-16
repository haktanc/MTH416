# Stage 3: Native-resolution HOG and GLCM

CPU execution; all 1,300 images remain 512 x 512. Per-image Min-Max normalization; no fitted preprocessing.

## Training-only benchmark

50 deterministic training images (seed 42), shared by all methods; one untimed warm-up per isolated worker.
Descriptor time excludes decoding/normalization. Peak RSS is the worker's lifetime OS working-set high-water mark, including native libraries.

| Method | Dimensions | Feature seconds | ms/image | End-to-end seconds | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| hog8 | 142884 | 6.374860 | 127.4972 | 6.670564 | 54.21 |
| hog16 | 34596 | 3.111320 | 62.2264 | 3.378554 | 53.25 |
| glcm | 48 | 0.429755 | 8.5951 | 0.749234 | 70.81 |

## Cached features

| Split | Method | Shape | Payload MiB |
| --- | --- | --- | ---: |
| train | hog8 | 1040 x 142884 | 566.8616 |
| train | hog16 | 1040 x 34596 | 137.2522 |
| train | glcm | 1040 x 48 | 0.1904 |
| test | hog8 | 260 x 142884 | 141.7154 |
| test | hog16 | 260 x 34596 | 34.3130 |
| test | glcm | 260 x 48 | 0.0476 |

Extraction, writes and cache checks: 261.7265 seconds.
Measured workflow through preservation checks: 274.6433 seconds.

The HOG comparison shows local gradient structure at two cell scales; renderings are not feature vectors or classifier explanations.
GLCM matrix panels show distance-1 probabilities for all four angles. Property panels include both distances and all six statistics.
Caches are streamed float32 .npy files under ignored results/cache; indexes retain exact manifest row order and saved training validation folds.
No combined matrices are duplicated on disk. Load read-only memory maps with load_cached_features; fit future scalers inside training CV only.
File SHA-256, shape, dtype, index and label checks passed. Raw fingerprints, manifests and CV were unchanged.

## Limitations

- Resource measurements are single runs, not performance benchmarks across machines.
- Physical-part independence is unverified; perceptual candidates are advisory.
- Stage 4 must compare configurations using the existing training-only CV assignments.

## Visualization quality assurance

After visual inspection, HOG row spacing and pixels-per-cell labels were clarified, and GLCM properties now share one legend. Descriptor/pipeline code, benchmark numbers, all cache checksums and all sample property values are unchanged. Original measurement plotting provenance and final rendering provenance are both recorded in JSON. The rendering refresh took 7.0364 seconds.

## Final validation and tests

All 3,900 descriptor rows passed finite-value and descriptor/property-bound checks. Eighteen recomputed vectors matched cache rows exactly. Manifest ordering, labels and saved training CV folds were validated, and all four PNGs passed readability/300-DPI checks and visual review.

The final Stage 1-3 suite passed 151 tests in 30.26 seconds with `python -B -m pytest -q -p no:cacheprovider`. No classifier was fitted. Raw images, the existing 80/20 manifests and saved five-fold assignments remain unchanged. Eight local cache files, including labels and headers, occupy 923,147,924 bytes and remain excluded from Git.

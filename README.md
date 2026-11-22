# MAT 416 - Visual Data Analysis

## Surface Defect Detection in Casting Pump Impeller Images Using Texture Analysis and Machine Learning

This academic project investigates surface defect detection using classical
image processing, texture descriptors, and machine learning. The project
extracts HOG and GLCM descriptors and compares Support Vector Machine and
Random Forest baselines. Execution is local and CPU based.

Stages 1-3 provide dataset preparation, exploratory data analysis (EDA),
stateless grayscale preprocessing, and native-resolution feature extraction.
The saved manifests remain
**provisional**, because source and physical-part independence cannot be
established from the available data. HOG/GLCM resource benchmarks and feature
caches are available. Stage 4A measures fixed classifier baselines using
training-set cross-validation. Hyperparameter optimization and final holdout
evaluation are reserved for later stages.

## Dataset selection

Source: [Casting product image data for quality inspection on Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data).

The primary collection is `data/raw/casting_512x512/`, containing 1,300 original,
non-augmented 512 x 512 images:

| Class directory | Label | Expected images |
| --- | ---: | ---: |
| `ok_front` | 0 | 519 |
| `def_front` | 1 | 781 |

Defective images are the positive class. Although the images contain grayscale
content, the JPEG files have three components; the loader explicitly decodes
them to a two-dimensional grayscale array.

The 7,348-image augmented 300 x 300 collection remains at
`data/raw/casting_data/casting_data/` for possible future experiments. A prior
audit of the local copy found 64 exact duplicate pairs across that collection's
existing training and testing sets. This finding has not been independently
confirmed against the original Kaggle archive. The augmented collection is
not selected by the default entry point.

Raw images remain in their original class directories. Training and test
membership is stored only in repository-relative CSV manifests. The source
describes this collection as original and non-augmented; capture lineage cannot
be independently established from the files' names or metadata.

## Project architecture

```text
MTH416/
|-- data/
|   |-- raw/                         # Local only; excluded from Git
|   |   |-- casting_512x512/
|   |   |   |-- def_front/
|   |   |   `-- ok_front/
|   |   `-- casting_data/
|   |       `-- casting_data/
|   |           |-- train/{def_front,ok_front}/
|   |           `-- test/{def_front,ok_front}/
|   `-- splits/                     # Trackable; no copied images
|       |-- .gitkeep
|       |-- train.csv
|       `-- test.csv
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- dataset_audit.py
|   |-- eda.py
|   |-- prepare_splits.py
|   |-- reporting.py
|   |-- templates/pair_review.html
|   |-- preprocessing.py
|   |-- feature_extraction.py
|   |-- feature_pipeline.py
|   |-- model_training.py
|   |-- evaluation.py
|   `-- visualization.py
|-- results/
|   |-- figures/.gitkeep             # PNG figures; trackable
|   |-- tables/                     # Audit, CV, and validation reports
|   |-- cache/features/             # Large local .npy feature caches; ignored
|   `-- models/.gitkeep              # Generated binaries ignored
|-- tests/
|   |-- __init__.py
|   |-- test_initialization.py
|   |-- test_dataset_audit.py
|   |-- test_data_splitting.py
|   |-- test_preprocessing.py
|   |-- test_eda.py
|   |-- test_feature_extraction.py
|   |-- test_feature_pipeline.py
|   `-- test_pipeline.py             # Existing analysis tests; deferred
|-- notebooks/                      # Existing empty directory; unused
|-- main.py
|-- pytest.ini
|-- requirements-init.txt
|-- requirements-stage1.txt
|-- requirements-stage2.txt
|-- requirements-stage3.txt
|-- requirements.txt
|-- README.md
`-- .gitignore
```

The local `.venv/` and Git metadata are omitted from the diagram. No notebooks,
Jupyter dependencies, Colab services, or cloud storage are used.

| Module | Responsibility |
| --- | --- |
| `data_loader.py` | Primary dataset paths, labels, file indexing, grayscale decoding, and strict manifest reading; existing augmented-set helpers are retained. |
| `dataset_audit.py` | Full decoding, format/dimension/grayscale checks, metadata observations, SHA-256 and perceptual similarity screening, and review reports. |
| `prepare_splits.py` | Seeded holdout manifests, evidenced grouping, training-only CV indices, and leakage/reproducibility validation. |
| `eda.py` | Training-focused image statistics, deterministic samples, descriptive class comparisons, optional isolated test summaries, and Stage 2 output generation. |
| `reporting.py` | Deterministic CSV/JSON serialization and raw-directory output protection. |
| `preprocessing.py` | Validated grayscale arrays, configurable resizing, and safe per-image Min-Max normalization with float32 outputs. |
| `feature_extraction.py` | Configurable HOG, GLCM, and combined feature vectors. |
| `feature_pipeline.py` | Training-only benchmarks, resource gates, streaming split-specific caches, provenance and read-only loading. |
| `model_training.py` | CPU classifier construction, fixed Stage 4A pipelines, and separate tuning and serialization helpers. |
| `baseline_experiment.py` | Stage 4A cache alignment, saved-fold evaluation, resource checks, and result serialization. |
| `baseline_scaling.py` | StandardScaler with bounded fit-statistics batches. |
| `baseline_figures.py` | Training CV comparisons with across-fold variability. |
| `evaluation.py` | Accuracy, precision, recall, F1, confusion matrices, ROC-AUC, and comparison tables. |
| `visualization.py` | Publication-quality EDA/preprocessing PNGs and preserved helpers for later texture/model plots. |
| `main.py` | Lightweight initialization and primary dataset accessibility check. |

Existing preprocessing, feature, training, evaluation, visualization functions,
and analysis tests were preserved. Their presence does not mean that scientific
validation or experiments have been completed. Optional analysis libraries load
only when those functions are explicitly called.

Both manifests use the columns `image_path,label,class_name,split,image_id`.
Paths use forward slashes and are relative to the repository. `image_id` is the
SHA-256 of that relative path; it identifies a file entry, **not a physical
part**. The reader validates path confinement, existing files, class/label
agreement, split membership, unique entries, identities, and deterministic order.

## Local Python environment

The Stage 1-4A environment uses Python 3.14.7 on Windows.
Use the existing `.venv/`. Only if it is absent, create it from an installed
Python interpreter:

```powershell
python -m venv .venv
```

Install the pinned Stage 3 dependencies and run the project from the
repository root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage3.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest -q
```

Alternatively, activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

If activation is restricted by the local PowerShell policy, use the explicit
interpreter commands above. In VS Code, select `.venv\Scripts\python.exe`
using **Python: Select Interpreter**.

Startup checks the expected class directories and file counts and decodes one
512 x 512 sample per class. It prints a real filesystem summary, returns a
nonzero exit status on failure, and writes no dataset, split, or result files.
This accessibility check is not a full image integrity or duplicate audit.
The default path is anchored to the source location, not the current directory.
A custom primary dataset root can be supplied explicitly:

```powershell
.\.venv\Scripts\python.exe main.py --data-dir "D:\datasets\casting_512x512"
```

Initialization requires NumPy and OpenCV; pytest supports verification.
`requirements-init.txt` pins that minimal environment. `requirements-stage1.txt`
adds Pillow for image/metadata auditing and scikit-learn plus its dependencies
for splitting and CV. Their installation does not run machine learning models.
`requirements-stage2.txt` adds Matplotlib and its plotting dependencies.
`requirements-stage3.txt` adds scikit-image 0.26.0 and pins its additional
dependencies (networkx, imageio, tifffile and lazy-loader), preserving all
Stage 2 pins. Dependency compatibility was checked with `pip check`.
`requirements.txt` includes Stage 3 plus optional planned pandas and seaborn;
those two packages remain deferred and are not required by Stages 1-3.
Before later analysis work, install and validate the full environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The full analysis dependency set is not yet locked or runtime-verified.

## Stage 1 audit findings

All **1,300 images** fully decode with Pillow and the existing OpenCV grayscale
loader. They are JPEG files with 512 x 512 dimensions, RGB encoding with three
channels, and equal R/G/B values at every pixel. No unreadable image, unexpected
dimension, non-grayscale content, or missing-count discrepancy was found.

- Exact file SHA-256 duplicate pairs: **0**.
- Additional canonical decoded-RGB SHA-256 duplicate pairs: **0**.
- Conservative perceptual-screen candidates: **3,485 pairs**, involving
  **1,107 images**; **264 pairs** have different structural class labels.
- Images with EXIF metadata: **0**.
- Names follow `cast_def_0_<number>` or `cast_ok_0_<number>`. The common `0`
  counter and numeric suffixes are not documented source or physical-part IDs.
  There are 38 repeated numeric suffixes across the two class prefixes; these
  coincidences do not establish a relationship.

The similarity screen compares every unordered pair using a 63-bit DCT hash:
32 x 32 downsampling, the low-frequency 8 x 8 block with DC removed, and median
thresholding. A Hamming distance of at most **4** under any of eight rotations
and reflections flags a pair. Aligned 64 x 64 normalized RMSE and correlation
are reported as review aids. The cutoff is a declared screening rule, not a
validated identity classifier. Industrial parts share strong global structure;
**a candidate is not a confirmed near-duplicate or a verified shared part**.
False positives and missed relationships are both possible.

Review [audit_report.md](results/tables/audit_report.md),
[dataset_audit.csv](results/tables/dataset_audit.csv),
[near_duplicate_candidates.csv](results/tables/near_duplicate_candidates.csv),
and [near_duplicate_review.html](results/tables/near_duplicate_review.html).
Open the HTML locally to see pairs side by side; it references the original
images without copying them. Visual class-label correctness remains unverified.
One candidate pair was inspected visually and remains unresolved; see
[visual_review_notes.md](results/tables/visual_review_notes.md). This limited
review does not clear the remaining candidates.

## Holdout split and leakage limitations

No reliable source/physical-part groups were established. All exact-content
groups are singletons. Therefore the current split uses image-level stratified
sampling with **80% training, 20% testing, random_state=42**, and deterministic
path sorting before and after allocation. Treating image entries as independent
is an explicit, unverified assumption, so this is a **provisional split**.

| Partition | Defective | OK | Total | Defective proportion |
| --- | ---: | ---: | ---: | ---: |
| Original | 781 | 519 | 1300 | 60.08% |
| Training | 625 | 415 | 1040 | 60.10% |
| Test | 156 | 104 | 260 | 60.00% |

No images are discarded. Paths, identities, and SHA-256 contents are disjoint
across the holdout boundary. Exact-byte and exact-pixel groups, plus any
explicitly evidenced source groups supplied later, must stay together.
The code uses a group-aware strategy whenever a non-singleton verified group
exists: 1,000 seeded group-shuffle candidates are ranked by test-size error,
then class-count error, with group isolation taking priority. Exact 80/20 sizes
or stratification may then be unattainable; actual proportions are reported.

**1,019 unresolved similarity candidates cross the holdout boundary.** Another
**1,892 candidate pairs cross the training CV validation-fold assignments**.
These observations are not proven leakage, but prevent a claim of complete
leakage safety. See
[near_duplicate_partition_review.csv](results/tables/near_duplicate_partition_review.csv)
and the [interactive pair review](results/tables/near_duplicate_partition_review.html).
The offline HTML provides boundary/class filters, pagination, and optional
display-only alignment of image B; clicking an image opens its original file.
Do not remove or group these images automatically just because their impeller
outlines match. Review the pairs and, where possible, obtain acquisition records
or confirmation from the dataset provider. Do not use model/test performance
to decide which split or relationship interpretation to prefer.

Optional substantiated groups can be supplied using
`--verified-groups data/splits/verified_groups.csv`. The file must use this
header and one row per evidenced image assignment:

```text
image_path,group_id,evidence
```

Paths must refer to audited primary images. Group IDs and evidence must be
nonempty; paths must be unique. Evidence should cite an actual acquisition
record or confirmed source relationship. The software can validate the mapping
structure but cannot authenticate the evidence. Perceptual scores, filename
suffixes, or the common counter alone are insufficient. Do not invent physical
identifiers. Rerun preparation after verified assignments are available.

## Five-fold cross-validation

The current configuration is **StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)** on the **1,040 training rows only**. Each fold has:

| Fold partition | Defective | OK | Total |
| --- | ---: | ---: | ---: |
| Fit | 500 | 332 | 832 |
| Validation | 125 | 83 | 208 |

Every training image validates exactly once. The test set never enters any
fold. [cv_folds.csv](results/tables/cv_folds.csv) records validation membership
using fold numbers 1-5; a fold's fit set is the other training rows. When
verified groups exist in training, the implementation uses
`StratifiedGroupKFold` and checks actual class distributions and group isolation.
Preparation fails if valid class-containing folds cannot be constructed.

`generate_cv_indices` returns zero-based positional indices into the sorted
training rows. Example for future orchestration, without fitting a model:

```python
from src.data_loader import PROJECT_ROOT, read_manifest
from src.dataset_audit import audit_dataset
from src.prepare_splits import independence_groups, generate_cv_indices

audit = audit_dataset()
groups, _ = independence_groups(audit)  # Supply the same evidence file if used.
train = read_manifest(PROJECT_ROOT / "data/splits/train.csv", "train")
test = read_manifest(PROJECT_ROOT / "data/splits/test.csv", "test")
folds, strategy = generate_cv_indices(train, test, audit, groups)
```

Future model orchestration must use these validated folds with the same row
order and group configuration. Do not blindly reuse an image-only splitter
after verified groups have been introduced. Fit learned scaling and feature
selection **inside the training/CV pipeline**. Never fit statistics on test
images or use held-out test outcomes for model/hyperparameter selection.

## Regeneration, validation, and tests

Run from the repository root with the pinned Stage 1 environment:

```powershell
# Read-only image audit with generated reports.
.\.venv\Scripts\python.exe -m src.dataset_audit

# Re-audit and regenerate train/test manifests and CV assignments.
.\.venv\Scripts\python.exe -m src.prepare_splits

# Re-audit and compare saved manifests/folds with exact regeneration.
.\.venv\Scripts\python.exe -m src.prepare_splits --validate-only

# Initialization and Stage 1 tests only; no feature extraction or model fitting.
.\.venv\Scripts\python.exe -m pytest -q tests/test_initialization.py tests/test_dataset_audit.py tests/test_data_splitting.py
```

Preparation checks complete coverage, valid repository-relative paths, class
labels, disjoint identities and exact contents, verified-group isolation,
training-only CV, exactly-once validation membership, and deterministic
regeneration. It rehashes primary raw files and verifies membership against the
initial audit. Saved CSVs use UTF-8 and LF newlines. Reproduction depends on
unchanged raw files and the recorded package versions and parameters.

Meaningful artifacts under `results/tables/` include:

- `dataset_audit.csv`, `audit_summary.json`, and `audit_report.md`.
- `exact_duplicates.csv` and `near_duplicate_candidates.csv`.
- `near_duplicate_review.html` and `near_duplicate_partition_review.csv/.html`.
- `independence_groups.csv` (bookkeeping groups, not invented part IDs).
- `cv_folds.csv`, `split_config.json`, and `stage1_validation.json`.
- `split_validation_report.md` explaining checks, assumptions, and limitations.
- `raw_preservation.json` recording the before/after fingerprint of all 8,648
  files across both raw collections; the augmented collection was fingerprinted
  only to verify preservation, not used for primary audit/split decisions.
- `visual_review_notes.md`, a limited review record not overwritten by scripts.

`pytest.ini` selects initialization, audit, split, preprocessing, EDA, and feature tests.
They use temporary
fixtures and exercise corruption, exact byte/pixel duplicates, metadata
differences, group isolation, duplicate-leakage rejection, manifest tampering,
training-only CV, raw-file preservation, and full regeneration. The existing
`test_pipeline.py` analysis suite remains deferred. The legacy
`main.inspect_dataset` function is preserved; its feature-extracting preview
mode is not exposed by the initialization CLI.

**Readiness:** infrastructure and identifiable exact-content checks are ready
for provisional development. Reliable final performance claims require review
of unresolved similarities and better source/physical-part provenance. Neither
the absence of exact duplicates nor passing validation proves independence of
unknown physical parts. Descriptor correctness is validated in Stage 3;
classification performance evaluation remains future work.

## Stage 2: EDA and preprocessing

**Resolution policy: preserve the original 512 x 512 dimensions by default.**
Spatial resizing is optional and requires an explicit different target.
Min-Max normalization changes intensity values, never image dimensions.

**Verified output status (2026-11-16):** the seven Stage 2 PNGs and nine
table/report files were regenerated from all **1,300 original images at
512 x 512**. These outputs replace the historical 128 x 128 run. The current
run summary records `processing_height = processing_width = 512`,
`resizing_applied = false`, and no interpolation. The original, float32
processing-input, and normalized arrays all retain 512 x 512 dimensions.

The existing train/test manifests are read and validated; they are never
regenerated by Stage 2. All image-level class comparisons and image figures use
the **1,040 training images**. The class-distribution figure also includes
manifest-only counts for the full collection and test set. No model, feature,
or hyperparameter is selected in this stage.

The default command does not decode test images for descriptive statistics.
Optional test descriptions use the identical preset transform and are marked
`test_descriptive_only`; they never enter training comparisons or image figures.
The verified native-resolution run included those separate descriptors:
**1,040 training images and 260 test images**, each at 512 x 512. Test summaries
must not be used to choose later preprocessing, features, models, or
hyperparameters. The existing 80/20 manifests and saved five-fold CV assignments
were preserved byte for byte.

### Grayscale decoding and intensity normalization

`load_grayscale` decodes each JPEG with OpenCV into a single-channel
`uint8` array with intensity units in `[0, 255]`. This decoding step is distinct
from normalization. The JPEG files contain grayscale content but are encoded
with three components; decoding removes that channel redundancy.

`prepare_image(image)` preserves the input dimensions and applies per-image
normalization directly. For project images its output is **512 x 512**.
`resize_grayscale(image)` without a target returns float32 pixels at the same
dimensions and in the original intensity units. Neither default invokes spatial
resampling. An explicit matching target also bypasses resampling.

The EDA API and CLI share `DEFAULT_IMAGE_SIZE = (512, 512)`. A different
**(height, width)** can be passed explicitly when a separate experiment requires
resizing. Only then is area interpolation used when both dimensions shrink,
otherwise bilinear interpolation. Floating-point resizing avoids premature
rounding to integer intensity levels. No 128 x 128 or 256 x 256 processing
default is used.

Let `r` be the grayscale processing input at native resolution (or at an
explicitly requested alternative size), `m = min(r)`, and `M = max(r)`.
Per-image Min-Max normalization is:

$$
z_{ij} =
\begin{cases}
\dfrac{r_{ij} - m}{M - m}, & M > m, \\
0, & M = m.
\end{cases}
$$

The output is a two-dimensional **float32** array in **[0, 1]**, with the
same dimensions as `r`. Intermediate normalization arithmetic uses float64.
Constant images map to zero without division by zero, and resizing preserves
constant inputs exactly so round-off cannot be amplified into artificial
contrast. Nonfinite values, color arrays, empty arrays, invalid dimensions,
and unsupported dtypes are rejected. The functions do not mutate their inputs.

Each image supplies only its own minimum and maximum. There is no dataset-fitted
normalizer or fitted preprocessing state, and no test-set statistic is reused
for training. A future learned scaler or feature selector must still be fitted
inside training/CV pipelines, as described above.

### Statistics and scientific interpretation

The workflow computes actual mean, population standard deviation (`ddof=0`),
minimum, and maximum for every processed image at three stages: original,
processing input, and normalized. The legacy CSV key `resized` identifies the
float32 processing-input stage; at the 512 x 512 default it involves no spatial
resampling. New run summaries explicitly record `resizing_applied`.
Class tables distinguish pooled pixel standard
deviation, average within-image standard deviation, and between-image mean
standard deviation (`ddof=1`). A single-image group has no sample SD; that field
is left empty rather than replaced with a fabricated value.

Histograms use 256 bins per stage and are normalized to pixel probabilities,
which sum to one. This allows before/after comparisons despite differing pixel
counts. Original/resized axes use `[0, 255]` intensity units; normalized axes use
`[0, 1]`. Image displays also use fixed ranges, rather than allowing Matplotlib
to silently rescale each image.

Training class comparisons use one observation per image: mean and median
differences (Defective minus OK), and observed Cliff's delta:

$$
\delta = P(X_{\mathrm{Defective}} > X_{\mathrm{OK}})
       - P(X_{\mathrm{Defective}} < X_{\mathrm{OK}}).
$$

Ties contribute zero. These are descriptive effects, with no p-values or
claims of independent physical parts. Full-image intensity summaries include
the background and do not by themselves establish defect severity or model
performance. Stage 1 source/physical-part uncertainties remain unchanged.

Three samples per class are selected without replacement from sorted training
entries using a local NumPy generator with **seed 42**. Their paths and ranks
are recorded so representative images are reproducible. Selection is random,
not a manual choice of visually persuasive examples.

### Running Stage 2

```powershell
# Training-focused EDA; test pixels are excluded from descriptive analysis.
.\.venv\Scripts\python.exe -m src.eda --seed 42

# Reproduce the verified native-resolution run, with separate test descriptors.
.\.venv\Scripts\python.exe -B -m src.eda --size 512 512 --seed 42 --include-test-descriptive

# Explicit opt-in only: keep any non-native experiment in a separate directory.
.\.venv\Scripts\python.exe -m src.eda --size 384 400 --seed 42 --output-dir results/optional_384x400

# Stage 1-2 tests only, without HOG/GLCM extraction or model training.
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_initialization.py tests/test_dataset_audit.py tests/test_data_splitting.py tests/test_preprocessing.py tests/test_eda.py
```

The optional `--samples-per-class` accepts 1-6 samples for compact figures.
`--output-dir results/another_run` writes a separate set of figure/table outputs
under the results tree. Repeating a command refreshes its EDA outputs only;
Stage 1 audit reports and manifests are preserved. Parameters and package
versions are recorded in `eda_run_summary.json`. Source-image and manifest
SHA-256 fingerprints are checked before/after processing.

### Verified native-resolution findings

The collection contains 781 defective images (60.0769%) and 519 OK images
(39.9231%). Training contains 625 defective and 415 OK images; test contains
156 defective and 104 OK images. Every saved CV validation fold still contains
208 training images (125 defective, 83 OK), with 832 remaining fit images.

The following statistics use **training images only**. Original intensities
use [0, 255] units; normalized intensities use [0, 1]. Pooled pixel SD includes
both within-image and between-image variation. The mean within-image SD is a
different quantity and is reported separately.

| Stage | Class | Pooled mean | Pooled pixel SD | Mean within-image SD |
| --- | --- | ---: | ---: | ---: |
| Original | OK | 148.843053 | 63.130976 | 62.364518 |
| Original | Defective | 138.834889 | 60.762945 | 59.852658 |
| Normalized | OK | 0.579783 | 0.250633 | 0.247704 |
| Normalized | Defective | 0.542676 | 0.239455 | 0.236015 |

The defective-minus-OK difference in image mean is -10.008164 before
normalization and -0.037107 after normalization. The corresponding observed
Cliff's deltas are -0.579104 and -0.566558. For within-image SD, the deltas are
-0.354572 before normalization and -0.398724 after normalization. Defective
training images therefore tend to have lower full-image mean intensity and
lower within-image SD in this collection, with overlapping class distributions.
These are descriptive associations: background, illumination, and acquisition
conditions can contribute. They do not establish defect severity, causal
effects, statistical significance, or future classifier performance.

The original and processing-input statistics and histograms are identical
because no spatial resampling occurred. Normalization rescales each image's
intensities while preserving spatial detail; it does not make the class
histograms identical. Fine spikes or empty bins in normalized histograms can
arise when discrete uint8 intensity levels are remapped into 256 bins.

All **1,300 preprocessing validations passed**: 512 x 512 output, float32,
finite values, minimum 0 and maximum 1. No constant-intensity source images
were found; constant-image behavior is covered by automated tests. Additional
checks compared saved normalized means/SDs with the per-image Min-Max formula,
verified all histogram pixel totals/probability sums, independently recomputed
the training class differences and Cliff's deltas, and checked all PNGs for
readability and 300-DPI metadata. The saved tables contain 3,900 image-statistic
rows, 3,072 histogram-bin rows, and 12 training-only class comparisons.

The source inventory fingerprint matches Stage 1. Before/after SHA-256 checks
also confirmed that both manifests, CV assignments, and all other Stage 1 table
and review artifacts remained unchanged (19 protected files including directory
placeholders). All seven figures were visually inspected for readable English
labels and correct native-resolution presentation.

Measured locally on the existing Windows/Python 3.14.7 CPU environment:

| Execution | Measured elapsed time |
| --- | ---: |
| Native EDA, including imports, image reads, preservation checks, statistics and all output exports | 58.95 s |
| Existing five-module test suite, pytest-reported duration | 30.36 s |
| Test wrapper including imports and pytest startup | 31.54 s |

These are single-run wall-clock measurements using a monotonic timer, not
benchmark averages or GPU comparisons. Process-launch/approval time and manual
figure inspection are excluded. **115 tests passed**, with no failures or
warnings reported. No feature extraction or classifier experiment was run.

### Generated Stage 2 outputs

All seven figures are **300-DPI PNGs** with English labels:

| Figure in `results/figures/` | Content |
| --- | --- |
| `eda_class_distribution.png` | Approximately 60/40 defective/OK balance is preserved across the existing partitions. |
| `eda_training_samples.png` | Three seed-42 examples per class illustrate appearance variation; selection does not establish physical-part identity. |
| `eda_training_intensity_histograms.png` | Original and processing-input curves coincide; class intensity distributions remain different after normalization. |
| `eda_training_image_statistics.png` | Defective images tend toward lower mean intensity and within-image SD, with overlap and outliers. |
| `eda_preprocessing_comparison.png` | The same examples retain 512 x 512 dimensions across uint8 decoding, float32 conversion and normalization. |
| `eda_preprocessing_histograms.png` | Per-image intensity remapping changes the scale to [0, 1] while the native processing-input histogram matches the original. |
| `eda_training_class_comparisons.png` | Negative mean/SD Cliff's deltas describe lower values in defective training images; no p-values or model claims. |

Computed numerical outputs in `results/tables/`:

- `eda_class_distribution.csv`: counts and percentages.
- `eda_image_statistics.csv`: per-image, per-stage statistics and analysis scope.
- `eda_class_statistics.csv`: class summaries, with train/test rows separated.
- `eda_class_comparisons.csv`: training-only image-level differences/effects.
- `eda_intensity_histograms.csv`: bin edges, counts, and probabilities.
- `eda_selected_samples.csv`: reproducible sample identities and seed.
- `eda_preprocessing_validation.csv`: dimensions, dtype, finiteness, bounds,
  extrema, and constant-image checks for every processed image.
- `eda_run_summary.json`: configuration, versions, preservation, and scope.
- `eda_report.md`: computed findings, methodology, and limitations.

The automated tests cover normalization arithmetic, extreme finite values,
constant images, non-square and mixed-direction resizing, invalid inputs,
histogram endpoints, class effects with ties, deterministic samples, input
preservation, native 512 x 512 defaults without resampling, dimension-preserving
normalization, fixed display ranges, publication DPI, and isolation of optional
test descriptions from training comparisons. Temporary fixtures are used.
No HOG, GLCM, SVM, or Random Forest experiment is performed by this workflow.

The smaller 32/64-pixel buffers inside the Stage 1 similarity audit are internal
fingerprint/review calculations, not project preprocessing defaults. They do
not change raw images or processed image dimensions. Histogram bin counts of
256 and grayscale values in [0, 255] are intensity conventions, not resolutions.

## Stage 3: Native-resolution HOG and GLCM

All feature inputs are the original **512 x 512** grayscale images after the
existing stateless per-image Min-Max transform. No spatial resizing, learned
scaler, PCA, feature selection, augmentation, or classifier fitting is applied.
Both HOG configurations are retained for Stage 4 comparison using the existing
training-only CV assignments. Test descriptors do not select a configuration.

### Descriptor definitions and ordering

HOG uses **9 unsigned orientation bins, 2 x 2 cells per block, L2-Hys block
normalization, one-cell block stride**, and either **8 x 8** or **16 x 16**
pixels per cell. `transform_sqrt=False`, `channel_axis=None`, and
`feature_vector=True` are explicit. At 512 x 512 there are no partial cells:

- 8 x 8: 64 x 64 cells, 63 x 63 blocks, **142,884 features**.
- 16 x 16: 32 x 32 cells, 31 x 31 blocks, **34,596 features**.

The length is `(height/cell_height - 1) * (width/cell_width - 1) * 2 * 2 * 9`.
Flattening follows scikit-image's block-row, block-column, cell-row,
cell-column, orientation order. Descriptors and cached matrices are float32.
HOG renderings visualize gradient structure; they are not the flattened,
block-normalized descriptors and are not classifier explanations.

GLCM quantizes normalized pixels with `q = min(floor(32 * z), 31)`, including
the upper endpoint safely. It uses **32 levels**, distances **[1, 2]**, and
angles **[0, pi/4, pi/2, 3*pi/4]** (0, 45, 90, 135 degrees), with
`symmetric=True` and `normed=True`. Every distance/angle matrix is a symmetric
probability distribution. Pixel offsets follow scikit-image's integer-grid
distance/angle convention.

For matrix P, Contrast is `sum((i-j)^2 P)`, Dissimilarity is
`sum(abs(i-j) P)`, Homogeneity is `sum(P/(1+(i-j)^2))`, ASM is `sum(P^2)`,
Energy is `sqrt(ASM)`, and Correlation is the standardized joint covariance.
For degenerate constant inputs, scikit-image reports correlation 1 by convention;
this behavior is tested. **48 features** are retained without averaging over
distances or angles: six properties x two distances x four angles. Ordering is
Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM; within each
property, distance then angle. Normalized matrices use float64 internally;
the flattened cached descriptor uses float32.

The implementations follow the official
[scikit-image HOG](https://scikit-image.org/docs/0.26.x/api/skimage.feature.html#skimage.feature.hog),
[GLCM](https://scikit-image.org/docs/0.26.x/api/skimage.feature.html#skimage.feature.graycomatrix),
and [property](https://scikit-image.org/docs/0.26.x/api/skimage.feature.html#skimage.feature.graycoprops)
definitions. Synthetic tests check block geometry, constant gradients,
quantization boundaries, and hand-computed alternating-stripe GLCMs.

### Measured resource comparison

The same **50 training images (30 defective, 20 OK)** are selected with seed
42 and proportional class allocation, independently of test pixels. Each method
runs sequentially in a fresh CPU process with one untimed warm-up. OpenCV and
common numerical thread pools are constrained to one thread. Descriptor time
excludes JPEG decoding and normalization; end-to-end time includes them and
per-vector validation. OS peak RSS includes native allocations and imports,
not just Python objects; the baseline is recorded after warm-up.

| Method | Dimensions | Descriptor time, 50 images | End-to-end time | Worker peak RSS | All-image float32 payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG 8 x 8 | 142,884 | 6.3749 s | 6.6706 s | 54.21 MiB | 708.5770 MiB |
| HOG 16 x 16 | 34,596 | 3.1113 s | 3.3786 s | 53.25 MiB | 171.5652 MiB |
| GLCM | 48 | 0.4298 s | 0.7492 s | 70.81 MiB | 0.2380 MiB |

These are actual single-run measurements on the current Windows CPU environment,
not speed guarantees or classification results. The raw payload for all three
methods is **923,145,600 bytes (880.3802 MiB)**; .npy headers and labels are extra.
HOG 16 x 16 has about 4.13 times fewer features. That resource advantage does not
establish better classification performance. Separate full-extraction timings
and file sizes are recorded in `feature_cache_inventory.csv` and
`feature_run_summary.json`.

The completed full-dataset extraction, file writes and cache validation took
**261.73 seconds**. The extraction invocation reusing the saved benchmark took
**274.64 seconds through its preservation checks and figure exports**, excluding
the earlier benchmark invocation and final metadata/report serialization.
Its OS peak working set was **679.95 MiB**, including memory-mapped validation
pages and plotting allocations; it finished at 215.13 MiB. Streaming avoids
whole-matrix write allocations, but memory-mapped pages can still contribute
to the observed working set during validation. All eight cache files, including
labels and headers, occupy **923,147,924 bytes (880.3825 MiB)**.

Feasibility uses training benchmark costs only: each method must project to at
most 600 seconds, available RAM must exceed both 512 MiB and twice measured
worker peak RSS, and selected caches must leave 10% disk headroom. Both HOG
configurations and GLCM passed. This is a resource gate, not model selection.

### Reproduction and cache use

```powershell
# Training-only measurements and benchmark figure, without full extraction.
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -B -m src.feature_pipeline --benchmark-only

# Reuse matching benchmark metadata, then cache feasible configurations.
.\.venv\Scripts\python.exe -B -m src.feature_pipeline --reuse-benchmark

# Feature and cache correctness tests.
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_feature_extraction.py tests/test_feature_pipeline.py
```

Without flags, the workflow benchmarks and then extracts feasible configurations.
Existing cache files are never silently overwritten. A deliberate fresh run can
use `--cache-dir results/cache/features_new_run`; source, configuration,
dependency or input changes require a fresh benchmark. Such a run refreshes the
current Stage 3 reports/indexes. `--max-projected-seconds` changes the documented
resource time budget; it does not affect descriptor values.

A plotting-only quality correction clarified pixels-per-cell labels and panel
spacing after extraction. Metadata retains the plotting hash used during the
measurements, the final plotting hash, and the **7.04-second** rendering refresh.
Descriptor/pipeline sources, measured benchmark values, sample GLCM values and
all cache checksums were verified unchanged; features were not re-extracted.

Features are streamed one image and one descriptor at a time into uncompressed
.npy files, using bounded write buffers. Complete matrices and a combined HOG+GLCM
matrix are never assembled in RAM or duplicated on disk. After extraction,
checks validate dimensions, dtype, finite values in small row chunks, SHA-256
checksums, labels and exact manifest row ordering. Raw-image fingerprints must
match Stage 1; manifest and CV fingerprints are checked before and after the run.

Large files remain **local and ignored** in `results/cache/features/`:

- `train_hog8.npy`, `test_hog8.npy`: 1,040/260 rows x 142,884 columns.
- `train_hog16.npy`, `test_hog16.npy`: 1,040/260 rows x 34,596 columns.
- `train_glcm.npy`, `test_glcm.npy`: 1,040/260 rows x 48 columns.
- `train_labels.npy`, `test_labels.npy`: manifest-aligned int8 labels.

Read without allocating a whole matrix:

```python
from src.data_loader import PROJECT_ROOT
from src.feature_pipeline import load_cached_features

x_train, y_train, row_index = load_cached_features(PROJECT_ROOT, 'hog8', 'train')
# x_train is a read-only np.memmap. Existing validation_fold values are in row_index.
```

The loader rejects changed input/CV fingerprints, configurations, implementation
sources, dependency versions, cache bytes, index files, labels, or row mappings.
Checksum verification is enabled by default. Cache generation validates raw
content; the loader does not rehash every source image on each load. Future
Stage 4 fitting must use the saved folds and fit any learned scaler only on
each fold's training rows. Avoid materializing both large HOG matrices or all
CV copies simultaneously.

### Stage 3 outputs and interpretation

Trackable figures in `results/figures/` are 300-DPI English-labeled PNGs:

- `feature_benchmark.png`: measured training costs and projected payload sizes.
- `feature_hog_comparison.png`: normalized inputs and HOG renderings at both
  cell scales for the first reviewed Stage 2 training sample of each class.
  Renderings share a per-row display range with a 99th-percentile upper bound.
- `feature_glcm_matrices.png`: distance-1 probabilities for both sample images
  at all four angles, with a shared color range; distance 2 is also extracted.
- `feature_glcm_properties.png`: all six properties at both distances and all
  angles for those same two images. These illustrative curves do not prove
  population class separation.

Technical tables and metadata in `results/tables/`:

- `feature_benchmark.csv`, `feature_benchmark_samples.csv`, and
  `feature_benchmark_summary.json`: actual timing/memory, the exact 50 training
  identities, parameters, versions and source/input fingerprints.
- `feature_train_index.csv`, `feature_test_index.csv`: cache row, image identity,
  path, label and split; training rows retain saved validation-fold membership.
- `feature_cache_inventory.csv`: cache shapes, sizes, hashes and measured times.
- `feature_visualization_samples.csv`, `feature_sample_glcm.csv`: two reviewed
  training identities and their 96 property/distance/angle values.
- `feature_run_summary.json`, `feature_report.md`: measured extraction outcomes,
  feasibility decisions, preservation checks, methodology and limitations.

Final validation checked all **3,900 cached descriptor rows** for finiteness
and valid descriptor/property bounds. Eighteen recomputed vectors (first,
middle and last image in each split, for all three methods) matched cached
values exactly. Manifest ordering, labels, all saved CV fold assignments and
300-DPI PNG metadata passed checks. All four figures were visually reviewed.
The final Stage 1-3 suite passed **151 tests in 30.26 seconds**, including a
regression check for nonoverlapping HOG panel titles and explicit cell-size
units. No model was fitted and no best feature configuration was selected.

## Stage 4A: Fixed SVM and Random Forest baselines

The experiment reads the Stage 3 training feature caches at native 512 × 512
preprocessing resolution. It compares HOG 8 × 8, HOG 16 × 16, GLCM,
HOG 8 × 8 + GLCM, and HOG 16 × 16 + GLCM with both classifiers. Combined designs append
GLCM columns after HOG columns, using an explicit join on saved image IDs.
Cache integrity checks verify provenance, labels and manifest row order.

The accepted 1,040 training images consist of 625 defective and 415 OK images.
The saved five-fold assignments give 832 fit and 208 validation images per
fold. Each training image is validated once for every complete combination.
The held-out 260 images are reserved for final evaluation. The Stage 3 loader
checks the test manifest fingerprint along with the other saved inputs.

| Classifier | Fixed baseline settings |
| --- | --- |
| RBF SVM | `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=None, random_state=42)` with a `StandardScaler` inside the pipeline. |
| Random Forest | `RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight=None, random_state=42, n_jobs=1)` on the unscaled descriptors. |

These settings are specified before evaluation. To bound dense temporary
allocations, the SVM scaler accumulates StandardScaler statistics in batches
of 16 rows from the current fit fold. It then standardizes the fit and
validation designs with those statistics. A fresh pipeline is fitted per fold.
The experiment builds only the current phase's dense matrix from read-only
memory maps. SVM uses float64, which is the format used by the dense libsvm
backend. Random Forest uses float32. Fold workers run one at a time, with
numerical thread pools limited to one.

Before each combination, a resource smoke test fits 96 images and predicts 32
images sampled from the respective fit and validation portions of saved fold 1
with seed 42. Full folds have a preflight memory estimate and a 384 MiB physical
memory reserve. The runner stops an affected configuration if the estimate
cannot fit or available RAM falls below the reserve on two consecutive samples.
Peak RSS is the OS working-set high-water mark of the worker and includes
mapped pages and imports. CPU and available memory can vary across runs.
The initial Random Forest estimate counted the dense fit matrix and 256 MiB of
overhead. Its observed peak on HOG 8 × 8 exceeded that estimate. The reproduction
command now budgets for resident source cache pages alongside the fit matrix.
The CSV preserves the estimates used for the measured run.

Defective is label 1. The experiment reports accuracy, balanced accuracy,
defective precision, recall, F1 and ROC-AUC. AUC uses SVM decision scores or
Random Forest defective probabilities. False negatives count defective images
predicted as OK. Reports show unweighted mean and sample standard deviation
across the five folds. The total false negatives sum out-of-fold counts over
the 625 defective images. Fit time includes SVM scaler fitting and fit-data
standardization. SVM prediction time includes fold-local validation
transformation and batched evaluation of the fitted RBF decision function,
which yields both label predictions and AUC scores. The calculation uses the
fitted support vectors, dual coefficients, resolved gamma and intercept.
Near-zero margins use scikit-learn's prediction directly. The first, middle
and last validation row of every fold are checked against SVC's decision scores
and predicted labels. Random Forest uses `predict` for labels and times
`predict_proba` separately. Data loading and SVC parity checks have separate
timing columns.

Reproduce from the repository root with the Stage 3 environment and local
training caches:

```powershell
.\.venv\Scripts\python.exe -B -m src.baseline_experiment
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

The runner writes smoke timings, individual folds, and aggregate results to
`baseline_cv_smoke.csv`, `baseline_cv_folds.csv`, and
`baseline_cv_summary.csv` in `results/tables/`. The accompanying
`baseline_cv_run.json` records the fixed settings, versions, and input hashes.
Five `baseline_cv_*.png` figures in `results/figures/` show F1, recall,
accuracy and balanced accuracy, fit time, and the HOG cell-size comparison.
Their error bars show variability between saved folds, not confidence intervals.

### Measured baseline results

All ten combinations completed their five saved folds. The table reports the
mean ± sample SD across folds. FN totals count missed defective images out of
625 out-of-fold defective examples. Full precision is available in the
[summary CSV](results/tables/baseline_cv_summary.csv).

| Features | Classifier | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | FN | Fit (s) | Predict (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG 8×8 | SVM | 0.749 ± 0.037 | 0.726 ± 0.040 | 0.767 ± 0.032 | 0.838 ± 0.043 | 0.800 ± 0.030 | 0.868 ± 0.024 | 101 | 37.763 ± 3.621 | 2.805 ± 0.174 |
| HOG 8×8 | Random Forest | 0.716 ± 0.040 | 0.687 ± 0.042 | 0.733 ± 0.031 | 0.830 ± 0.039 | 0.779 ± 0.032 | 0.789 ± 0.031 | 106 | 21.990 ± 0.224 | 0.037 ± 0.003 |
| HOG 16×16 | SVM | 0.857 ± 0.040 | 0.846 ± 0.045 | 0.870 ± 0.043 | 0.898 ± 0.036 | 0.883 ± 0.032 | 0.943 ± 0.018 | 64 | 7.987 ± 0.262 | 0.610 ± 0.023 |
| HOG 16×16 | Random Forest | 0.771 ± 0.040 | 0.750 ± 0.042 | 0.784 ± 0.033 | 0.856 ± 0.041 | 0.818 ± 0.032 | 0.880 ± 0.020 | 90 | 10.242 ± 0.162 | 0.020 ± 0.003 |
| GLCM | SVM | 0.785 ± 0.022 | 0.775 ± 0.028 | 0.821 ± 0.036 | 0.822 ± 0.031 | 0.821 ± 0.017 | 0.871 ± 0.022 | 111 | 0.049 ± 0.004 | 0.003 ± 0.000 |
| GLCM | Random Forest | 0.898 ± 0.015 | 0.895 ± 0.018 | 0.920 ± 0.023 | 0.910 ± 0.021 | 0.915 ± 0.012 | 0.964 ± 0.010 | 56 | 0.427 ± 0.029 | 0.011 ± 0.001 |
| HOG 8×8 + GLCM | SVM | 0.752 ± 0.039 | 0.729 ± 0.042 | 0.769 ± 0.034 | 0.842 ± 0.041 | 0.803 ± 0.031 | 0.869 ± 0.024 | 99 | 34.320 ± 0.291 | 2.535 ± 0.037 |
| HOG 8×8 + GLCM | Random Forest | 0.735 ± 0.038 | 0.705 ± 0.040 | 0.744 ± 0.030 | 0.851 ± 0.039 | 0.794 ± 0.031 | 0.811 ± 0.027 | 93 | 21.937 ± 1.568 | 0.038 ± 0.006 |
| HOG 16×16 + GLCM | SVM | 0.861 ± 0.038 | 0.852 ± 0.042 | 0.876 ± 0.040 | 0.896 ± 0.035 | 0.885 ± 0.031 | 0.944 ± 0.018 | 65 | 7.603 ± 0.071 | 0.597 ± 0.016 |
| HOG 16×16 + GLCM | Random Forest | 0.801 ± 0.037 | 0.780 ± 0.042 | 0.806 ± 0.038 | 0.883 ± 0.034 | 0.842 ± 0.029 | 0.899 ± 0.017 | 73 | 11.002 ± 0.410 | 0.021 ± 0.002 |

**GLCM + Random Forest** had the highest mean F1 (0.9148), defective recall
(0.9104), balanced accuracy (0.8950), and ROC-AUC (0.9645). It missed 56 of the
625 defective images and averaged 0.427 seconds to fit. The RBF SVM had a
higher mean F1 than Random Forest on all four feature sets containing HOG.
Moving from 8×8 to 16×16 cells improved mean F1 and recall for both classifiers,
with or without GLCM. For HOG alone, the change reduced mean fit time from
37.76 to 7.99 seconds for SVM and from 21.99 to 10.24 seconds for Random Forest.

For Stage 4B, prioritize GLCM + Random Forest and retain the two HOG 16×16 SVM
variants. Adding GLCM to HOG 16×16 increased SVM mean F1 from 0.8829 to 0.8855
while increasing false negatives from 64 to 65. That small F1 gain should be
weighed alongside the fold variability and recall. Physical-part independence
remains unverified, so these training-CV comparisons cannot establish
performance on independently acquired parts. Hyperparameter search and final
holdout evaluation remain separate stages.

The final CPU run took **1,076.36 seconds (17 min 56 s)** including smoke
checks, input verification, worker startup and figure export. The sum of the
50 measured fit times was **766.60 seconds**. This run follows a separate
pilot, which timed standard SVC label and score calls individually. The final
run uses the batched decision calculation described above. Peak worker RSS was
**1,945.29 MiB**, and sampled available RAM stayed at or above **1,912.02 MiB**
across full folds. The smoke workers peaked between **127.32 and 339.90 MiB**.
No configuration triggered a resource stop.

The configured Stage 1–4A test suite passed **160 tests in 36.45 seconds**.
The added tests cover ID-based alignment, exclusion of holdout cache paths,
exact saved-fold membership, fold-local scaling, SVC score and label parity,
fixed classifier parameters, metric arithmetic, and CSV/JSON serialization.
All 1,300 raw-image hashes, both manifest hashes, and the saved CV checksum
matched the recorded Stage 1 and Stage 3 fingerprints.

## Git and dataset management

The existing GitHub repository is
[haktanc/MTH416](https://github.com/haktanc/MTH416).
The local `origin` remote uses `https://github.com/haktanc/MTH416.git` for
fetching and pushing. Raw datasets and the local Python environment are not
included in version control.

`/data/raw/` is ignored; `/data/` itself is not. Future `data/splits/*.csv`, Python
source, documentation, figures, and result tables remain eligible for tracking.
The `.gitkeep` in `results/models/` preserves the directory while generated model
binaries are ignored. Virtual environments, caches, secrets, and build artifacts
are also ignored. `results/cache/` will be ignored if created later.

To inspect the configuration without changing the Git index:

```powershell
git status --short
git remote -v
git check-ignore -v data/raw/casting_512x512/def_front/cast_def_0_0.jpeg
git ls-files -- data/raw/
```

Ignore rules do not remove already tracked files. No raw images were tracked at
the start of initialization, so no dataset index cleanup was needed. Dataset
audit and preparation scripts do not commit, push, rewrite Git history, or
change remote configuration. Review staged changes before making a commit.

## Planned development stages

Stages 2 and 3 are complete. Unconfirmed perceptual-similarity candidates are advisory
and do not block development. Keep the accepted manifests and saved CV
assignments unchanged. Unknown physical-part relationships remain a limitation
to disclose; only substantiated new evidence would justify a separately
reviewed protocol change, without consulting model/test performance.

1. Shortlist configurations using Stage 4A baseline cross-validation results.
2. Tune CPU-based SVM and Random Forest models in Stage 4B using training data only
   and the existing saved CV assignments.
3. Evaluate on held-out data and save actual metrics and comparison tables.
4. Extend the Stage 2 scientific figures with genuine model results and conclusions.

Use English throughout file names, identifiers, comments, docstrings, logging,
charts, and documentation. Follow PEP 8 for Python development.

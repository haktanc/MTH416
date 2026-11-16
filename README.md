# MAT 416 - Visual Data Analysis

## Surface Defect Detection in Casting Pump Impeller Images Using Texture Analysis and Machine Learning

This academic project investigates surface defect detection using classical
image processing, texture descriptors, and machine learning. Planned methods
include HOG, GLCM, Support Vector Machines, and Random Forests. Deep learning
and GPU acceleration are not required. Execution is local and CPU based.

Stages 1 and 2 provide dataset preparation, exploratory data analysis (EDA),
and stateless grayscale preprocessing. The saved manifests remain
**provisional**, because source and physical-part independence cannot be
established from the available data. No HOG/GLCM experiments, classifier
training, or model selection have run.

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
|   |-- model_training.py
|   |-- evaluation.py
|   `-- visualization.py
|-- results/
|   |-- figures/.gitkeep             # PNG figures; trackable
|   |-- tables/                     # Audit, CV, and validation reports
|   `-- models/.gitkeep              # Generated binaries ignored
|-- tests/
|   |-- __init__.py
|   |-- test_initialization.py
|   |-- test_dataset_audit.py
|   |-- test_data_splitting.py
|   |-- test_preprocessing.py
|   |-- test_eda.py
|   `-- test_pipeline.py             # Existing analysis tests; deferred
|-- notebooks/                      # Existing empty directory; unused
|-- main.py
|-- pytest.ini
|-- requirements-init.txt
|-- requirements-stage1.txt
|-- requirements-stage2.txt
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
| `model_training.py` | CPU classifiers, training-only cross-validation, hyperparameter search, and serialization. |
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

The initialization, Stage 1, and Stage 2 environment uses Python 3.14.7 on Windows.
Use the existing `.venv/`. Only if it is absent, create it from an installed
Python interpreter:

```powershell
python -m venv .venv
```

Install the pinned Stage 2 dependencies and run the project from the
repository root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage2.txt
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
`requirements.txt` also lists the planned pandas, seaborn, and scikit-image
dependencies. These additional libraries remain deferred; Stage 2 does not
require them.
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
.\.venv\Scripts\python.exe -m pytest -q
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

`pytest.ini` selects initialization, audit, split, preprocessing, and EDA tests.
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
unknown physical parts. Scientific validation of HOG/GLCM helpers and all model
experiments remain future work.

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

# All relevant tests, without HOG/GLCM extraction or model training.
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
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

Stage 2 is complete. Unconfirmed perceptual-similarity candidates are advisory
and do not block development. Keep the accepted manifests and saved CV
assignments unchanged. Unknown physical-part relationships remain a limitation
to disclose; only substantiated new evidence would justify a separately
reviewed protocol change, without consulting model/test performance.

1. Validate HOG and GLCM configurations in the later feature-extraction stage.
2. Fit and tune CPU-based SVM and Random Forest models using training data only
   and the existing saved CV assignments.
3. Evaluate on held-out data and save actual metrics and comparison tables.
4. Extend the Stage 2 scientific figures with genuine model results and conclusions.

Use English throughout file names, identifiers, comments, docstrings, logging,
charts, and documentation. Follow PEP 8 for Python development.

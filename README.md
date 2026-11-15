# MAT 416 - Visual Data Analysis

## Surface Defect Detection in Casting Pump Impeller Images Using Texture Analysis and Machine Learning

This academic project investigates surface defect detection using classical
image processing, texture descriptors, and machine learning. Planned methods
include HOG, GLCM, Support Vector Machines, and Random Forests. Deep learning
and GPU acceleration are not required. Execution is local and CPU based.

Stage 1 audits the primary collection and prepares reproducible holdout and
cross-validation assignments. The manifests are **provisional**, because source
and physical-part independence cannot be established from the available data.
No HOG/GLCM experiments, classifier training, or model selection have run.

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
|   `-- test_pipeline.py             # Existing analysis tests; deferred
|-- notebooks/                      # Existing empty directory; unused
|-- main.py
|-- pytest.ini
|-- requirements-init.txt
|-- requirements-stage1.txt
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
| `reporting.py` | Deterministic CSV/JSON serialization and raw-directory output protection. |
| `preprocessing.py` | Resizing and Min-Max normalization. |
| `feature_extraction.py` | Configurable HOG, GLCM, and combined feature vectors. |
| `model_training.py` | CPU classifiers, training-only cross-validation, hyperparameter search, and serialization. |
| `evaluation.py` | Accuracy, precision, recall, F1, confusion matrices, ROC-AUC, and comparison tables. |
| `visualization.py` | Sample, histogram, texture, confusion matrix, and comparison plots saved as PNG. |
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

The initialization and Stage 1 environment is verified with Python 3.14.7 on Windows.
Use the existing `.venv/`. Only if it is absent, create it from an installed
Python interpreter:

```powershell
python -m venv .venv
```

Install the pinned Stage 1 dependencies and run the project from the
repository root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage1.txt
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
`requirements.txt` also lists the planned pandas, matplotlib, seaborn, and
scikit-image dependencies. These optional analysis libraries remain deferred.
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

`pytest.ini` selects initialization, audit, and split tests. They use temporary
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

1. Review Stage 1 similarity candidates and source/physical-part provenance.
2. Record any verified groups and regenerate holdout/CV assignments; finalize
   the dataset protocol without consulting model or test performance.
3. Validate resizing, normalization, HOG, and GLCM configurations.
4. Fit and tune CPU-based SVM and Random Forest models using training data only.
5. Evaluate on held-out data and save actual metrics and comparison tables.
6. Produce English-labeled scientific PNG figures and document conclusions.

Use English throughout file names, identifiers, comments, docstrings, logging,
charts, and documentation. Follow PEP 8 for Python development.

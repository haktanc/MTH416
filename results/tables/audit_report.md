# Primary Dataset Audit

Dataset: `data/raw/casting_512x512`.
Images: 1300; readable: 1300; eligible: 1300.
Class counts: {'def_front': 781, 'ok_front': 519}.
Dimensions: {'512x512': 1300}; formats: {'JPEG': 1300}; modes: {'RGB': 1300}.
Exact file pairs: 0; additional exact decoded-pixel pairs: 0.
Near-duplicate candidates: 3485 pairs involving 1107 images.

## Method and interpretation

Every image is verified and fully decoded with Pillow and decoded as grayscale with the existing OpenCV loader. File and canonical RGB pixel contents are hashed with SHA-256. Canonical RGB hashing includes dimensions and dtype. Eligibility requires a readable grayscale JPEG with 512 x 512 dimensions. No invalid images are silently excluded.

Every unordered image pair is screened with a 63-bit DCT hash (32 x 32 resize; 8 x 8 low-frequency block; DC excluded; median threshold). A minimum Hamming distance <= 4 across eight rotations/reflections flags a candidate. This deliberately conservative, predeclared cutoff is not a validated identity classifier. RMSE and correlation on aligned 64 x 64 previews are descriptive review aids, not acceptance criteria. False positives and negatives remain possible, particularly for similar industrial components.

## Filename and metadata investigation

Observed filename prefixes: {'cast_def_0': 781, 'cast_ok_0': 519}.
Images with nonempty EXIF: 0.
Repeated numeric filename suffixes: {'1091': 2, '1155': 2, '1187': 2, '1192': 2, '1589': 2, '1715': 2, '1817': 2, '2060': 2, '2437': 2, '2614': 2, '3136': 2, '3199': 2, '3246': 2, '3812': 2, '3869': 2, '3916': 2, '4697': 2, '5339': 2, '5524': 2, '5573': 2, '583': 2, '6236': 2, '6239': 2, '6738': 2, '6869': 2, '688': 2, '7076': 2, '7117': 2, '7338': 2, '7664': 2, '773': 2, '8067': 2, '8208': 2, '8278': 2, '833': 2, '8646': 2, '93': 2, '9591': 2}.
The filename prefix/counter and suffix have no supplied mapping to physical parts or original captures. EXIF values are recorded in the inventory; none are promoted to verified group identifiers. No reliable physical-part grouping is established by this automated audit.

## Limitations and review

- Perceptual similarity is not proof of a shared source or physical part.
- The common filename counter and numeric suffixes are not verified part IDs.
- EXIF or filename coincidences are recorded, not automatically trusted as groups.
- No near-duplicate screen can exclude every transformation or shared part.
- Class names are structural labels; visual label correctness is not verified.

Review `near_duplicate_candidates.csv` and open `near_duplicate_review.html` locally. The HTML references raw images and intentionally contains no copied image files. Obtain acquisition records or supplier confirmation to establish source groups. A split with unresolved relationships must remain provisional.

Blocking issues: [].
Inventory SHA-256: `d3ecba68cd0c1f166268d17d3558b680e8cde6b6e69aaa9d7508d8a2970e6fdd`.

References: [Pillow image verification](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify).

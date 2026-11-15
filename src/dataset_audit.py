"""Audit primary images and screen similarity without changing raw files."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import ExifTags, Image

from src.data_loader import (
    EXPECTED_PRIMARY_COUNTS,
    PRIMARY_DATASET,
    PROJECT_ROOT,
    collect_primary_records,
    image_identifier,
    load_grayscale,
)
from src.reporting import (
    inventory_digest,
    output_directory,
    write_csv,
    write_json,
)


TRANSFORMS = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "mirror", "mirror_rotate_90", "mirror_rotate_180", "mirror_rotate_270",
)
AUDIT_FIELDS = [
    "image_path", "image_id", "class_name", "label", "file_bytes", "sha256",
    "pixel_sha256", "readable", "eligible", "width", "height", "format",
    "mode", "channels", "grayscale", "exif_json", "info_keys", "phash",
    "filename_prefix", "filename_index", "error",
]
PAIR_FIELDS = [
    "image_path_a", "image_path_b", "class_a", "class_b", "kind",
    "phash_distance", "transform_b", "rmse_64", "correlation_64",
    "review_status",
]


@dataclass
class AuditResult:
    """Contain actual observations, exact pairs, and unresolved candidates."""

    rows: list[dict]
    exact_pairs: list[dict]
    near_pairs: list[dict]
    summary: dict


def transform_image(image: np.ndarray, index: int) -> np.ndarray:
    """Apply one of eight square-image rotations/reflections for screening."""
    base = np.fliplr(image) if index >= 4 else image
    return np.rot90(base, index % 4)


def perceptual_hash(image: np.ndarray) -> int:
    """Threshold 63 low-frequency DCT coefficients, excluding the DC term.

    This is a similarity screen, not a source or physical-part identifier.
    """
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    values = cv2.dct(small.astype(np.float32))[:8, :8].ravel()[1:]
    bits = values > np.median(values)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _pixel_digest(image: np.ndarray) -> str:
    header = f"{image.shape}:{image.dtype}:".encode("ascii")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def _pair(first: dict, second: dict, kind: str) -> dict:
    return {
        "image_path_a": first["image_path"],
        "image_path_b": second["image_path"],
        "class_a": first["class_name"],
        "class_b": second["class_name"],
        "kind": kind,
    }


def audit_dataset(
    dataset_directory: Path = PRIMARY_DATASET,
    project_root: Path = PROJECT_ROOT,
    near_threshold: int = 4,
    expected_counts: dict[str, int] | None = None,
) -> AuditResult:
    """Decode all images and compare every pair at a declared hash threshold.

    Invalid directory structure is fatal. Unreadable or unexpected images are
    retained in the report and block preparation; none are silently discarded.
    """
    if not 0 <= near_threshold <= 63:
        raise ValueError("Perceptual hash threshold must be between 0 and 63.")
    root = Path(project_root).resolve()
    primary = (root / "data" / "raw" / "casting_512x512").resolve()
    if Path(dataset_directory).resolve() != primary:
        raise ValueError("Stage 1 audits only data/raw/casting_512x512.")
    records = sorted(
        collect_primary_records(primary), key=lambda record: record.path
    )
    rows, hashes, previews = [], {}, {}
    for record in records:
        relative = record.path.relative_to(root).as_posix()
        row = dict.fromkeys(AUDIT_FIELDS, "")
        row.update(
            image_path=relative, image_id=image_identifier(relative),
            class_name=record.class_name, label=record.label,
            readable=False, eligible=False,
        )
        match = re.fullmatch(r"(cast_(?:def|ok)_\d+)_(\d+)", record.path.stem)
        if match:
            row["filename_prefix"], row["filename_index"] = match.groups()
        else:
            row["filename_prefix"] = "unrecognized"
        try:
            data = record.path.read_bytes()
            row.update(file_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                exif = image.getexif()
                metadata = {
                    ExifTags.TAGS.get(tag, str(tag)): str(value)
                    for tag, value in exif.items()
                }
                # Nested EXIF can contain image/source identifiers.
                for tag in (34665, 34853):
                    if tag in exif:
                        for key, value in exif.get_ifd(tag).items():
                            name = ExifTags.TAGS.get(key, str(key))
                            metadata[f"IFD_{tag}.{name}"] = str(value)
                rgb = np.asarray(image.convert("RGB"))
                grayscale = bool(
                    np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                    and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
                )
                row.update(
                    width=image.width, height=image.height,
                    format=image.format, mode=image.mode,
                    channels=len(image.getbands()), grayscale=grayscale,
                    exif_json=json.dumps(metadata, sort_keys=True),
                    info_keys=",".join(sorted(image.info)),
                    pixel_sha256=_pixel_digest(rgb),
                )
            gray = load_grayscale(record.path)
            variants = tuple(
                perceptual_hash(transform_image(gray, index))
                for index in range(8)
            )
            hashes[relative] = variants
            previews[relative] = cv2.resize(gray, (64, 64))
            row.update(readable=True, phash=f"{variants[0]:016x}")
            problems = []
            if (row["width"], row["height"]) != (512, 512):
                problems.append("Unexpected dimensions; expected 512 x 512")
            if row["format"] != "JPEG" or record.path.suffix.lower() not in {
                ".jpg", ".jpeg",
            }:
                problems.append("Unexpected file format; expected JPEG")
            if not grayscale:
                problems.append("Non-grayscale pixel content")
            row["eligible"] = not problems
            row["error"] = "; ".join(problems)
        except (OSError, ValueError, cv2.error, SyntaxError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    exact, near = [], []
    for index, first in enumerate(rows):
        if not first["readable"]:
            continue
        for second in rows[index + 1:]:
            if not second["readable"]:
                continue
            if first["sha256"] == second["sha256"]:
                exact.append(_pair(first, second, "file_sha256"))
                continue
            if first["pixel_sha256"] == second["pixel_sha256"]:
                exact.append(_pair(first, second, "decoded_rgb_sha256"))
                continue
            first_hash = hashes[first["image_path"]][0]
            distances = [
                (first_hash ^ value).bit_count()
                for value in hashes[second["image_path"]]
            ]
            distance = min(distances)
            if distance > near_threshold:
                continue
            orientation = distances.index(distance)
            a = previews[first["image_path"]].astype(np.float64)
            b = transform_image(
                previews[second["image_path"]], orientation
            ).astype(np.float64)
            correlation = (
                float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
                if a.std() > 0 and b.std() > 0 else None
            )
            pair = _pair(first, second, "perceptual_candidate")
            pair.update(
                phash_distance=distance, transform_b=TRANSFORMS[orientation],
                rmse_64=round(float(np.sqrt(np.mean(((a - b) / 255) ** 2))), 6),
                correlation_64=(
                    round(correlation, 6) if correlation is not None else ""
                ),
                review_status="unresolved",
            )
            near.append(pair)
    near.sort(key=lambda pair: (
        pair["phash_distance"], pair["rmse_64"],
        pair["image_path_a"], pair["image_path_b"],
    ))
    counts = dict(sorted(Counter(row["class_name"] for row in rows).items()))
    expected = EXPECTED_PRIMARY_COUNTS if expected_counts is None else expected_counts
    issues = [row["image_path"] for row in rows if not row["eligible"]]
    if counts != expected:
        issues.append(f"Count mismatch: expected {expected}, observed {counts}")
    if any(pair["class_a"] != pair["class_b"] for pair in exact):
        issues.append("Exact duplicates have conflicting class labels")
    summary = {
        "dataset": primary.relative_to(root).as_posix(),
        "image_count": len(rows), "class_counts": counts,
        "expected_class_counts": expected,
        "readable_count": sum(row["readable"] for row in rows),
        "eligible_count": sum(row["eligible"] for row in rows),
        "dimensions": dict(Counter(
            f"{row['width']}x{row['height']}" for row in rows
        )),
        "formats": dict(Counter(str(row["format"]) for row in rows)),
        "modes": dict(Counter(str(row["mode"]) for row in rows)),
        "channel_counts": dict(Counter(str(row["channels"]) for row in rows)),
        "file_extensions": dict(Counter(
            Path(row["image_path"]).suffix for row in rows
        )),
        "metadata_info_keys": dict(Counter(row["info_keys"] for row in rows)),
        "grayscale_count": sum(row["grayscale"] is True for row in rows),
        "images_with_exif": sum(row["exif_json"] not in {"", "{}"} for row in rows),
        "filename_prefixes": dict(Counter(row["filename_prefix"] for row in rows)),
        "repeated_filename_indices": {
            value: count for value, count in Counter(
                row["filename_index"] for row in rows if row["filename_index"]
            ).items() if count > 1
        },
        "exact_file_pairs": sum(p["kind"] == "file_sha256" for p in exact),
        "exact_pixel_only_pairs": sum(
            p["kind"] == "decoded_rgb_sha256" for p in exact
        ),
        "near_candidate_pairs": len(near),
        "near_candidate_images": len({
            p[key] for p in near for key in ("image_path_a", "image_path_b")
        }),
        "near_cross_class_pairs": sum(p["class_a"] != p["class_b"] for p in near),
        "perceptual_hash_bits": 63, "near_threshold": near_threshold,
        "transforms_screened": list(TRANSFORMS),
        "inventory_sha256": inventory_digest(rows),
        "blocking_issues": issues,
        "physical_part_groups_verified": False,
        "warnings": [
            "Perceptual similarity is not proof of a shared source or physical part.",
            "The common filename counter and numeric suffixes are not verified part IDs.",
            "EXIF or filename coincidences are recorded, not automatically trusted as groups.",
            "No near-duplicate screen can exclude every transformation or shared part.",
            "Class names are structural labels; visual label correctness is not verified.",
        ],
    }
    return AuditResult(rows, exact, near, summary)


def assert_raw_unchanged(audit: AuditResult, project_root: Path = PROJECT_ROOT):
    """Reject changed contents, additions, and removals since the audit."""
    root = Path(project_root).resolve()
    current = collect_primary_records(root / "data/raw/casting_512x512")
    paths = {record.path.relative_to(root).as_posix() for record in current}
    if paths != {row["image_path"] for row in audit.rows}:
        raise ValueError("Raw dataset membership changed during preparation.")
    for row in audit.rows:
        digest = hashlib.sha256((root / row["image_path"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Raw image changed: {row['image_path']}")


def write_audit_reports(
    audit: AuditResult, directory: Path,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Save CSV observations, machine-readable findings, and pair review HTML."""
    directory = output_directory(directory, project_root)
    write_csv(directory / "dataset_audit.csv", audit.rows, AUDIT_FIELDS)
    write_csv(directory / "exact_duplicates.csv", audit.exact_pairs, PAIR_FIELDS)
    write_csv(directory / "near_duplicate_candidates.csv", audit.near_pairs, PAIR_FIELDS)
    write_json(directory / "audit_summary.json", audit.summary)
    write_pair_review(
        audit.near_pairs, directory / "near_duplicate_review.html", project_root
    )
    summary = audit.summary
    report = [
        "# Primary Dataset Audit", "",
        f"Dataset: `{summary['dataset']}`.",
        f"Images: {summary['image_count']}; readable: {summary['readable_count']}; "
        f"eligible: {summary['eligible_count']}.",
        f"Class counts: {summary['class_counts']}.",
        f"Dimensions: {summary['dimensions']}; formats: {summary['formats']}; "
        f"modes: {summary['modes']}.",
        f"Exact file pairs: {summary['exact_file_pairs']}; additional exact "
        f"decoded-pixel pairs: {summary['exact_pixel_only_pairs']}.",
        f"Near-duplicate candidates: {len(audit.near_pairs)} pairs involving "
        f"{summary['near_candidate_images']} images.", "",
        "## Method and interpretation", "",
        "Every image is verified and fully decoded with Pillow and decoded "
        "as grayscale with the existing OpenCV loader. File and canonical RGB "
        "pixel contents are hashed with SHA-256. Canonical RGB hashing includes "
        "dimensions and dtype. Eligibility requires a readable grayscale JPEG "
        "with 512 x 512 dimensions. No invalid images are silently excluded.", "",
        f"Every unordered image pair is screened with a {63}-bit DCT hash "
        f"(32 x 32 resize; 8 x 8 low-frequency block; DC excluded; median "
        f"threshold). A minimum Hamming distance <= {summary['near_threshold']} "
        "across eight rotations/reflections flags a candidate. This deliberately "
        "conservative, predeclared cutoff is not a validated identity classifier. "
        "RMSE and correlation on aligned 64 x 64 previews are descriptive "
        "review aids, not acceptance criteria. False positives and negatives "
        "remain possible, particularly for similar industrial components.", "",
        "## Filename and metadata investigation", "",
        f"Observed filename prefixes: {summary['filename_prefixes']}.",
        f"Images with nonempty EXIF: {summary['images_with_exif']}.",
        f"Repeated numeric filename suffixes: "
        f"{summary['repeated_filename_indices']}.",
        "The filename prefix/counter and suffix have no supplied mapping to "
        "physical parts or original captures. EXIF values are recorded in the "
        "inventory; none are promoted to verified group identifiers. No "
        "reliable physical-part grouping is established by this automated audit.",
        "", "## Limitations and review", "",
        *[f"- {warning}" for warning in summary["warnings"]], "",
        "Review `near_duplicate_candidates.csv` and open "
        "`near_duplicate_review.html` locally. The HTML references raw images "
        "and intentionally contains no copied image files. Obtain acquisition "
        "records or supplier confirmation to establish source groups. A split "
        "with unresolved relationships must remain provisional.", "",
        f"Blocking issues: {summary['blocking_issues']}.",
        f"Inventory SHA-256: `{summary['inventory_sha256']}`.", "",
        "References: [Pillow image verification](https://pillow.readthedocs.io/"
        "en/stable/reference/Image.html#PIL.Image.Image.verify).", "",
    ]
    (directory / "audit_report.md").write_text("\n".join(report), encoding="utf-8")


def write_pair_review(
    pairs: list[dict], path: Path, project_root: Path = PROJECT_ROOT,
) -> None:
    """Build an offline, paginated review with optional partition filters."""
    output_directory(path.parent, project_root)
    data = []
    for number, pair in enumerate(pairs, 1):
        orientation = TRANSFORMS.index(pair["transform_b"])
        display = f"rotate({-90 * (orientation % 4)}deg)"
        if orientation >= 4:
            display += " scaleX(-1)"
        entry = {**pair, "pair_number": number, "display_transform": display}
        for suffix in ("a", "b"):
            source = Path(project_root) / pair[f"image_path_{suffix}"]
            entry[f"source_{suffix}"] = Path(
                os.path.relpath(source, path.parent)
            ).as_posix()
        data.append(entry)
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    template = Path(__file__).with_name("templates") / "pair_review.html"
    document = template.read_text(encoding="utf-8").replace("__PAIR_DATA__", payload)
    path.write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/tables")
    args = parser.parse_args(argv)
    try:
        audit = audit_dataset(near_threshold=args.near_threshold)
        assert_raw_unchanged(audit)
        write_audit_reports(audit, args.output_dir)
        print(json.dumps(audit.summary, indent=2, sort_keys=True))
        return 1 if audit.summary["blocking_issues"] else 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"Audit failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

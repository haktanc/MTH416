"""Check project initialization and access to the unsplit primary dataset."""

import argparse
from pathlib import Path

from src.data_loader import (
    EXPECTED_PRIMARY_COUNTS,
    PRIMARY_DATASET,
    check_integrity,
    collect_primary_records,
    collect_records,
    find_dataset,
    image_counts,
    load_grayscale,
    primary_image_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def inspect_dataset(
    raw_directory: Path,
    results_directory: Path,
    verify_images: bool = False,
    preview: bool = False,
) -> None:
    """Preserve the legacy augmented-set inspector for explicit future use.

    This function is not called by the initialization entry point. Its optional
    previews require the planned analysis dependencies and extract features.
    """
    import pandas as pd

    from src.preprocessing import prepare_image

    dataset = find_dataset(raw_directory)
    records = collect_records(dataset)
    counts = pd.DataFrame(image_counts(records))
    tables = results_directory / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    counts.to_csv(tables / "dataset_counts.csv", index=False)
    print(f"Dataset: {dataset}")
    print(counts.to_string(index=False))
    print(f"Total train/test images: {len(records)}")
    if verify_images:
        failures = check_integrity(records)
        pd.DataFrame(failures, columns=["path", "error"]).to_csv(
            tables / "integrity_failures.csv", index=False,
        )
        if failures:
            raise ValueError(
                f"Image integrity check failed for {len(failures)} files."
            )
        print("Image integrity check passed for all train/test images.")
    if preview:
        from src.visualization import (
            plot_glcm,
            plot_histogram,
            plot_hog,
            plot_samples,
        )

        samples = [
            next(
                r for r in records if r.split == "train" and r.label == label
            )
            for label in (0, 1)
        ]
        images = [
            prepare_image(load_grayscale(record.path)) for record in samples
        ]
        figures = results_directory / "figures"
        plot_samples(
            images, ["Acceptable", "Defective"], figures / "samples.png"
        )
        plot_histogram(images[1], figures / "intensity_histogram.png")
        plot_hog(images[1], figures / "hog.png")
        plot_glcm(images[1], figures / "glcm.png")
        print(f"Preview figures saved to: {figures}")
    print("Inspection complete. No models were trained.")


def initialize_project(dataset_directory: Path = PRIMARY_DATASET) -> None:
    """Check counts and decode a sample per class without writing files."""
    records = collect_primary_records(dataset_directory)
    counts = primary_image_counts(records)
    for name, expected in EXPECTED_PRIMARY_COUNTS.items():
        if counts[name] != expected:
            raise ValueError(
                f"Unexpected image count for {name}: "
                f"expected {expected}, found {counts[name]}."
            )
    print(f"Primary dataset: {Path(dataset_directory).resolve()}")
    for name in EXPECTED_PRIMARY_COUNTS:
        sample = next(
            record for record in records if record.class_name == name
        )
        image = load_grayscale(sample.path)
        if image.shape != (512, 512):
            raise ValueError(
                f"Unexpected sample dimensions for {sample.path}: "
                f"{image.shape}"
            )
        print(
            f"{name}: {counts[name]} images; label={sample.label}; "
            "sample decoded as 512 x 512 grayscale."
        )
    print(f"Total primary images: {len(records)}")
    print("Accessibility check passed (one decoded sample per class).")
    print(
        "Initialization complete. "
        "No split, features, or models were generated."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PRIMARY_DATASET,
        help="Path containing the unsplit def_front and ok_front directories.",
    )
    args = parser.parse_args(argv)
    try:
        initialize_project(args.data_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

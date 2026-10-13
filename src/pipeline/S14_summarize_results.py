"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Summarize final segmentation and resource metrics across training seeds.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "results_master.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "results_summary.csv"
)


METRICS = [
    "test_miou",
    "test_iou_stem",
    "test_iou_leaf",
    "training_seconds",
    "evaluation_seconds",
    "training_peak_gpu_memory_mb",
    "evaluation_peak_gpu_memory_mb",
]


def load_rows():

    with INPUT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    return rows


def to_float(value):
    return float(value)


def sample_std(values):

    if len(values) < 2:
        return 0.0

    return stdev(values)


def validate_rows(rows):

    errors = []

    if len(rows) != 160:
        errors.append(
            f"Expected 160 rows, got {len(rows)}"
        )

    for idx, row in enumerate(
        rows,
        start=1,
    ):

        for metric in [
            "test_miou",
            "test_iou_stem",
            "test_iou_leaf",
            "val_miou",
            "val_iou_stem",
            "val_iou_leaf",
        ]:

            value = to_float(
                row[metric]
            )

            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"Row {idx}: "
                    f"{metric} out of range: "
                    f"{value}"
                )

        for metric in [
            "training_seconds",
            "evaluation_seconds",
            "training_peak_gpu_memory_mb",
            "evaluation_peak_gpu_memory_mb",
        ]:

            value = to_float(
                row[metric]
            )

            if value <= 0:
                errors.append(
                    f"Row {idx}: "
                    f"{metric} must be > 0, "
                    f"got {value}"
                )

    return errors


def build_summary(rows):

    groups = defaultdict(list)

    for row in rows:

        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        groups[key].append(
            row
        )

    summary_rows = []

    for key in sorted(groups):

        species, method, target = key

        group = groups[key]

        result = {
            "species": species,
            "method": method,
            "target": target,
            "n_runs": len(group),
        }

        for metric in METRICS:

            values = [
                to_float(row[metric])
                for row in group
            ]

            result[
                f"{metric}_mean"
            ] = mean(values)

            result[
                f"{metric}_std"
            ] = sample_std(values)

            result[
                f"{metric}_min"
            ] = min(values)

            result[
                f"{metric}_max"
            ] = max(values)

        summary_rows.append(
            result
        )

    return summary_rows


def save_summary(summary_rows):

    fieldnames = list(
        summary_rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


def print_compact_summary(
    summary_rows,
):

    print()
    print("=" * 100)
    print("Q5 DESCRIPTIVE SUMMARY")
    print("=" * 100)

    print(
        f"{'Species':<10} "
        f"{'Method':<15} "
        f"{'Target':>8} "
        f"{'n':>3} "
        f"{'mIoU mean':>12} "
        f"{'mIoU std':>12} "
        f"{'Stem mean':>12} "
        f"{'Leaf mean':>12}"
    )

    print("-" * 100)

    for row in summary_rows:

        print(
            f"{row['species']:<10} "
            f"{row['method']:<15} "
            f"{row['target']:>8} "
            f"{row['n_runs']:>3} "
            f"{row['test_miou_mean']:>12.4f} "
            f"{row['test_miou_std']:>12.4f} "
            f"{row['test_iou_stem_mean']:>12.4f} "
            f"{row['test_iou_leaf_mean']:>12.4f}"
        )

    print("=" * 100)


def main():

    rows = load_rows()

    errors = validate_rows(
        rows
    )

    if errors:

        print()
        print("DATA VALIDATION ERRORS")
        print("=" * 80)

        for error in errors:
            print(error)

        raise SystemExit(1)

    summary_rows = build_summary(
        rows
    )

    save_summary(
        summary_rows
    )

    print_compact_summary(
        summary_rows
    )

    print()
    print(
        "Rows validated:",
        len(rows),
        "/ 160",
    )

    print(
        "Summary groups:",
        len(summary_rows),
        "/ 32",
    )

    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "DESCRIPTIVE SUMMARY COMPLETE"
    )


if __name__ == "__main__":
    main()

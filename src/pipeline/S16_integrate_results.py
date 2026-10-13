"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Integrate frozen segmentation summaries with Chamfer and robust Hausdorff H99.
"""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

SUMMARY_INPUT = (
    ANALYSIS_DIR
    / "results_summary.csv"
)

CHAMFER_INPUT = (
    RESULTS_DIR
    / "chamfer_summary.csv"
)

H99_INPUT = (
    ANALYSIS_DIR
    / "hausdorff_h99_summary.csv"
)

OUTPUT_CSV = (
    ANALYSIS_DIR
    / "results_integrated.csv"
)


EXPECTED_TOTAL_GROUPS = 32
EXPECTED_FACTORIAL_GROUPS = 30
EXPECTED_CONTROL_GROUPS = 2

EXPECTED_SPECIES = {
    "maize",
    "tomato",
}

EXPECTED_METHODS = {
    "random",
    "voxel",
    "fps",
    "control_48000",
}

EXPECTED_TARGETS = {
    1024,
    2048,
    4096,
    8192,
    16384,
    48000,
}

CHAMFER_SUMMARY_DECIMALS = 6


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def build_index(
    rows,
    source_name,
):
    index = {}

    for row in rows:
        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        if key in index:
            raise RuntimeError(
                f"Duplicate key in {source_name}: {key}"
            )

        index[key] = row

    return index


def validate_summary_rows(rows):
    if len(rows) != EXPECTED_TOTAL_GROUPS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_GROUPS} rows "
            f"in results_summary.csv, got {len(rows)}"
        )

    seen = set()

    for row in rows:
        species = row["species"]
        method = row["method"]
        target = int(row["target"])

        key = (
            species,
            method,
            target,
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate group in results_summary.csv: {key}"
            )

        seen.add(key)

        if species not in EXPECTED_SPECIES:
            raise RuntimeError(
                f"Unexpected species: {species}"
            )

        if method not in EXPECTED_METHODS:
            raise RuntimeError(
                f"Unexpected method: {method}"
            )

        if target not in EXPECTED_TARGETS:
            raise RuntimeError(
                f"Unexpected target: {target}"
            )

        if method == "control_48000":
            if target != 48000:
                raise RuntimeError(
                    f"Control with unexpected target: {key}"
                )

        else:
            if target == 48000:
                raise RuntimeError(
                    f"Factorial method with control target: {key}"
                )


def copy_chamfer(
    result,
    chamfer_row,
):
    result[
        "chamfer_n_clouds"
    ] = chamfer_row[
        "n_clouds"
    ]

    result[
        "chamfer_mean"
    ] = chamfer_row[
        "chamfer_mean"
    ]

    result[
        "chamfer_std"
    ] = chamfer_row[
        "chamfer_std"
    ]

    result[
        "original_to_reduced_mean"
    ] = chamfer_row[
        "original_to_reduced_mean"
    ]

    result[
        "original_to_reduced_std"
    ] = chamfer_row[
        "original_to_reduced_std"
    ]


def copy_h99(
    result,
    h99_row,
):
    required = [
        "n_clouds",
        "hausdorff_percentile",
        "percentile_method",
        "hausdorff_h99_mean",
        "hausdorff_h99_std",
        "hausdorff_h99_median",
        "hausdorff_h99_min",
        "hausdorff_h99_max",
        "chamfer_recomputed_mean",
    ]

    for column in required:
        if column not in h99_row:
            raise RuntimeError(
                f"Missing H99 column: {column}"
            )

    result[
        "h99_n_clouds"
    ] = h99_row[
        "n_clouds"
    ]

    result[
        "hausdorff_percentile"
    ] = h99_row[
        "hausdorff_percentile"
    ]

    result[
        "hausdorff_percentile_method"
    ] = h99_row[
        "percentile_method"
    ]

    result[
        "hausdorff_h99_mean"
    ] = h99_row[
        "hausdorff_h99_mean"
    ]

    result[
        "hausdorff_h99_std"
    ] = h99_row[
        "hausdorff_h99_std"
    ]

    result[
        "hausdorff_h99_median"
    ] = h99_row[
        "hausdorff_h99_median"
    ]

    result[
        "hausdorff_h99_min"
    ] = h99_row[
        "hausdorff_h99_min"
    ]

    result[
        "hausdorff_h99_max"
    ] = h99_row[
        "hausdorff_h99_max"
    ]

    result[
        "h99_chamfer_recomputed_mean"
    ] = h99_row[
        "chamfer_recomputed_mean"
    ]


def fill_empty_geometry(
    result,
):
    empty_columns = [
        "chamfer_n_clouds",
        "chamfer_mean",
        "chamfer_std",
        "original_to_reduced_mean",
        "original_to_reduced_std",
        "h99_n_clouds",
        "hausdorff_percentile",
        "hausdorff_percentile_method",
        "hausdorff_h99_mean",
        "hausdorff_h99_std",
        "hausdorff_h99_median",
        "hausdorff_h99_min",
        "hausdorff_h99_max",
        "h99_chamfer_recomputed_mean",
    ]

    for column in empty_columns:
        result[column] = ""


def validate_chamfer_consistency(
    result,
    key,
):
    chamfer_mean = float(
        result[
            "chamfer_mean"
        ]
    )

    h99_chamfer_mean = float(
        result[
            "h99_chamfer_recomputed_mean"
        ]
    )

    rounded_chamfer = round(
        chamfer_mean,
        CHAMFER_SUMMARY_DECIMALS,
    )

    rounded_h99_chamfer = round(
        h99_chamfer_mean,
        CHAMFER_SUMMARY_DECIMALS,
    )

    if (
        rounded_chamfer
        != rounded_h99_chamfer
    ):
        raise RuntimeError(
            "Chamfer/H99 validation mismatch after "
            f"rounding to {CHAMFER_SUMMARY_DECIMALS} decimals "
            f"for {key}: "
            f"{rounded_chamfer} vs "
            f"{rounded_h99_chamfer}"
        )

    chamfer_n = int(
        result[
            "chamfer_n_clouds"
        ]
    )

    h99_n = int(
        result[
            "h99_n_clouds"
        ]
    )

    if chamfer_n != h99_n:
        raise RuntimeError(
            "Cloud-count mismatch between "
            f"Chamfer and H99 for {key}: "
            f"{chamfer_n} vs {h99_n}"
        )


def main():
    summary_rows = load_csv(
        SUMMARY_INPUT
    )

    chamfer_rows = load_csv(
        CHAMFER_INPUT
    )

    h99_rows = load_csv(
        H99_INPUT
    )

    validate_summary_rows(
        summary_rows
    )

    chamfer_index = build_index(
        chamfer_rows,
        "chamfer_summary.csv",
    )

    h99_index = build_index(
        h99_rows,
        "hausdorff_h99_summary.csv",
    )

    if len(chamfer_index) != EXPECTED_FACTORIAL_GROUPS:
        raise RuntimeError(
            f"Expected {EXPECTED_FACTORIAL_GROUPS} "
            f"Chamfer groups, got {len(chamfer_index)}"
        )

    if len(h99_index) != EXPECTED_FACTORIAL_GROUPS:
        raise RuntimeError(
            f"Expected {EXPECTED_FACTORIAL_GROUPS} "
            f"H99 groups, got {len(h99_index)}"
        )

    integrated = []

    factorial_count = 0
    control_count = 0

    for row in summary_rows:
        species = row["species"]
        method = row["method"]
        target = int(row["target"])

        key = (
            species,
            method,
            target,
        )

        is_control = (
            method == "control_48000"
        )

        result = dict(row)

        if is_control:
            control_count += 1

            fill_empty_geometry(
                result
            )

        else:
            factorial_count += 1

            if key not in chamfer_index:
                raise RuntimeError(
                    f"Missing Chamfer summary for {key}"
                )

            if key not in h99_index:
                raise RuntimeError(
                    f"Missing H99 summary for {key}"
                )

            copy_chamfer(
                result,
                chamfer_index[key],
            )

            copy_h99(
                result,
                h99_index[key],
            )

            validate_chamfer_consistency(
                result,
                key,
            )

        integrated.append(
            result
        )

    if (
        len(integrated)
        != EXPECTED_TOTAL_GROUPS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_GROUPS} "
            f"integrated groups, got "
            f"{len(integrated)}"
        )

    if (
        factorial_count
        != EXPECTED_FACTORIAL_GROUPS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_FACTORIAL_GROUPS} "
            f"factorial groups, got "
            f"{factorial_count}"
        )

    if (
        control_count
        != EXPECTED_CONTROL_GROUPS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROL_GROUPS} "
            f"control groups, got "
            f"{control_count}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        integrated[0].keys()
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
            integrated
        )

    print()
    print("=" * 90)
    print("Q5 RESULTS INTEGRATION")
    print("=" * 90)

    print(
        "Segmentation groups:",
        len(summary_rows),
        "/ 32",
    )

    print(
        "Chamfer groups:",
        len(chamfer_index),
        "/ 30",
    )

    print(
        "H99 groups:",
        len(h99_index),
        "/ 30",
    )

    print(
        "Factorial groups:",
        factorial_count,
        "/ 30",
    )

    print(
        "Control groups:",
        control_count,
        "/ 2",
    )

    print(
        "Total integrated groups:",
        len(integrated),
        "/ 32",
    )

    print(
        "Chamfer/H99 summary consistency:",
        f"validated at {CHAMFER_SUMMARY_DECIMALS} stored decimals",
    )

    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "INTEGRATION COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()

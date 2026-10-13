"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Combine final segmentation, geometry, sampling, and inference measurements.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INTEGRATED_INPUT = (
    ANALYSIS_DIR
    / "results_integrated.csv"
)

INFERENCE_INPUT = (
    ANALYSIS_DIR
    / "inference_benchmark_summary.csv"
)

SAMPLING_INPUT = (
    PROCESSED_DIR
    / "metadata.csv"
)

CONTROL_SAMPLING_INPUT = (
    PROCESSED_DIR
    / "control_48000_metadata.csv"
)

OUTPUT_CSV = (
    ANALYSIS_DIR
    / "results_tradeoff.csv"
)


def load_csv(path):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def sample_std(values):

    if len(values) < 2:
        return 0.0

    return stdev(values)


def build_sampling_summary():

    rows = load_csv(
        SAMPLING_INPUT
    )

    groups = defaultdict(list)

    for row in rows:

        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        groups[key].append(
            float(
                row["sampling_seconds"]
            )
        )

    # ---------------------------------------------
    # CONTROL 48k
    # ---------------------------------------------

    control_rows = load_csv(
        CONTROL_SAMPLING_INPUT
    )

    for row in control_rows:

        key = (
            row["species"],
            "control_48000",
            int(row["target"]),
        )

        groups[key].append(
            float(
                row["sampling_seconds"]
            )
        )

    summary = {}

    for key, values in groups.items():

        summary[key] = {
            "sampling_n_clouds":
                len(values),

            "sampling_seconds_mean":
                mean(values),

            "sampling_seconds_std":
                sample_std(values),

            "sampling_ms_mean":
                mean(values) * 1000.0,

            "sampling_ms_std":
                sample_std(values) * 1000.0,

            "sampling_seconds_min":
                min(values),

            "sampling_seconds_max":
                max(values),
        }

    return summary


def build_inference_index():

    rows = load_csv(
        INFERENCE_INPUT
    )

    index = {}

    for row in rows:

        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        index[key] = row

    return index


def main():

    integrated_rows = load_csv(
        INTEGRATED_INPUT
    )

    sampling = build_sampling_summary()

    inference = build_inference_index()

    output_rows = []

    for row in integrated_rows:

        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        if key not in sampling:

            raise RuntimeError(
                f"Missing sampling summary for {key}"
            )

        if key not in inference:

            raise RuntimeError(
                f"Missing inference summary for {key}"
            )

        sampling_row = sampling[key]

        inference_row = inference[key]

        result = dict(row)

        # -----------------------------------------
        # SAMPLING
        # -----------------------------------------

        result.update(
            sampling_row
        )

        # -----------------------------------------
        # INFERENCE
        # -----------------------------------------

        result[
            "inference_n_seeds"
        ] = inference_row[
            "n_seeds"
        ]

        result[
            "inference_seconds_per_cloud_mean"
        ] = inference_row[
            "inference_seconds_per_cloud_mean"
        ]

        result[
            "inference_seconds_per_cloud_std"
        ] = inference_row[
            "inference_seconds_per_cloud_std"
        ]

        result[
            "inference_ms_per_cloud_mean"
        ] = inference_row[
            "inference_ms_per_cloud_mean"
        ]

        result[
            "inference_ms_per_cloud_std"
        ] = inference_row[
            "inference_ms_per_cloud_std"
        ]

        result[
            "inference_peak_gpu_memory_mb_mean"
        ] = inference_row[
            "inference_peak_gpu_memory_mb_mean"
        ]

        result[
            "inference_peak_gpu_memory_mb_std"
        ] = inference_row[
            "inference_peak_gpu_memory_mb_std"
        ]

        result[
            "inference_peak_gpu_memory_mb_max"
        ] = inference_row[
            "inference_peak_gpu_memory_mb_max"
        ]

        output_rows.append(
            result
        )

    # ---------------------------------------------
    # VALIDACIONES
    # ---------------------------------------------

    if len(output_rows) != 32:

        raise RuntimeError(
            f"Expected 32 rows, got "
            f"{len(output_rows)}"
        )

    # ---------------------------------------------
    # GUARDAR
    # ---------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    print()
    print("=" * 90)
    print("Q5 TRADE-OFF TABLE")
    print("=" * 90)

    print(
        "Rows:",
        len(output_rows),
        "/ 32",
    )

    print(
        "Sampling groups:",
        len(sampling),
    )

    print(
        "Inference groups:",
        len(inference),
    )

    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "TRADE-OFF TABLE COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()

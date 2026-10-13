"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Compare factorial configurations with the practical high-resolution control.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_INPUT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "results_master.csv"
)

TRADEOFF_INPUT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "results_tradeoff.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "operating_points.csv"
)


SPECIES = [
    "maize",
    "tomato",
]

METHODS = [
    "random",
    "voxel",
    "fps",
]

TARGETS = [
    1024,
    2048,
    4096,
    8192,
    16384,
]

EXPECTED_SEEDS = {
    0,
    1,
    2,
    3,
    4,
}

CONTROL_METHOD = "control_48000"
CONTROL_TARGET = 48000

ALPHA = 0.05
EXPECTED_MASTER_ROWS = 160
EXPECTED_TRADEOFF_ROWS = 32
EXPECTED_COMPARISONS = 30

SUMMARY_TOLERANCE = 1e-9
ZERO_TOLERANCE = 1e-15
TIE_TOLERANCE = 1e-15


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def parse_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def get_float(row, column):
    value = row.get(column)

    if (
        value is None
        or str(value).strip() == ""
    ):
        return float("nan")

    return float(value)


def percent_change_from_control(
    control_value,
    configuration_value,
):
    if (
        not math.isfinite(control_value)
        or not math.isfinite(configuration_value)
        or control_value == 0
    ):
        return float("nan")

    return (
        100.0
        * (
            configuration_value
            - control_value
        )
        / control_value
    )


def percent_saving(
    control_value,
    configuration_value,
):
    if (
        not math.isfinite(control_value)
        or not math.isfinite(configuration_value)
        or control_value == 0
    ):
        return float("nan")

    return (
        100.0
        * (
            control_value
            - configuration_value
        )
        / control_value
    )


def holm_adjust(p_values):
    """
    Holm step-down adjustment.

    Input:
        raw p-values from one family of hypotheses.

    Output:
        adjusted p-values in the same order.
    """

    m = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (
        original_index,
        p_value,
    ) in enumerate(indexed):

        multiplier = m - rank

        candidate = min(
            1.0,
            p_value * multiplier,
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[
            original_index
        ] = running_max

    return adjusted


def matched_pairs_rank_biserial(
    differences,
):
    """
    Matched-pairs rank-biserial correlation.

    Difference convention:
        configuration - control

    Interpretation of sign:
        positive -> higher mIoU in configuration
        negative -> lower mIoU in configuration

    Range:
        -1 to +1
    """

    nonzero = [
        value
        for value in differences
        if abs(value) > ZERO_TOLERANCE
    ]

    if not nonzero:
        return 0.0

    absolute_differences = [
        abs(value)
        for value in nonzero
    ]

    ranks = rankdata(
        absolute_differences,
        method="average",
    )

    positive_rank_sum = sum(
        rank
        for rank, difference
        in zip(
            ranks,
            nonzero,
        )
        if difference > 0
    )

    negative_rank_sum = sum(
        rank
        for rank, difference
        in zip(
            ranks,
            nonzero,
        )
        if difference < 0
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / denominator
    )


def count_signs(differences):
    positive = sum(
        value > ZERO_TOLERANCE
        for value in differences
    )

    negative = sum(
        value < -ZERO_TOLERANCE
        for value in differences
    )

    zero = (
        len(differences)
        - positive
        - negative
    )

    return (
        positive,
        negative,
        zero,
    )


def validate_exact_wilcoxon_conditions(
    differences,
    key,
):
    """
    SciPy's textbook exact Wilcoxon calculation is most
    straightforward when there are no zero differences
    and no ties among absolute differences.

    Q5 currently satisfies these conditions.

    If future data violate them, stop rather than silently
    switching to an approximation.
    """

    zeros = [
        value
        for value in differences
        if abs(value) <= ZERO_TOLERANCE
    ]

    if zeros:
        raise RuntimeError(
            "Exact Wilcoxon validation failed for "
            f"{key}: zero paired difference detected. "
            "Do not silently replace the statistical method."
        )

    absolute = sorted(
        abs(value)
        for value in differences
    )

    for left, right in zip(
        absolute[:-1],
        absolute[1:],
    ):
        if abs(left - right) <= TIE_TOLERANCE:
            raise RuntimeError(
                "Exact Wilcoxon validation failed for "
                f"{key}: tied absolute differences detected. "
                "Do not silently replace the statistical method."
            )


def exact_wilcoxon(
    configuration_values,
    control_values,
    key,
):
    differences = [
        configuration - control
        for configuration, control
        in zip(
            configuration_values,
            control_values,
        )
    ]

    validate_exact_wilcoxon_conditions(
        differences,
        key,
    )

    result = wilcoxon(
        configuration_values,
        control_values,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="exact",
    )

    return {
        "statistic":
            float(result.statistic),

        "p_value":
            float(result.pvalue),

        "differences":
            differences,
    }


def validate_master(rows):
    if len(rows) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MASTER_ROWS} rows "
            f"in results_master.csv, got {len(rows)}"
        )

    controls = defaultdict(dict)
    factorial = defaultdict(dict)

    for row in rows:
        species = row["species"]
        method = row["method"]
        target = int(row["target"])
        seed = int(row["seed"])

        is_control = parse_bool(
            row["is_control"]
        )

        test_miou = float(
            row["test_miou"]
        )

        if species not in SPECIES:
            raise RuntimeError(
                f"Unexpected species: {species}"
            )

        if is_control:
            if (
                method != CONTROL_METHOD
                or target != CONTROL_TARGET
            ):
                raise RuntimeError(
                    "Unexpected control identifier: "
                    f"{species}, {method}, "
                    f"{target}, seed={seed}"
                )

            if seed in controls[species]:
                raise RuntimeError(
                    "Duplicate control seed: "
                    f"{species}, seed={seed}"
                )

            controls[
                species
            ][seed] = test_miou

        else:
            if method not in METHODS:
                raise RuntimeError(
                    f"Unexpected factorial method: {method}"
                )

            if target not in TARGETS:
                raise RuntimeError(
                    f"Unexpected factorial target: {target}"
                )

            key = (
                species,
                method,
                target,
            )

            if seed in factorial[key]:
                raise RuntimeError(
                    "Duplicate factorial run: "
                    f"{key}, seed={seed}"
                )

            factorial[
                key
            ][seed] = test_miou

    for species in SPECIES:
        found_seeds = set(
            controls[species]
        )

        if found_seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"Invalid control seeds for {species}: "
                f"{sorted(found_seeds)}"
            )

    expected_groups = (
        len(SPECIES)
        * len(METHODS)
        * len(TARGETS)
    )

    if len(factorial) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} factorial groups, "
            f"got {len(factorial)}"
        )

    for key, values in factorial.items():
        found_seeds = set(values)

        if found_seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"Invalid seed set for {key}: "
                f"{sorted(found_seeds)}"
            )

    return (
        controls,
        factorial,
    )


def build_tradeoff_index(rows):
    if len(rows) != EXPECTED_TRADEOFF_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRADEOFF_ROWS} rows "
            f"in results_tradeoff.csv, got {len(rows)}"
        )

    index = {}

    for row in rows:
        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        if key in index:
            raise RuntimeError(
                f"Duplicate tradeoff row: {key}"
            )

        index[key] = row

    return index


def validate_summary_miou(
    values,
    tradeoff_row,
    key,
):
    calculated = mean(values)

    stored = get_float(
        tradeoff_row,
        "test_miou_mean",
    )

    if not math.isfinite(stored):
        raise RuntimeError(
            f"Missing stored mIoU mean for {key}"
        )

    error = abs(
        calculated - stored
    )

    if error > SUMMARY_TOLERANCE:
        raise RuntimeError(
            "Mismatch between results_master.csv and "
            "results_tradeoff.csv for "
            f"{key}: error={error:.12g}"
        )


def observed_direction(delta):
    if delta > ZERO_TOLERANCE:
        return "observed_improvement"

    if delta < -ZERO_TOLERANCE:
        return "observed_loss"

    return "no_observed_mean_difference"


def inferential_status(
    delta,
    adjusted_p,
):
    if adjusted_p >= ALPHA:
        return (
            "no_significant_difference_detected"
        )

    if delta < 0:
        return "significant_loss"

    if delta > 0:
        return "significant_improvement"

    return "significant_zero_mean_delta"


def build_rows(
    controls,
    factorial,
    tradeoff_index,
):
    output_rows = []

    for species in SPECIES:
        seeds = sorted(
            EXPECTED_SEEDS
        )

        control_values = [
            controls[
                species
            ][seed]
            for seed in seeds
        ]

        control_key = (
            species,
            CONTROL_METHOD,
            CONTROL_TARGET,
        )

        control_cost = (
            tradeoff_index[
                control_key
            ]
        )

        validate_summary_miou(
            control_values,
            control_cost,
            control_key,
        )

        control_miou_mean = mean(
            control_values
        )

        control_sampling_ms = get_float(
            control_cost,
            "sampling_ms_mean",
        )

        control_inference_ms = get_float(
            control_cost,
            "inference_ms_per_cloud_mean",
        )

        control_combined_ms = (
            control_sampling_ms
            + control_inference_ms
        )

        control_training_seconds = get_float(
            control_cost,
            "training_seconds_mean",
        )

        control_training_gpu_mb = get_float(
            control_cost,
            "training_peak_gpu_memory_mb_mean",
        )

        control_inference_gpu_mb = get_float(
            control_cost,
            "inference_peak_gpu_memory_mb_mean",
        )

        species_rows = []

        for method in METHODS:
            for target in TARGETS:
                key = (
                    species,
                    method,
                    target,
                )

                values_by_seed = (
                    factorial[key]
                )

                configuration_values = [
                    values_by_seed[
                        seed
                    ]
                    for seed in seeds
                ]

                tradeoff = (
                    tradeoff_index[key]
                )

                validate_summary_miou(
                    configuration_values,
                    tradeoff,
                    key,
                )

                test = exact_wilcoxon(
                    configuration_values,
                    control_values,
                    key,
                )

                differences = test[
                    "differences"
                ]

                delta_mean = mean(
                    differences
                )

                delta_median = median(
                    differences
                )

                (
                    n_positive,
                    n_negative,
                    n_zero,
                ) = count_signs(
                    differences
                )

                sampling_ms = get_float(
                    tradeoff,
                    "sampling_ms_mean",
                )

                inference_ms = get_float(
                    tradeoff,
                    "inference_ms_per_cloud_mean",
                )

                combined_ms = (
                    sampling_ms
                    + inference_ms
                )

                training_seconds = get_float(
                    tradeoff,
                    "training_seconds_mean",
                )

                training_gpu_mb = get_float(
                    tradeoff,
                    "training_peak_gpu_memory_mb_mean",
                )

                inference_gpu_mb = get_float(
                    tradeoff,
                    "inference_peak_gpu_memory_mb_mean",
                )

                row = {
                    "species":
                        species,

                    "method":
                        method,

                    "target":
                        target,

                    "n_pairs":
                        len(differences),

                    "seed_order":
                        "0|1|2|3|4",

                    "control_method":
                        CONTROL_METHOD,

                    "control_target":
                        CONTROL_TARGET,

                    "test_miou_mean":
                        mean(
                            configuration_values
                        ),

                    "control_miou_mean":
                        control_miou_mean,

                    "delta_miou_mean_config_minus_control":
                        delta_mean,

                    "delta_miou_median_config_minus_control":
                        delta_median,

                    "relative_delta_miou_pct_vs_control":
                        (
                            100.0
                            * delta_mean
                            / control_miou_mean
                        ),

                    "n_seed_differences_positive":
                        n_positive,

                    "n_seed_differences_negative":
                        n_negative,

                    "n_seed_differences_zero":
                        n_zero,

                    "delta_seed_0":
                        differences[0],

                    "delta_seed_1":
                        differences[1],

                    "delta_seed_2":
                        differences[2],

                    "delta_seed_3":
                        differences[3],

                    "delta_seed_4":
                        differences[4],

                    "wilcoxon_statistic":
                        test[
                            "statistic"
                        ],

                    "p_value_raw_exact":
                        test[
                            "p_value"
                        ],

                    "p_value_holm":
                        None,

                    "holm_significant_alpha_0_05":
                        None,

                    "rank_biserial_r":
                        matched_pairs_rank_biserial(
                            differences
                        ),

                    "observed_direction":
                        observed_direction(
                            delta_mean
                        ),

                    "inferential_status":
                        None,

                    "equivalence_demonstrated":
                        False,

                    "sampling_ms_mean":
                        sampling_ms,

                    "control_sampling_ms_mean":
                        control_sampling_ms,

                    "sampling_saving_ms":
                        (
                            control_sampling_ms
                            - sampling_ms
                        ),

                    "sampling_saving_pct":
                        percent_saving(
                            control_sampling_ms,
                            sampling_ms,
                        ),

                    "inference_ms_per_cloud_mean":
                        inference_ms,

                    "control_inference_ms_per_cloud_mean":
                        control_inference_ms,

                    "inference_saving_ms_per_cloud":
                        (
                            control_inference_ms
                            - inference_ms
                        ),

                    "inference_saving_pct":
                        percent_saving(
                            control_inference_ms,
                            inference_ms,
                        ),

                    "preprocess_plus_inference_ms_per_cloud":
                        combined_ms,

                    "control_preprocess_plus_inference_ms_per_cloud":
                        control_combined_ms,

                    "combined_saving_ms_per_cloud":
                        (
                            control_combined_ms
                            - combined_ms
                        ),

                    "combined_saving_pct":
                        percent_saving(
                            control_combined_ms,
                            combined_ms,
                        ),

                    "training_seconds_mean":
                        training_seconds,

                    "control_training_seconds_mean":
                        control_training_seconds,

                    "training_saving_seconds":
                        (
                            control_training_seconds
                            - training_seconds
                        ),

                    "training_saving_pct":
                        percent_saving(
                            control_training_seconds,
                            training_seconds,
                        ),

                    "training_peak_gpu_memory_mb_mean":
                        training_gpu_mb,

                    "control_training_peak_gpu_memory_mb_mean":
                        control_training_gpu_mb,

                    "training_gpu_memory_saving_mb":
                        (
                            control_training_gpu_mb
                            - training_gpu_mb
                        ),

                    "training_gpu_memory_saving_pct":
                        percent_saving(
                            control_training_gpu_mb,
                            training_gpu_mb,
                        ),

                    "inference_peak_gpu_memory_mb_mean":
                        inference_gpu_mb,

                    "control_inference_peak_gpu_memory_mb_mean":
                        control_inference_gpu_mb,

                    "inference_gpu_memory_saving_mb":
                        (
                            control_inference_gpu_mb
                            - inference_gpu_mb
                        ),

                    "inference_gpu_memory_saving_pct":
                        percent_saving(
                            control_inference_gpu_mb,
                            inference_gpu_mb,
                        ),

                    "chamfer_mean":
                        get_float(
                            tradeoff,
                            "chamfer_mean",
                        ),
                }

                species_rows.append(
                    row
                )

        raw_p_values = [
            row[
                "p_value_raw_exact"
            ]
            for row in species_rows
        ]

        adjusted_p_values = (
            holm_adjust(
                raw_p_values
            )
        )

        for row, adjusted_p in zip(
            species_rows,
            adjusted_p_values,
        ):
            row[
                "p_value_holm"
            ] = adjusted_p

            significant = (
                adjusted_p < ALPHA
            )

            row[
                "holm_significant_alpha_0_05"
            ] = significant

            row[
                "inferential_status"
            ] = inferential_status(
                row[
                    "delta_miou_mean_config_minus_control"
                ],
                adjusted_p,
            )

        output_rows.extend(
            species_rows
        )

    output_rows.sort(
        key=lambda row: (
            row["species"],
            row["method"],
            row["target"],
        )
    )

    return output_rows


def save_csv(rows):
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def print_summary(rows):
    print()
    print("=" * 152)

    print(
        "Q5 CONTROL COMPARISONS — "
        "PRACTICAL HIGH-RESOLUTION CONTROL"
    )

    print("=" * 152)

    print(
        "Test: paired two-sided exact Wilcoxon signed-rank."
    )

    print(
        "Multiplicity correction: Holm within each species "
        "(15 comparisons per species)."
    )

    print(
        "Effect size: matched-pairs rank-biserial correlation."
    )

    print(
        "Difference convention: configuration - control."
    )

    print(
        "IMPORTANT: p >= 0.05 means only that no statistically "
        "significant difference was detected."
    )

    print(
        "It does NOT demonstrate equivalence."
    )

    print(
        "No operating point is selected automatically by this script."
    )

    theoretical_minimum_p = (
        2.0
        / (
            2 ** len(
                EXPECTED_SEEDS
            )
        )
    )

    print(
        "With n=5 non-zero untied pairs, the smallest possible "
        "two-sided exact Wilcoxon p-value is "
        f"{theoretical_minimum_p:.4f}."
    )

    for species in SPECIES:
        print()
        print(
            species.upper()
        )

        print("-" * 152)

        print(
            f"{'Method':<9} "
            f"{'Target':>7} "
            f"{'mIoU':>8} "
            f"{'Control':>8} "
            f"{'Delta':>9} "
            f"{'+/-':>5} "
            f"{'p exact':>9} "
            f"{'p Holm':>9} "
            f"{'r_rb':>7} "
            f"{'Sample ms':>11} "
            f"{'Infer ms':>10} "
            f"{'Infer save%':>11}"
        )

        species_rows = [
            row
            for row in rows
            if row["species"] == species
        ]

        for row in species_rows:
            signs = (
                f"{row['n_seed_differences_positive']}"
                "/"
                f"{row['n_seed_differences_negative']}"
            )

            print(
                f"{row['method']:<9} "
                f"{row['target']:>7} "
                f"{row['test_miou_mean']:>8.4f} "
                f"{row['control_miou_mean']:>8.4f} "
                f"{row['delta_miou_mean_config_minus_control']:>+9.4f} "
                f"{signs:>5} "
                f"{row['p_value_raw_exact']:>9.4f} "
                f"{row['p_value_holm']:>9.4f} "
                f"{row['rank_biserial_r']:>+7.3f} "
                f"{row['sampling_ms_mean']:>11.3f} "
                f"{row['inference_ms_per_cloud_mean']:>10.3f} "
                f"{row['inference_saving_pct']:>11.2f}"
            )

        significant_count = sum(
            bool(
                row[
                    "holm_significant_alpha_0_05"
                ]
            )
            for row in species_rows
        )

        observed_losses = sum(
            row[
                "delta_miou_mean_config_minus_control"
            ] < 0
            for row in species_rows
        )

        observed_improvements = sum(
            row[
                "delta_miou_mean_config_minus_control"
            ] > 0
            for row in species_rows
        )

        print()
        print(
            "Observed mean direction: "
            f"{observed_improvements} improvements, "
            f"{observed_losses} losses."
        )

        print(
            "Holm-significant comparisons at alpha=0.05: "
            f"{significant_count}/15."
        )

    print()
    print("=" * 152)

    print(
        "Interpretation rule:"
    )

    print(
        "- Delta mIoU quantifies the observed performance change."
    )

    print(
        "- Rank-biserial quantifies direction and magnitude "
        "of the paired effect."
    )

    print(
        "- Holm-adjusted p-values control the family-wise "
        "error rate across the 15 comparisons per species."
    )

    print(
        "- A non-significant result must NOT be interpreted "
        "as equivalence to the control."
    )

    print(
        "- Cost metrics are reported separately; this script "
        "does not collapse them into an automatic winner."
    )

    print()
    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "CONTROL COMPARISON ANALYSIS COMPLETE"
    )


def main():
    master_rows = load_csv(
        MASTER_INPUT
    )

    tradeoff_rows = load_csv(
        TRADEOFF_INPUT
    )

    (
        controls,
        factorial,
    ) = validate_master(
        master_rows
    )

    tradeoff_index = (
        build_tradeoff_index(
            tradeoff_rows
        )
    )

    output_rows = build_rows(
        controls,
        factorial,
        tradeoff_index,
    )

    if (
        len(output_rows)
        != EXPECTED_COMPARISONS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_COMPARISONS} "
            f"comparisons, got {len(output_rows)}"
        )

    save_csv(
        output_rows
    )

    print_summary(
        output_rows
    )


if __name__ == "__main__":
    main()
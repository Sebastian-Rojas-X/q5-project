"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Run frozen local post-hoc comparisons and the global Friedman test.
"""

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "results_master.csv"
)

FRIEDMAN_INPUT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "friedman_test_miou.csv"
)

POSTHOC_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "posthoc_test_miou.csv"
)

GLOBAL_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "friedman_global_test_miou.csv"
)


METHODS = [
    "random",
    "voxel",
    "fps",
]

ALPHA = 0.05


def load_csv(path):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def holm_adjust(p_values):

    """
    Corrección Holm step-down.

    Retorna los p-values ajustados
    en el mismo orden de entrada.
    """

    m = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [0.0] * m

    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(
        indexed
    ):

        multiplier = m - rank

        adjusted_value = min(
            1.0,
            p_value * multiplier,
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted[
            original_index
        ] = running_max

    return adjusted


def rank_biserial_from_differences(
    differences,
):

    """
    Rank-biserial correlation para
    datos apareados.

    Positivo:
        method_a > method_b

    Negativo:
        method_a < method_b
    """

    nonzero = [
        value
        for value in differences
        if value != 0
    ]

    if not nonzero:
        return 0.0

    absolute = [
        abs(value)
        for value in nonzero
    ]

    # Ranking manual con promedio para empates.
    order = sorted(
        range(len(absolute)),
        key=lambda i: absolute[i],
    )

    ranks = [0.0] * len(absolute)

    position = 0

    while position < len(order):

        end = position + 1

        while (
            end < len(order)
            and absolute[
                order[end]
            ]
            == absolute[
                order[position]
            ]
        ):
            end += 1

        average_rank = (
            (position + 1 + end)
            / 2.0
        )

        for k in range(
            position,
            end,
        ):
            ranks[
                order[k]
            ] = average_rank

        position = end

    positive_sum = sum(
        rank
        for rank, diff
        in zip(
            ranks,
            nonzero,
        )
        if diff > 0
    )

    negative_sum = sum(
        rank
        for rank, diff
        in zip(
            ranks,
            nonzero,
        )
        if diff < 0
    )

    total = (
        positive_sum
        + negative_sum
    )

    if total == 0:
        return 0.0

    return (
        positive_sum
        - negative_sum
    ) / total


def build_factorial_data(rows):

    data = defaultdict(dict)

    for row in rows:

        method = row["method"]

        if method not in METHODS:
            continue

        species = row["species"]
        target = int(row["target"])
        seed = int(row["seed"])

        key = (
            species,
            target,
            seed,
        )

        data[key][method] = float(
            row["test_miou"]
        )

    return data


def run_posthoc(
    rows,
    friedman_rows,
):

    data = build_factorial_data(
        rows
    )

    significant_groups = []

    for row in friedman_rows:

        significant = (
            row["significant"]
            .strip()
            .lower()
            == "true"
        )

        if significant:

            significant_groups.append(
                (
                    row["species"],
                    int(row["target"]),
                )
            )

    results = []

    print()
    print("=" * 110)
    print("POST-HOC WILCOXON + HOLM — TEST mIoU")
    print("=" * 110)

    for species, target in significant_groups:

        print()
        print(
            f"{species.upper()} | "
            f"target = {target}"
        )

        print("-" * 110)

        comparisons = []

        for method_a, method_b in combinations(
            METHODS,
            2,
        ):

            values_a = []
            values_b = []

            for seed in range(5):

                key = (
                    species,
                    target,
                    seed,
                )

                methods = data[key]

                if not all(
                    method in methods
                    for method in [
                        method_a,
                        method_b,
                    ]
                ):
                    raise RuntimeError(
                        f"Missing pair: "
                        f"{key} "
                        f"{method_a} "
                        f"{method_b}"
                    )

                values_a.append(
                    methods[method_a]
                )

                values_b.append(
                    methods[method_b]
                )

            differences = [
                a - b
                for a, b
                in zip(
                    values_a,
                    values_b,
                )
            ]

            if all(
                diff == 0
                for diff in differences
            ):

                statistic = 0.0
                p_value = 1.0

            else:

                result = wilcoxon(
                    values_a,
                    values_b,
                    alternative="two-sided",
                    method="auto",
                )

                statistic = float(
                    result.statistic
                )

                p_value = float(
                    result.pvalue
                )

            effect = (
                rank_biserial_from_differences(
                    differences
                )
            )

            mean_difference = (
                sum(differences)
                / len(differences)
            )

            comparisons.append(
                {
                    "species": species,
                    "target": target,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": len(values_a),
                    "wilcoxon_statistic": statistic,
                    "p_value_raw": p_value,
                    "mean_difference_a_minus_b":
                        mean_difference,
                    "rank_biserial_r": effect,
                }
            )

        adjusted = holm_adjust(
            [
                item["p_value_raw"]
                for item in comparisons
            ]
        )

        for item, p_adjusted in zip(
            comparisons,
            adjusted,
        ):

            item["p_value_holm"] = (
                p_adjusted
            )

            item["significant_holm"] = (
                p_adjusted < ALPHA
            )

            results.append(item)

            print(
                f"{item['method_a']:<8} vs "
                f"{item['method_b']:<8} | "
                f"p={item['p_value_raw']:.6f} | "
                f"Holm={item['p_value_holm']:.6f} | "
                f"r_rb={item['rank_biserial_r']:+.3f} | "
                f"Δ={item['mean_difference_a_minus_b']:+.4f} | "
                f"sig={item['significant_holm']}"
            )

    if results:

        with POSTHOC_OUTPUT.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            fieldnames = list(
                results[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

    return results


def run_global_friedman(rows):

    data = build_factorial_data(
        rows
    )

    random_values = []
    voxel_values = []
    fps_values = []

    blocks = []

    for key in sorted(data):

        methods = data[key]

        if not all(
            method in methods
            for method in METHODS
        ):
            continue

        blocks.append(key)

        random_values.append(
            methods["random"]
        )

        voxel_values.append(
            methods["voxel"]
        )

        fps_values.append(
            methods["fps"]
        )

    if len(blocks) != 50:

        raise RuntimeError(
            f"Expected 50 global blocks, "
            f"got {len(blocks)}"
        )

    statistic, p_value = (
        friedmanchisquare(
            random_values,
            voxel_values,
            fps_values,
        )
    )

    # Kendall's W para Friedman:
    # W = chi2 / [n * (k - 1)]
    n_blocks = len(blocks)
    k_methods = len(METHODS)

    kendalls_w = (
        statistic
        / (
            n_blocks
            * (k_methods - 1)
        )
    )

    result = {
        "metric": "test_miou",
        "n_blocks": n_blocks,
        "n_methods": k_methods,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": p_value < ALPHA,
        "kendalls_w": kendalls_w,
    }

    with GLOBAL_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                result.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(result)

    print()
    print("=" * 90)
    print("GLOBAL FRIEDMAN — TEST mIoU")
    print("=" * 90)

    print(
        "Blocks:",
        n_blocks,
    )

    print(
        "Methods:",
        ", ".join(METHODS),
    )

    print(
        f"Statistic: {statistic:.6f}"
    )

    print(
        f"p-value:   {p_value:.10f}"
    )

    print(
        f"Kendall W: {kendalls_w:.6f}"
    )

    print(
        "Significant:",
        p_value < ALPHA,
    )

    print("=" * 90)

    return result


def main():

    rows = load_csv(
        INPUT_CSV
    )

    friedman_rows = load_csv(
        FRIEDMAN_INPUT
    )

    posthoc_results = run_posthoc(
        rows,
        friedman_rows,
    )

    global_result = run_global_friedman(
        rows
    )

    print()
    print(
        "Post-hoc output:",
        POSTHOC_OUTPUT.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Global output:",
        GLOBAL_OUTPUT.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "POST-HOC + GLOBAL FRIEDMAN COMPLETE"
    )


if __name__ == "__main__":
    main()

"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Run frozen global paired comparisons with Holm correction and effect sizes.
"""

import csv
from itertools import combinations
from pathlib import Path

from scipy.stats import wilcoxon


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
    / "posthoc_global_test_miou.csv"
)

METHODS = [
    "random",
    "voxel",
    "fps",
]

ALPHA = 0.05


def load_rows():

    with INPUT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def holm_adjust(p_values):

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
            and absolute[order[end]]
            == absolute[order[position]]
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


def build_blocks(rows):

    blocks = {}

    for row in rows:

        method = row["method"]

        if method not in METHODS:
            continue

        key = (
            row["species"],
            int(row["target"]),
            int(row["seed"]),
        )

        if key not in blocks:
            blocks[key] = {}

        blocks[key][method] = float(
            row["test_miou"]
        )

    return blocks


def main():

    rows = load_rows()

    blocks = build_blocks(
        rows
    )

    complete_blocks = []

    for key in sorted(blocks):

        methods = blocks[key]

        if all(
            method in methods
            for method in METHODS
        ):

            complete_blocks.append(
                (
                    key,
                    methods,
                )
            )

    if len(complete_blocks) != 50:
        raise RuntimeError(
            f"Expected 50 complete blocks, "
            f"got {len(complete_blocks)}"
        )

    comparisons = []

    for method_a, method_b in combinations(
        METHODS,
        2,
    ):

        values_a = []
        values_b = []

        for _, methods in complete_blocks:

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

        result = wilcoxon(
            values_a,
            values_b,
            alternative="two-sided",
            method="auto",
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

        wins_a = sum(
            diff > 0
            for diff in differences
        )

        wins_b = sum(
            diff < 0
            for diff in differences
        )

        ties = sum(
            diff == 0
            for diff in differences
        )

        comparisons.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_blocks": len(differences),
                "wilcoxon_statistic":
                    float(result.statistic),
                "p_value_raw":
                    float(result.pvalue),
                "mean_difference_a_minus_b":
                    mean_difference,
                "rank_biserial_r":
                    effect,
                "wins_method_a":
                    wins_a,
                "wins_method_b":
                    wins_b,
                "ties":
                    ties,
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

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                comparisons[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            comparisons
        )

    print()
    print("=" * 120)
    print("GLOBAL POST-HOC — TEST mIoU")
    print("=" * 120)

    for item in comparisons:

        print(
            f"{item['method_a']:<8} vs "
            f"{item['method_b']:<8} | "
            f"p={item['p_value_raw']:.10f} | "
            f"Holm={item['p_value_holm']:.10f} | "
            f"r_rb={item['rank_biserial_r']:+.3f} | "
            f"Δ={item['mean_difference_a_minus_b']:+.4f} | "
            f"wins={item['wins_method_a']}-"
            f"{item['wins_method_b']} | "
            f"ties={item['ties']} | "
            f"sig={item['significant_holm']}"
        )

    print("=" * 120)

    print()
    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "GLOBAL POST-HOC COMPLETE"
    )


if __name__ == "__main__":
    main()

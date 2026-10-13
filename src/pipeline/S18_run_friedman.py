"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Run the frozen Friedman comparisons for each species and cardinality.
"""

import csv
from collections import defaultdict
from pathlib import Path

from scipy.stats import friedmanchisquare


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
    / "friedman_test_miou.csv"
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


def build_groups(rows):

    groups = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in rows:

        if row["method"] not in METHODS:
            continue

        species = row["species"]
        target = int(row["target"])
        seed = int(row["seed"])
        method = row["method"]

        groups[
            (species, target)
        ][seed][method] = float(
            row["test_miou"]
        )

    return groups


def main():

    rows = load_rows()

    groups = build_groups(
        rows
    )

    results = []

    print()
    print("=" * 90)
    print("FRIEDMAN TEST — TEST mIoU")
    print("=" * 90)

    print(
        f"{'Species':<10} "
        f"{'Target':>8} "
        f"{'n':>4} "
        f"{'Statistic':>14} "
        f"{'p-value':>14} "
        f"{'Significant':>12}"
    )

    print("-" * 90)

    for species, target in sorted(groups):

        seed_data = groups[
            (species, target)
        ]

        random_values = []
        voxel_values = []
        fps_values = []

        valid_seeds = []

        for seed in sorted(seed_data):

            methods_for_seed = (
                seed_data[seed]
            )

            if not all(
                method in methods_for_seed
                for method in METHODS
            ):
                continue

            valid_seeds.append(seed)

            random_values.append(
                methods_for_seed["random"]
            )

            voxel_values.append(
                methods_for_seed["voxel"]
            )

            fps_values.append(
                methods_for_seed["fps"]
            )

        if len(valid_seeds) != 5:
            raise RuntimeError(
                f"{species} {target}: "
                f"expected 5 paired seeds, "
                f"got {len(valid_seeds)}"
            )

        statistic, p_value = (
            friedmanchisquare(
                random_values,
                voxel_values,
                fps_values,
            )
        )

        significant = (
            p_value < ALPHA
        )

        result = {
            "species": species,
            "target": target,
            "n_blocks": len(valid_seeds),
            "statistic": statistic,
            "p_value": p_value,
            "alpha": ALPHA,
            "significant": significant,
        }

        results.append(result)

        print(
            f"{species:<10} "
            f"{target:>8} "
            f"{len(valid_seeds):>4} "
            f"{statistic:>14.6f} "
            f"{p_value:>14.6f} "
            f"{str(significant):>12}"
        )

    with OUTPUT_CSV.open(
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

        writer.writerows(
            results
        )

    print("=" * 90)

    print()
    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "FRIEDMAN COMPLETE"
    )


if __name__ == "__main__":
    main()

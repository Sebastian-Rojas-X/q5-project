"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Summarize frozen Chamfer measurements by species, sampler, and cardinality.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "chamfer_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "chamfer_summary.csv"
)


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No existe:\n{INPUT_PATH}"
        )

    groups = defaultdict(list)

    # --------------------------------------------------
    # LEER RESULTADOS
    # --------------------------------------------------

    with open(
        INPUT_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            key = (
                row["species"],
                row["method"],
                int(row["target"]),
            )

            groups[key].append(
                {
                    "chamfer_distance": float(
                        row["chamfer_distance"]
                    ),
                    "original_to_reduced": float(
                        row["original_to_reduced"]
                    ),
                }
            )

    # --------------------------------------------------
    # RESUMIR
    # --------------------------------------------------

    summary_rows = []

    for key in sorted(
        groups.keys(),
        key=lambda x: (
            x[0],
            x[1],
            -x[2],
        ),
    ):

        species, method, target = key

        values = groups[key]

        chamfer_values = np.array(
            [
                item["chamfer_distance"]
                for item in values
            ],
            dtype=float,
        )

        original_to_reduced_values = np.array(
            [
                item["original_to_reduced"]
                for item in values
            ],
            dtype=float,
        )

        summary_rows.append(
            {
                "species": species,
                "method": method,
                "target": target,
                "n_clouds": len(values),

                "chamfer_mean": (
                    f"{np.mean(chamfer_values):.6f}"
                ),

                "chamfer_std": (
                    f"{np.std(chamfer_values, ddof=1):.6f}"
                ),

                "original_to_reduced_mean": (
                    f"{np.mean(original_to_reduced_values):.6f}"
                ),

                "original_to_reduced_std": (
                    f"{np.std(
                        original_to_reduced_values,
                        ddof=1
                    ):.6f}"
                ),
            }
        )

    # --------------------------------------------------
    # GUARDAR CSV
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "species",
        "method",
        "target",
        "n_clouds",
        "chamfer_mean",
        "chamfer_std",
        "original_to_reduced_mean",
        "original_to_reduced_std",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------
    # MOSTRAR RESULTADOS
    # --------------------------------------------------

    print()
    print("RESUMEN CHAMFER")
    print("---------------")
    print()

    for row in summary_rows:

        print(
            f"{row['species']:6s} | "
            f"{row['method']:6s} | "
            f"{int(row['target']):5d} | "
            f"n={int(row['n_clouds']):2d} | "
            f"CD={row['chamfer_mean']} "
            f"± {row['chamfer_std']}"
        )

    print()
    print(
        "Archivo:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
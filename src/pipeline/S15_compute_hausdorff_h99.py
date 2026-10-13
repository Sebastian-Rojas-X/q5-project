"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Compute frozen robust Hausdorff H99 metrics and validate against stored Chamfer.
"""

import argparse
import csv
import gc
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_MODULE_DIR = PROJECT_ROOT / "src" / "data"

sys.path.append(str(DATA_MODULE_DIR))

from pheno4d_loader import load_pheno4d_cloud


# ============================================================
# INPUTS
# ============================================================

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

METADATA_PATH = (
    PROCESSED_ROOT
    / "metadata.csv"
)

FROZEN_CHAMFER_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "chamfer_results.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "hausdorff_h99_results.csv"
)

FINAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "hausdorff_h99_summary.csv"
)

# ============================================================
# FROZEN PROTOCOL
# ============================================================

HAUSDORFF_PERCENTILE = 99.0
PERCENTILE_METHOD = "linear"

EXPECTED_TOTAL_METADATA_ROWS = 1890
EXPECTED_ORIGINAL_CLOUDS = 126
EXPECTED_REDUCTIONS_PER_CLOUD = 15

EXPECTED_METHODS = {
    "random",
    "voxel",
    "fps",
}

EXPECTED_TARGETS = {
    1024,
    2048,
    4096,
    8192,
    16384,
}

CHAMFER_TOLERANCE = 1e-9
POINT_COUNT_TOLERANCE = 0


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calcula Hausdorff robusta H99 para Q5 "
            "usando las distancias nearest-neighbor "
            "original post-ground -> nube reducida."
        )
    )

    return parser.parse_args()


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No hay filas para escribir en {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_metadata(rows):
    if len(rows) != EXPECTED_TOTAL_METADATA_ROWS:
        raise RuntimeError(
            "metadata.csv no coincide con el benchmark "
            "congelado.\n"
            f"Esperado: {EXPECTED_TOTAL_METADATA_ROWS} filas\n"
            f"Encontrado: {len(rows)} filas"
        )

    grouped = defaultdict(list)

    seen_outputs = set()

    for row in rows:
        species = row["species"]
        method = row["method"]
        target = int(row["target"])
        input_path = row["input_path"]
        output_path = row["output_path"]

        if species not in {
            "maize",
            "tomato",
        }:
            raise RuntimeError(
                f"Especie inesperada: {species}"
            )

        if method not in EXPECTED_METHODS:
            raise RuntimeError(
                f"Método inesperado: {method}"
            )

        if target not in EXPECTED_TARGETS:
            raise RuntimeError(
                f"Target inesperado: {target}"
            )

        if output_path in seen_outputs:
            raise RuntimeError(
                "output_path duplicado en metadata.csv:\n"
                f"{output_path}"
            )

        seen_outputs.add(
            output_path
        )

        grouped[
            input_path
        ].append(row)

    if len(grouped) != EXPECTED_ORIGINAL_CLOUDS:
        raise RuntimeError(
            "Número inesperado de nubes originales.\n"
            f"Esperado: {EXPECTED_ORIGINAL_CLOUDS}\n"
            f"Encontrado: {len(grouped)}"
        )

    expected_pairs = {
        (
            method,
            target,
        )
        for method in EXPECTED_METHODS
        for target in EXPECTED_TARGETS
    }

    for input_path, group_rows in grouped.items():
        if (
            len(group_rows)
            != EXPECTED_REDUCTIONS_PER_CLOUD
        ):
            raise RuntimeError(
                "Una nube original no tiene exactamente "
                "15 reducciones:\n"
                f"{input_path}\n"
                f"Encontradas: {len(group_rows)}"
            )

        found_pairs = {
            (
                row["method"],
                int(row["target"]),
            )
            for row in group_rows
        }

        if found_pairs != expected_pairs:
            missing = (
                expected_pairs
                - found_pairs
            )

            extra = (
                found_pairs
                - expected_pairs
            )

            raise RuntimeError(
                "Diseño factorial incompleto para:\n"
                f"{input_path}\n"
                f"Faltantes: {sorted(missing)}\n"
                f"Extras: {sorted(extra)}"
            )

    return grouped


def build_frozen_chamfer_index(rows):
    if (
        len(rows)
        != EXPECTED_TOTAL_METADATA_ROWS
    ):
        raise RuntimeError(
            "chamfer_results.csv no contiene "
            "las 1890 filas congeladas.\n"
            f"Encontradas: {len(rows)}"
        )

    index = {}

    for row in rows:
        key = row["output_path"]

        if key in index:
            raise RuntimeError(
                "output_path duplicado en "
                "chamfer_results.csv:\n"
                f"{key}"
            )

        index[key] = row

    return index


# ============================================================
# GEOMETRY
# ============================================================

def nearest_neighbor_distances(
    original_points,
    reduced_points,
):
    """
    Calcula una única vez:

        d_i = min_{s in S} ||p_i - s||_2

    para todo p_i perteneciente a la nube original
    post-ground.

    Se usa distancia Euclidiana L2 NO elevada
    al cuadrado.
    """

    if (
        original_points.ndim != 2
        or original_points.shape[1] != 3
    ):
        raise ValueError(
            "original_points debe tener forma (N, 3)."
        )

    if (
        reduced_points.ndim != 2
        or reduced_points.shape[1] != 3
    ):
        raise ValueError(
            "reduced_points debe tener forma (M, 3)."
        )

    if len(original_points) == 0:
        raise ValueError(
            "La nube original está vacía."
        )

    if len(reduced_points) == 0:
        raise ValueError(
            "La nube reducida está vacía."
        )

    tree_reduced = cKDTree(
        reduced_points
    )

    distances, _ = tree_reduced.query(
        original_points,
        k=1,
        workers=-1,
    )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(distances)
    ):
        raise RuntimeError(
            "Se encontraron distancias "
            "nearest-neighbor no finitas."
        )

    return distances


def compute_geometry_metrics(
    distances,
):
    """
    A partir DEL MISMO vector de distancias:

      original_to_reduced = media(d_i)

      Chamfer =
          0.5 * original_to_reduced

      H99 =
          percentil 99 de d_i

    Reduced -> Original es cero por construcción
    en el benchmark Q5, porque las reducidas
    conservan puntos originales.
    """

    original_to_reduced = float(
        np.mean(
            distances
        )
    )

    reduced_to_original = 0.0

    chamfer = 0.5 * (
        original_to_reduced
        + reduced_to_original
    )

    h99 = float(
        np.percentile(
            distances,
            HAUSDORFF_PERCENTILE,
            method=PERCENTILE_METHOD,
        )
    )

    return {
        "original_to_reduced":
            original_to_reduced,

        "reduced_to_original":
            reduced_to_original,

        "chamfer_distance":
            chamfer,

        "hausdorff_h99":
            h99,
    }


# ============================================================
# FROZEN CHAMFER VALIDATION
# ============================================================

def validate_against_frozen_chamfer(
    calculated,
    frozen_row,
    output_path,
):
    frozen_original_to_reduced = float(
        frozen_row[
            "original_to_reduced"
        ]
    )

    frozen_reduced_to_original = float(
        frozen_row[
            "reduced_to_original"
        ]
    )

    frozen_chamfer = float(
        frozen_row[
            "chamfer_distance"
        ]
    )

    original_error = abs(
        calculated[
            "original_to_reduced"
        ]
        - frozen_original_to_reduced
    )

    reverse_error = abs(
        calculated[
            "reduced_to_original"
        ]
        - frozen_reduced_to_original
    )

    chamfer_error = abs(
        calculated[
            "chamfer_distance"
        ]
        - frozen_chamfer
    )

    max_error = max(
        original_error,
        reverse_error,
        chamfer_error,
    )

    if max_error > CHAMFER_TOLERANCE:
        raise RuntimeError(
            "\nVALIDACIÓN CHAMFER FALLIDA\n"
            "--------------------------\n"
            f"Archivo reducido: {output_path}\n"
            f"O->R recalculado: "
            f"{calculated['original_to_reduced']:.12f}\n"
            f"O->R congelado:   "
            f"{frozen_original_to_reduced:.12f}\n"
            f"Chamfer recalculado: "
            f"{calculated['chamfer_distance']:.12f}\n"
            f"Chamfer congelado:   "
            f"{frozen_chamfer:.12f}\n"
            f"Error máximo: {max_error:.12g}\n"
            f"Tolerancia: {CHAMFER_TOLERANCE:.12g}"
        )

    return {
        "original_to_reduced_error":
            original_error,

        "chamfer_error":
            chamfer_error,

        "max_error":
            max_error,
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["species"],
            row["method"],
            int(row["target"]),
        )

        groups[
            key
        ].append(row)

    summary_rows = []

    for key in sorted(
        groups.keys(),
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    ):
        species, method, target = key

        values = groups[key]

        h99_values = np.asarray(
            [
                float(
                    row[
                        "hausdorff_h99"
                    ]
                )
                for row in values
            ],
            dtype=np.float64,
        )

        chamfer_values = np.asarray(
            [
                float(
                    row[
                        "chamfer_distance_recomputed"
                    ]
                )
                for row in values
            ],
            dtype=np.float64,
        )

        summary_rows.append(
            {
                "species":
                    species,

                "method":
                    method,

                "target":
                    target,

                "n_clouds":
                    len(values),

                "hausdorff_percentile":
                    int(
                        HAUSDORFF_PERCENTILE
                    ),

                "percentile_method":
                    PERCENTILE_METHOD,

                "hausdorff_h99_mean":
                    f"{np.mean(h99_values):.12f}",

                "hausdorff_h99_std":
                    f"{np.std(h99_values, ddof=1):.12f}"
                    if len(h99_values) > 1
                    else "nan",

                "hausdorff_h99_median":
                    f"{np.median(h99_values):.12f}",

                "hausdorff_h99_min":
                    f"{np.min(h99_values):.12f}",

                "hausdorff_h99_max":
                    f"{np.max(h99_values):.12f}",

                "chamfer_recomputed_mean":
                    f"{np.mean(chamfer_values):.12f}",
            }
        )

    return summary_rows


# ============================================================
# ORDERING
# ============================================================

def cloud_sort_key(item):
    input_path, rows = item

    first = rows[0]

    return (
        first["species"],
        first["plant_id"],
        first["date"],
        input_path,
    )


def reduction_sort_key(row):
    method_order = {
        "random": 0,
        "voxel": 1,
        "fps": 2,
    }

    return (
        method_order[
            row["method"]
        ],
        int(
            row["target"]
        ),
    )


# ============================================================
# MAIN COMPUTATION
# ============================================================

def process(
    grouped_metadata,
    frozen_chamfer_index,
):
    cloud_groups = sorted(
        grouped_metadata.items(),
        key=cloud_sort_key,
    )

    expected_rows = (
        len(cloud_groups)
        * EXPECTED_REDUCTIONS_PER_CLOUD
    )

    print()
    print("=" * 80)

    print(
        "Q5 HAUSDORFF ROBUSTA H99"
    )

    print("=" * 80)

    print(
        "Percentil:",
        HAUSDORFF_PERCENTILE,
    )

    print(
        "Método percentil:",
        PERCENTILE_METHOD,
    )

    print(
        "Dirección:",
        "original post-ground -> reducida",
    )

    print(
        "Distancia:",
        "Euclidiana L2 no cuadrada",
    )

    print(
        "Nubes originales:",
        len(cloud_groups),
    )

    print(
        "Comparaciones esperadas:",
        expected_rows,
    )

    print(
        "Modo:",
        "COMPLETO",
    )

    print("=" * 80)
    print()

    results = []

    maximum_original_error = 0.0
    maximum_chamfer_error = 0.0

    total_clouds = len(
        cloud_groups
    )

    global_comparison_index = 0

    total_start = (
        time.perf_counter()
    )

    for cloud_index, (
        input_path_text,
        metadata_rows,
    ) in enumerate(
        cloud_groups,
        start=1,
    ):
        first_row = (
            metadata_rows[0]
        )

        species = (
            first_row[
                "species"
            ]
        )

        input_path = (
            PROJECT_ROOT
            / input_path_text
        )

        if not input_path.exists():
            raise FileNotFoundError(
                "No existe nube original:\n"
                f"{input_path}"
            )

        print(
            f"[Cloud "
            f"{cloud_index}/{total_clouds}] "
            f"{species} | "
            f"{first_row['plant_id']} | "
            f"{first_row['date']} | "
            f"{input_path.name}"
        )

        load_start = (
            time.perf_counter()
        )

        original_points, _ = (
            load_pheno4d_cloud(
                input_path,
                species=species,
            )
        )

        load_seconds = (
            time.perf_counter()
            - load_start
        )

        metadata_original_points = int(
            first_row[
                "original_points"
            ]
        )

        if (
            abs(
                len(original_points)
                - metadata_original_points
            )
            > POINT_COUNT_TOLERANCE
        ):
            raise RuntimeError(
                "Cantidad de puntos originales "
                "no coincide con metadata.csv:\n"
                f"{input_path}\n"
                f"Cargados: "
                f"{len(original_points)}\n"
                f"Metadata: "
                f"{metadata_original_points}"
            )

        print(
            f"  Original post-ground: "
            f"{len(original_points):,} points | "
            f"load={load_seconds:.3f}s"
        )

        ordered_reductions = sorted(
            metadata_rows,
            key=reduction_sort_key,
        )

        for row in ordered_reductions:
            global_comparison_index += 1

            reduced_path_text = (
                row[
                    "output_path"
                ]
            )

            reduced_path = (
                PROJECT_ROOT
                / reduced_path_text
            )

            if not reduced_path.exists():
                raise FileNotFoundError(
                    "No existe nube reducida:\n"
                    f"{reduced_path}"
                )

            if (
                reduced_path_text
                not in frozen_chamfer_index
            ):
                raise RuntimeError(
                    "No existe referencia Chamfer "
                    "congelada para:\n"
                    f"{reduced_path_text}"
                )

            expected_target = int(
                row[
                    "target"
                ]
            )

            with np.load(
                reduced_path
            ) as reduced_data:
                if (
                    "points"
                    not in reduced_data.files
                ):
                    raise RuntimeError(
                        "El NPZ no contiene la clave "
                        "'points':\n"
                        f"{reduced_path}"
                    )

                reduced_points = (
                    np.asarray(
                        reduced_data[
                            "points"
                        ]
                    )
                )

            if (
                len(reduced_points)
                != expected_target
            ):
                raise RuntimeError(
                    "Cantidad incorrecta de puntos "
                    "reducidos:\n"
                    f"{reduced_path}\n"
                    f"Esperados: {expected_target}\n"
                    f"Encontrados: "
                    f"{len(reduced_points)}"
                )

            metric_start = (
                time.perf_counter()
            )

            distances = (
                nearest_neighbor_distances(
                    original_points,
                    reduced_points,
                )
            )

            metrics = (
                compute_geometry_metrics(
                    distances
                )
            )

            metric_seconds = (
                time.perf_counter()
                - metric_start
            )

            validation = (
                validate_against_frozen_chamfer(
                    metrics,
                    frozen_chamfer_index[
                        reduced_path_text
                    ],
                    reduced_path_text,
                )
            )

            maximum_original_error = max(
                maximum_original_error,
                validation[
                    "original_to_reduced_error"
                ],
            )

            maximum_chamfer_error = max(
                maximum_chamfer_error,
                validation[
                    "chamfer_error"
                ],
            )

            results.append(
                {
                    "species":
                        row[
                            "species"
                        ],

                    "split":
                        row[
                            "split"
                        ],

                    "plant_id":
                        row[
                            "plant_id"
                        ],

                    "date":
                        row[
                            "date"
                        ],

                    "method":
                        row[
                            "method"
                        ],

                    "target":
                        int(
                            row[
                                "target"
                            ]
                        ),

                    "input_path":
                        input_path_text,

                    "output_path":
                        reduced_path_text,

                    "original_points":
                        len(
                            original_points
                        ),

                    "sampled_points":
                        len(
                            reduced_points
                        ),

                    "hausdorff_percentile":
                        int(
                            HAUSDORFF_PERCENTILE
                        ),

                    "percentile_method":
                        PERCENTILE_METHOD,

                    "hausdorff_h99":
                        f"{metrics['hausdorff_h99']:.12f}",

                    "original_to_reduced_recomputed":
                        f"{metrics['original_to_reduced']:.12f}",

                    "chamfer_distance_recomputed":
                        f"{metrics['chamfer_distance']:.12f}",

                    "frozen_original_to_reduced":
                        frozen_chamfer_index[
                            reduced_path_text
                        ][
                            "original_to_reduced"
                        ],

                    "frozen_chamfer_distance":
                        frozen_chamfer_index[
                            reduced_path_text
                        ][
                            "chamfer_distance"
                        ],

                    "original_to_reduced_abs_error":
                        f"{validation['original_to_reduced_error']:.12e}",

                    "chamfer_abs_error":
                        f"{validation['chamfer_error']:.12e}",

                    "h99_seconds":
                        f"{metric_seconds:.6f}",
                }
            )

            del distances
            del reduced_points

            print(
                "    "
                f"{row['method']:6s} | "
                f"{expected_target:5d} | "
                f"H99="
                f"{metrics['hausdorff_h99']:.6f} | "
                f"CD="
                f"{metrics['chamfer_distance']:.6f} | "
                f"check="
                f"{validation['chamfer_error']:.2e} | "
                f"{metric_seconds:.3f}s"
            )

        del original_points

        gc.collect()

        print()

    elapsed = (
        time.perf_counter()
        - total_start
    )

    if len(results) != expected_rows:
        raise RuntimeError(
            "Número inesperado de resultados.\n"
            f"Esperado: {expected_rows}\n"
            f"Encontrado: {len(results)}"
        )

    print("=" * 80)

    print(
        "CÁLCULO COMPLETADO"
    )

    print(
        "Comparaciones:",
        len(results),
    )

    print(
        "Máx. error O->R vs Chamfer congelado:",
        f"{maximum_original_error:.12e}",
    )

    print(
        "Máx. error CD vs Chamfer congelado:",
        f"{maximum_chamfer_error:.12e}",
    )

    print(
        "Tiempo total:",
        f"{elapsed:.2f}s",
    )

    print("=" * 80)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parse_arguments()

    metadata_rows = read_csv(
        METADATA_PATH
    )

    frozen_chamfer_rows = read_csv(
        FROZEN_CHAMFER_PATH
    )

    grouped_metadata = (
        validate_metadata(
            metadata_rows
        )
    )

    frozen_chamfer_index = (
        build_frozen_chamfer_index(
            frozen_chamfer_rows
        )
    )

    results = process(
        grouped_metadata,
        frozen_chamfer_index,
    )

    summary = build_summary(
        results
    )

    results_path = (
        FINAL_RESULTS_PATH
    )

    summary_path = (
        FINAL_SUMMARY_PATH
    )

    write_csv(
        results_path,
        results,
    )

    write_csv(
        summary_path,
        summary,
    )

    print()
    print(
        "Resultados:",
        results_path.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Resumen:",
        summary_path.relative_to(
            PROJECT_ROOT
        ),
    )

    print()

    print(
        "H99 COMPLETE"
    )


if __name__ == "__main__":
    main()

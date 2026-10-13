"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Compute Chamfer distance for every precomputed factorial cloud.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_MODULE_DIR = PROJECT_ROOT / "src" / "data"
METRICS_DIR = PROJECT_ROOT / "src" / "metrics"

sys.path.append(str(DATA_MODULE_DIR))
sys.path.append(str(METRICS_DIR))


# --------------------------------------------------
# IMPORTS DEL PROYECTO
# --------------------------------------------------

from pheno4d_loader import load_pheno4d_cloud
from chamfer_distance import chamfer_distance_subset


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

METADATA_PATH = PROCESSED_ROOT / "metadata.csv"


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Calcula Chamfer Distance para las "
            "nubes reducidas del experimento Q5."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Procesa solo las primeras N filas "
            "de metadata.csv."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/results/chamfer_results.csv",
        help="Archivo CSV de salida.",
    )

    return parser.parse_args()


# --------------------------------------------------
# LEER METADATA
# --------------------------------------------------

def read_metadata():
    """
    Lee metadata.csv generado durante
    la precomputación de downsampling.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe:\n{METADATA_PATH}"
        )

    rows = []

    with open(
        METADATA_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


# --------------------------------------------------
# GUARDAR RESULTADOS
# --------------------------------------------------

def save_results(rows, output_path):
    """
    Guarda resultados de Chamfer en CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "species",
        "split",
        "plant_id",
        "date",
        "method",
        "target",
        "input_path",
        "output_path",
        "original_points",
        "sampled_points",
        "original_to_reduced",
        "reduced_to_original",
        "chamfer_distance",
        "chamfer_seconds",
    ]

    with open(
        output_path,
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


# --------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------

def main():

    args = parse_arguments()

    metadata_rows = read_metadata()

    # --------------------------------------------------
    # LIMITE DE FILAS SOLICITADO
    # --------------------------------------------------

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit debe ser mayor que cero."
            )

        metadata_rows = metadata_rows[:args.limit]

    # --------------------------------------------------
    # RUTA DE SALIDA
    # --------------------------------------------------

    output_path = Path(
        args.output
    )

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT
            / output_path
        )

    print()
    print("CÁLCULO DE CHAMFER DISTANCE")
    print("---------------------------")
    print(
        "Nubes a procesar:",
        len(metadata_rows),
    )
    print(
        "Salida:",
        output_path,
    )

    if args.limit is not None:
        print(
            "LIMIT:",
            args.limit,
        )

    results = []

    # --------------------------------------------------
    # CACHE DE NUBES ORIGINALES
    #
    # Una misma nube original aparece 15 veces:
    #
    # 3 métodos x 5 targets
    #
    # Por eso evitamos volver a leer el TXT
    # innecesariamente.
    # --------------------------------------------------

    original_cache = {}

    total = len(
        metadata_rows
    )

    # --------------------------------------------------
    # RECORRER TODAS LAS NUBES PROCESADAS
    # --------------------------------------------------

    for index, row in enumerate(
        metadata_rows,
        start=1,
    ):

        species = row[
            "species"
        ]

        input_path = (
            PROJECT_ROOT
            / row["input_path"]
        )

        reduced_path = (
            PROJECT_ROOT
            / row["output_path"]
        )

        # --------------------------------------------------
        # CARGAR NUBE ORIGINAL
        # --------------------------------------------------

        cache_key = str(
            input_path
        )

        if cache_key not in original_cache:

            original_points, _ = (
                load_pheno4d_cloud(
                    input_path,
                    species=species,
                )
            )

            original_cache[
                cache_key
            ] = original_points

        else:

            original_points = (
                original_cache[
                    cache_key
                ]
            )

        # --------------------------------------------------
        # CARGAR NUBE REDUCIDA
        # --------------------------------------------------

        reduced_data = np.load(
            reduced_path
        )

        reduced_points = (
            reduced_data[
                "points"
            ]
        )

        # --------------------------------------------------
        # VALIDAR TARGET
        # --------------------------------------------------

        expected_target = int(
            row["target"]
        )

        if (
            len(reduced_points)
            != expected_target
        ):
            raise RuntimeError(
                f"{reduced_path.name}: "
                f"se esperaban "
                f"{expected_target} puntos "
                f"pero contiene "
                f"{len(reduced_points)}."
            )

        # --------------------------------------------------
        # CALCULAR CHAMFER
        #
        # Usamos la versión optimizada para nuestro
        # benchmark.
        #
        # Random, UVS y FPS seleccionan puntos reales
        # de la nube original.
        #
        # Por construcción:
        #
        # Reduced -> Original = 0
        #
        # Solo necesitamos calcular:
        #
        # Original -> Reduced
        # --------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        (
            cd,
            original_to_reduced,
            reduced_to_original,
        ) = chamfer_distance_subset(
            original_points,
            reduced_points,
        )

        chamfer_seconds = (
            time.perf_counter()
            - start_time
        )

        # --------------------------------------------------
        # VERIFICACIÓN
        # --------------------------------------------------

        if (
            reduced_to_original
            != 0.0
        ):
            raise RuntimeError(
                "Reduced -> Original "
                "debería ser exactamente 0 "
                "en este benchmark."
            )

        # --------------------------------------------------
        # GUARDAR RESULTADO
        # --------------------------------------------------

        results.append(
            {
                "species": species,
                "split": row[
                    "split"
                ],
                "plant_id": row[
                    "plant_id"
                ],
                "date": row[
                    "date"
                ],
                "method": row[
                    "method"
                ],
                "target": row[
                    "target"
                ],
                "input_path": row[
                    "input_path"
                ],
                "output_path": row[
                    "output_path"
                ],
                "original_points": len(
                    original_points
                ),
                "sampled_points": len(
                    reduced_points
                ),
                "original_to_reduced": (
                    f"{original_to_reduced:.12f}"
                ),
                "reduced_to_original": (
                    f"{reduced_to_original:.12f}"
                ),
                "chamfer_distance": (
                    f"{cd:.12f}"
                ),
                "chamfer_seconds": (
                    f"{chamfer_seconds:.6f}"
                ),
            }
        )

        # --------------------------------------------------
        # MOSTRAR PROGRESO
        # --------------------------------------------------

        print(
            f"[{index}/{total}] "
            f"{species} | "
            f"{row['split']} | "
            f"{row['method']} | "
            f"{row['target']} | "
            f"{input_path.name} | "
            f"O->R="
            f"{original_to_reduced:.6f} | "
            f"CD={cd:.6f}"
        )

    # --------------------------------------------------
    # GUARDAR CSV
    # --------------------------------------------------

    save_results(
        results,
        output_path,
    )

    # --------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("CHAMFER TERMINADO")
    print(
        "Resultados:",
        len(results),
    )
    print(
        "Archivo:",
        output_path,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
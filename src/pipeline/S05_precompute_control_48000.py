"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Precompute the practical high-resolution control with frozen sampling.
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
SAMPLING_DIR = PROJECT_ROOT / "src" / "sampling"

sys.path.append(str(DATA_MODULE_DIR))
sys.path.append(str(SAMPLING_DIR))


# --------------------------------------------------
# IMPORTS DEL PROYECTO
# --------------------------------------------------

from pheno4d_loader import load_pheno4d_cloud
from random_sampling import random_sampling


# --------------------------------------------------
# CONFIGURACIÓN DEL CONTROL
# --------------------------------------------------

SPECIES_LIST = [
    "maize",
    "tomato",
]

SPLIT_LIST = [
    "train",
    "val",
    "test",
]

# Control práctico de alta resolución.
#
# Decisión metodológica:
# - todas las nubes deben tener exactamente
#   el mismo número de puntos;
# - 48000 es menor que la nube etiquetada
#   más pequeña de Pheno4D post-soil
#   observada en nuestro subconjunto: 48352;
# - permite mantener batch_size=2 y la misma
#   arquitectura PointNet++.
CONTROL_TARGET = 48000

# Semilla fija del sampling.
#
# IMPORTANTE:
# Es independiente de las semillas de
# inicialización del entrenamiento.
SAMPLING_SEED = 42

CONTROL_METHOD_NAME = "control_48000"


# --------------------------------------------------
# LEER MANIFEST
# --------------------------------------------------

def read_manifest(species, split):
    """
    Lee uno de los manifests ya existentes.

    Ejemplo:
        data/manifests/maize_train.csv
    """

    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / f"{species}_{split}.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe el manifest:\n{manifest_path}"
        )

    rows = []

    with open(
        manifest_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


# --------------------------------------------------
# RESOLVER RUTA ORIGINAL
# --------------------------------------------------

def resolve_input_path(path_string):
    """
    Convierte la ruta almacenada en el manifest
    a una ruta utilizable dentro del proyecto.
    """

    path = Path(path_string)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# --------------------------------------------------
# GUARDAR NUBE
# --------------------------------------------------

def save_processed_cloud(
    output_path,
    points,
    labels,
):
    """
    Guarda puntos XYZ y etiquetas semánticas.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        points=points,
        labels=labels,
    )


# --------------------------------------------------
# PROCESAR UNA NUBE
# --------------------------------------------------

def process_cloud(
    row,
    species,
    split,
    output_root,
):
    """
    Carga una nube original post-soil y genera
    la referencia práctica de alta resolución
    de exactamente 48000 puntos mediante
    Random Sampling determinista.
    """

    input_path = resolve_input_path(
        row["path"]
    )

    # --------------------------------------------------
    # CARGAR NUBE
    #
    # El loader:
    # - elimina soil
    # - stem -> 0
    # - leaf -> 1
    # --------------------------------------------------

    points, labels = load_pheno4d_cloud(
        input_path,
        species=species,
    )

    original_points = len(points)

    if original_points < CONTROL_TARGET:
        raise RuntimeError(
            f"{input_path.name} contiene "
            f"{original_points} puntos post-soil, "
            f"menos que CONTROL_TARGET="
            f"{CONTROL_TARGET}."
        )

    # --------------------------------------------------
    # RANDOM SAMPLING
    #
    # Solo medimos la operación de sampling.
    # No incluye:
    # - lectura desde disco
    # - escritura NPZ
    #
    # Este tiempo NO se considerará todavía
    # como medición científica final.
    # --------------------------------------------------

    start_time = time.perf_counter()

    sampled_points, sampled_labels = (
        random_sampling(
            points,
            labels,
            n_points=CONTROL_TARGET,
            seed=SAMPLING_SEED,
        )
    )

    sampling_seconds = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------

    if len(sampled_points) != CONTROL_TARGET:
        raise RuntimeError(
            f"{input_path.name}: se obtuvieron "
            f"{len(sampled_points)} puntos, "
            f"pero se esperaban {CONTROL_TARGET}."
        )

    if len(sampled_points) != len(sampled_labels):
        raise RuntimeError(
            f"{input_path.name}: cantidad de puntos "
            "y etiquetas no coincide."
        )

    unique_labels = set(
        np.unique(sampled_labels).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise RuntimeError(
            f"{input_path.name}: etiquetas "
            f"inesperadas {unique_labels}."
        )

    # --------------------------------------------------
    # ESTADÍSTICAS
    # --------------------------------------------------

    stem_points = int(
        np.sum(sampled_labels == 0)
    )

    leaf_points = int(
        np.sum(sampled_labels == 1)
    )

    stem_ratio = (
        stem_points
        / len(sampled_labels)
    )

    # --------------------------------------------------
    # RUTA DE SALIDA
    #
    # Ejemplo:
    #
    # data/processed/
    #   maize/
    #     control_48000/
    #       train/
    #         M01_0313_a.npz
    # --------------------------------------------------

    output_path = (
        output_root
        / species
        / CONTROL_METHOD_NAME
        / split
        / f"{input_path.stem}.npz"
    )

    # --------------------------------------------------
    # GUARDAR
    # --------------------------------------------------

    save_processed_cloud(
        output_path,
        sampled_points,
        sampled_labels,
    )

    # --------------------------------------------------
    # METADATA DE ESTA NUBE
    # --------------------------------------------------

    metadata = {
        "species": species,
        "split": split,
        "plant_id": row["plant_id"],
        "date": row["date"],
        "condition": CONTROL_METHOD_NAME,
        "target": CONTROL_TARGET,
        "sampling_method": "random",
        "sampling_seed": SAMPLING_SEED,
        "input_path": str(
            input_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "output_path": str(
            output_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "original_points": original_points,
        "sampled_points": len(
            sampled_points
        ),
        "stem_points": stem_points,
        "leaf_points": leaf_points,
        "stem_ratio": f"{stem_ratio:.6f}",
        "sampling_seconds": (
            f"{sampling_seconds:.6f}"
        ),
    }

    return metadata


# --------------------------------------------------
# GUARDAR METADATA DEL CONTROL
# --------------------------------------------------

def save_metadata(
    metadata_rows,
    output_root,
):
    """
    Guarda metadata independiente del experimento
    factorial.

    IMPORTANTE:
    NO sobrescribe:
        data/processed/metadata.csv
    """

    if not metadata_rows:
        return

    metadata_path = (
        output_root
        / "control_48000_metadata.csv"
    )

    fieldnames = [
        "species",
        "split",
        "plant_id",
        "date",
        "condition",
        "target",
        "sampling_method",
        "sampling_seed",
        "input_path",
        "output_path",
        "original_points",
        "sampled_points",
        "stem_points",
        "leaf_points",
        "stem_ratio",
        "sampling_seconds",
    ]

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("Metadata del control guardada en:")
    print(metadata_path)


# --------------------------------------------------
# PROCESAR UNA CONFIGURACIÓN
# --------------------------------------------------

def process_configuration(
    species,
    split,
    output_root,
    metadata_rows,
    limit=None,
):
    """
    Procesa todas las nubes correspondientes
    a una especie y split.
    """

    manifest_rows = read_manifest(
        species,
        split,
    )

    if limit is not None:
        manifest_rows = (
            manifest_rows[:limit]
        )

    print()
    print("=" * 80)
    print(
        f"{species.upper()} | "
        f"{split.upper()} | "
        f"CONTROL 48000"
    )
    print("=" * 80)

    for cloud_number, row in enumerate(
        manifest_rows,
        start=1,
    ):

        input_path = resolve_input_path(
            row["path"]
        )

        print()
        print(
            f"[{cloud_number}/"
            f"{len(manifest_rows)}]"
        )

        print(
            f"Entrada: {input_path.name}"
        )

        metadata = process_cloud(
            row=row,
            species=species,
            split=split,
            output_root=output_root,
        )

        metadata_rows.append(
            metadata
        )

        print(
            "Original:",
            metadata[
                "original_points"
            ],
        )

        print(
            "Control: ",
            metadata[
                "sampled_points"
            ],
        )

        print(
            "Stem:    ",
            metadata[
                "stem_points"
            ],
        )

        print(
            "Leaf:    ",
            metadata[
                "leaf_points"
            ],
        )

        print(
            "Tiempo:  ",
            f'{metadata["sampling_seconds"]} s',
        )

        print(
            "Salida:  ",
            metadata[
                "output_path"
            ],
        )


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Precomputa el control práctico "
            "de alta resolución de 48000 puntos "
            "para Q5."
        )
    )

    parser.add_argument(
        "--species",
        choices=[
            "maize",
            "tomato",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Procesa solo las primeras N nubes "
            "de cada manifest."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed",
        help=(
            "Directorio raíz donde guardar "
            "las nubes del control."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------

def main():

    args = parse_arguments()

    species_to_process = (
        SPECIES_LIST
        if args.species == "all"
        else [args.species]
    )

    splits_to_process = (
        SPLIT_LIST
        if args.split == "all"
        else [args.split]
    )

    output_root = Path(
        args.output_root
    )

    if not output_root.is_absolute():
        output_root = (
            PROJECT_ROOT
            / output_root
        )

    metadata_rows = []

    print()
    print("PRECOMPUTACIÓN CONTROL Q5")
    print("-------------------------")

    print(
        "Species:",
        species_to_process,
    )

    print(
        "Splits:",
        splits_to_process,
    )

    print(
        "Condition:",
        CONTROL_METHOD_NAME,
    )

    print(
        "Target:",
        CONTROL_TARGET,
    )

    print(
        "Sampling method:",
        "random",
    )

    print(
        "Sampling seed:",
        SAMPLING_SEED,
    )

    print(
        "Output root:",
        output_root,
    )

    if args.limit is not None:

        print(
            "LIMIT:",
            args.limit,
            "nube(s) por manifest",
        )

    # --------------------------------------------------
    # RECORRER
    # --------------------------------------------------

    for species in species_to_process:

        for split in splits_to_process:

            process_configuration(
                species=species,
                split=split,
                output_root=output_root,
                metadata_rows=metadata_rows,
                limit=args.limit,
            )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    save_metadata(
        metadata_rows,
        output_root,
    )

    print()
    print("=" * 80)

    print(
        "PRECOMPUTACIÓN CONTROL TERMINADA"
    )

    print(
        "Archivos generados:",
        len(metadata_rows),
    )

    print("=" * 80)


if __name__ == "__main__":
    main()

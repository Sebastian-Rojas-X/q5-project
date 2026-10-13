"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Precompute the factorial Random, Voxel, and FPS point clouds.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


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
from voxel_grid import find_voxel_size_for_target
from fps import farthest_point_sampling


# --------------------------------------------------
# CONFIGURACIÓN EXPERIMENTAL
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

METHOD_LIST = [
    "random",
    "voxel",
    "fps",
]

TARGET_LIST = [
    16384,
    8192,
    4096,
    2048,
    1024,
]

# Semilla de sampling.
#
# IMPORTANTE:
# Es independiente de las cinco semillas que
# posteriormente usaremos para entrenamiento.
SAMPLING_SEED = 42


# --------------------------------------------------
# LEER MANIFEST
# --------------------------------------------------

def read_manifest(species, split):
    """
    Lee uno de los manifests creados previamente.

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
# RESOLVER RUTA DE NUBE ORIGINAL
# --------------------------------------------------

def resolve_input_path(path_string):
    """
    Convierte la ruta guardada en el manifest
    en una Path utilizable dentro del proyecto.
    """

    path = Path(path_string)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# --------------------------------------------------
# APLICAR MÉTODO DE DOWNSAMPLING
# --------------------------------------------------

def apply_sampling(
    points,
    labels,
    method,
    target,
):
    """
    Ejecuta uno de los tres métodos experimentales.

    Retorna:
        sampled_points
        sampled_labels
        voxel_size

    voxel_size será None para Random y FPS.
    """

    voxel_size = None

    # --------------------------------------------------
    # RANDOM
    # --------------------------------------------------

    if method == "random":

        sampled_points, sampled_labels = random_sampling(
            points,
            labels,
            n_points=target,
            seed=SAMPLING_SEED,
        )

    # --------------------------------------------------
    # VOXEL / UVS
    # --------------------------------------------------

    elif method == "voxel":

        (
            voxel_size,
            sampled_points,
            sampled_labels,
        ) = find_voxel_size_for_target(
            points,
            labels,
            target_points=target,
            device="cuda",
        )

    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    elif method == "fps":

        sampled_points, sampled_labels = (
            farthest_point_sampling(
                points,
                labels,
                n_points=target,
                device="cuda",
            )
        )

    else:

        raise ValueError(
            f"Método desconocido: {method}"
        )

    # --------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------

    if len(sampled_points) != target:
        raise RuntimeError(
            f"{method} produjo {len(sampled_points)} "
            f"puntos, pero se esperaban {target}."
        )

    if len(sampled_points) != len(sampled_labels):
        raise RuntimeError(
            "La cantidad de puntos y etiquetas "
            "no coincide."
        )

    return (
        sampled_points,
        sampled_labels,
        voxel_size,
    )


# --------------------------------------------------
# GUARDAR NUBE PROCESADA
# --------------------------------------------------

def save_processed_cloud(
    output_path,
    points,
    labels,
):
    """
    Guarda XYZ y labels en un archivo NPZ.
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
# GUARDAR METADATA
# --------------------------------------------------

def save_metadata(metadata_rows, output_root):
    """
    Guarda un CSV con la trazabilidad completa
    de todas las nubes producidas en esta ejecución.

    IMPORTANTE:
    Este archivo se sobrescribe en cada ejecución.

    La precomputación experimental final se ejecutará
    completa en una sola corrida.
    """

    if not metadata_rows:
        return

    metadata_path = (
        output_root
        / "metadata.csv"
    )

    fieldnames = [
        "species",
        "split",
        "plant_id",
        "date",
        "method",
        "target",
        "sampling_seed",
        "input_path",
        "output_path",
        "original_points",
        "sampled_points",
        "stem_points",
        "leaf_points",
        "stem_ratio",
        "sampling_seconds",
        "voxel_size",
    ]

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("Metadata guardada en:")
    print(metadata_path)


# --------------------------------------------------
# PROCESAR UNA CONFIGURACIÓN
# --------------------------------------------------

def process_configuration(
    species,
    split,
    method,
    target,
    output_root,
    metadata_rows,
    limit=None,
):
    """
    Procesa todas las nubes de una configuración:

        especie
        split
        método
        target
    """

    manifest_rows = read_manifest(
        species,
        split,
    )

    if limit is not None:
        manifest_rows = manifest_rows[:limit]

    print()
    print("=" * 80)
    print(
        f"{species.upper()} | "
        f"{split.upper()} | "
        f"{method.upper()} | "
        f"{target}"
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
            f"[{cloud_number}/{len(manifest_rows)}]"
        )
        print(
            f"Entrada: {input_path.name}"
        )

        # --------------------------------------------------
        # CARGAR NUBE
        #
        # El loader ya:
        #   - elimina soil
        #   - stem -> 0
        #   - leaf -> 1
        # --------------------------------------------------

        points, labels = load_pheno4d_cloud(
            input_path,
            species=species,
        )

        original_points = len(points)

        if original_points < target:
            raise RuntimeError(
                f"{input_path.name} contiene "
                f"{original_points} puntos, "
                f"menos que el target {target}."
            )

        # --------------------------------------------------
        # MEDIR SOLO EL SAMPLING
        #
        # NO incluimos:
        #   - lectura desde disco
        #   - escritura NPZ
        #
        # CUDA trabaja de forma asíncrona.
        # Sin synchronize(), el cronómetro podría terminar
        # antes de que la GPU haya finalizado realmente.
        # --------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        (
            sampled_points,
            sampled_labels,
            voxel_size,
        ) = apply_sampling(
            points,
            labels,
            method,
            target,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sampling_seconds = (
            time.perf_counter()
            - start_time
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
        #     random/
        #       16384/
        #         train/
        #           M01_0313_a.npz
        # --------------------------------------------------

        output_path = (
            output_root
            / species
            / method
            / str(target)
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
        # METADATA
        # --------------------------------------------------

        metadata_rows.append(
            {
                "species": species,
                "split": split,
                "plant_id": row["plant_id"],
                "date": row["date"],
                "method": method,
                "target": target,
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
                "stem_ratio": (
                    f"{stem_ratio:.6f}"
                ),
                "sampling_seconds": (
                    f"{sampling_seconds:.6f}"
                ),
                "voxel_size": (
                    ""
                    if voxel_size is None
                    else f"{voxel_size:.12f}"
                ),
            }
        )

        print(
            f"Original: {original_points}"
        )
        print(
            f"Final:    {len(sampled_points)}"
        )
        print(
            f"Stem:     {stem_points}"
        )
        print(
            f"Leaf:     {leaf_points}"
        )
        print(
            f"Tiempo:   "
            f"{sampling_seconds:.4f} s"
        )

        if voxel_size is not None:
            print(
                f"Voxel:    "
                f"{voxel_size:.6f}"
            )

        print(
            f"Salida:   {output_path}"
        )


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Precomputa las nubes reducidas "
            "del experimento Q5."
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
        "--method",
        choices=[
            "random",
            "voxel",
            "fps",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--target",
        choices=[
            "16384",
            "8192",
            "4096",
            "2048",
            "1024",
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
            "Directorio donde guardar las nubes."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------

def main():

    args = parse_arguments()

    # --------------------------------------------------
    # RESOLVER LISTAS
    # --------------------------------------------------

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

    methods_to_process = (
        METHOD_LIST
        if args.method == "all"
        else [args.method]
    )

    targets_to_process = (
        TARGET_LIST
        if args.target == "all"
        else [int(args.target)]
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
    print("PRECOMPUTACIÓN Q5")
    print("-----------------")
    print(
        "Species:",
        species_to_process,
    )
    print(
        "Splits:",
        splits_to_process,
    )
    print(
        "Methods:",
        methods_to_process,
    )
    print(
        "Targets:",
        targets_to_process,
    )
    print(
        "Sampling seed:",
        SAMPLING_SEED,
    )
    print(
        "Output:",
        output_root,
    )

    if args.limit is not None:
        print(
            "LIMIT:",
            args.limit,
            "nube(s) por manifest",
        )

    # --------------------------------------------------
    # RECORRER CONFIGURACIONES
    # --------------------------------------------------

    for species in species_to_process:

        for split in splits_to_process:

            for method in methods_to_process:

                for target in targets_to_process:

                    process_configuration(
                        species=species,
                        split=split,
                        method=method,
                        target=target,
                        output_root=output_root,
                        metadata_rows=metadata_rows,
                        limit=args.limit,
                    )

    # --------------------------------------------------
    # METADATA GLOBAL DE ESTA EJECUCIÓN
    # --------------------------------------------------

    save_metadata(
        metadata_rows,
        output_root,
    )

    print()
    print("=" * 80)
    print("PRECOMPUTACIÓN TERMINADA")
    print(
        "Archivos generados:",
        len(metadata_rows),
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

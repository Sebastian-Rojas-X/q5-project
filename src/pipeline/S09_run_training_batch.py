"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Run missing factorial training configurations with the frozen protocol.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------
# RUTAS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "training"
)


# --------------------------------------------------
# PROTOCOLO EXPERIMENTAL
# --------------------------------------------------

SPECIES_LIST = [
    "maize",
    "tomato",
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

SEED_LIST = [
    0,
    1,
    2,
    3,
    4,
]


# --------------------------------------------------
# HIPERPARÁMETROS CONGELADOS
# --------------------------------------------------

EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 0.0005

SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 1.0


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta todas las configuraciones "
            "faltantes de un método de downsampling "
            "para el experimento Q5."
        )
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=METHOD_LIST,
        help=(
            "Método experimental a ejecutar."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Muestra qué configuraciones faltan "
            "sin ejecutar entrenamiento."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# RUTA DEL RESULTADO FORMAL
# --------------------------------------------------

def get_summary_path(
    species,
    method,
    target,
    seed,
):

    return (
        RESULTS_ROOT
        / species
        / method
        / str(target)
        / f"seed_{seed}"
        / "no_aug"
        / "run_summary.json"
    )


# --------------------------------------------------
# GENERAR CONFIGURACIONES
# --------------------------------------------------

def build_configurations(method):

    configurations = []

    for species in SPECIES_LIST:

        for target in TARGET_LIST:

            for seed in SEED_LIST:

                configurations.append(
                    {
                        "species": species,
                        "method": method,
                        "target": target,
                        "seed": seed,
                    }
                )

    return configurations


# --------------------------------------------------
# COMANDO DE ENTRENAMIENTO
# --------------------------------------------------

def build_training_command(
    species,
    method,
    target,
    seed,
):

    return [
        sys.executable,
        "-m",
        "src.pipeline.S08_train_pointnet2",

        "--species",
        species,

        "--method",
        method,

        "--target",
        str(target),

        "--seed",
        str(seed),

        "--epochs",
        str(EPOCHS),

        "--batch-size",
        str(BATCH_SIZE),

        "--learning-rate",
        str(LEARNING_RATE),

        "--scheduler-step-size",
        str(SCHEDULER_STEP_SIZE),

        "--scheduler-gamma",
        str(SCHEDULER_GAMMA),
    ]


# --------------------------------------------------
# MOSTRAR CONFIGURACIÓN
# --------------------------------------------------

def print_configuration(
    number,
    total,
    config,
):

    print()
    print("=" * 80)

    print(
        f"CONFIGURACIÓN {number}/{total}"
    )

    print(
        "Species:",
        config["species"],
    )

    print(
        "Method: ",
        config["method"],
    )

    print(
        "Target: ",
        config["target"],
    )

    print(
        "Seed:   ",
        config["seed"],
    )

    print("=" * 80)


# --------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------

def main():

    args = parse_args()

    configurations = build_configurations(
        args.method
    )

    total = len(configurations)

    already_complete = 0
    executed = 0
    failed = 0
    pending = 0

    print()
    print("Q5 EXPERIMENT BATCH")
    print("-------------------")

    print(
        "Method:",
        args.method,
    )

    print(
        "Total configurations:",
        total,
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    print()
    print(
        "Frozen hyperparameters:"
    )

    print(
        "  epochs:",
        EPOCHS,
    )

    print(
        "  batch_size:",
        BATCH_SIZE,
    )

    print(
        "  learning_rate:",
        LEARNING_RATE,
    )

    print(
        "  scheduler_step_size:",
        SCHEDULER_STEP_SIZE,
    )

    print(
        "  scheduler_gamma:",
        SCHEDULER_GAMMA,
    )

    # --------------------------------------------------
    # RECORRER CONFIGURACIONES
    # --------------------------------------------------

    for number, config in enumerate(
        configurations,
        start=1,
    ):

        summary_path = get_summary_path(
            species=config["species"],
            method=config["method"],
            target=config["target"],
            seed=config["seed"],
        )

        # ----------------------------------------------
        # RESULTADO YA EXISTENTE
        # ----------------------------------------------

        if summary_path.exists():

            already_complete += 1

            print(
                "SKIP | "
                f'{config["species"]} | '
                f'{config["method"]} | '
                f'{config["target"]} | '
                f'seed {config["seed"]}'
            )

            continue

        # ----------------------------------------------
        # FALTA
        # ----------------------------------------------

        pending += 1

        print_configuration(
            number,
            total,
            config,
        )

        print(
            "PENDING:"
        )

        print(
            summary_path.relative_to(
                PROJECT_ROOT
            )
        )

        # ----------------------------------------------
        # DRY RUN
        # ----------------------------------------------

        if args.dry_run:

            continue

        # ----------------------------------------------
        # EJECUTAR TRAINING
        # ----------------------------------------------

        command = build_training_command(
            species=config["species"],
            method=config["method"],
            target=config["target"],
            seed=config["seed"],
        )

        print()
        print(
            "Starting training..."
        )

        print(
            "Command:"
        )

        print(
            " ".join(command)
        )

        print()

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
        )

        # ----------------------------------------------
        # VALIDAR EJECUCIÓN
        # ----------------------------------------------

        if result.returncode != 0:

            failed += 1

            print()
            print(
                "ERROR:"
            )

            print(
                f"Training failed with "
                f"return code "
                f"{result.returncode}."
            )

            print(
                "Batch execution stopped."
            )

            sys.exit(
                result.returncode
            )

        # ----------------------------------------------
        # VALIDAR SUMMARY
        # ----------------------------------------------

        if not summary_path.exists():

            failed += 1

            print()
            print(
                "ERROR:"
            )

            print(
                "Training returned successfully "
                "but run_summary.json was not found:"
            )

            print(
                summary_path
            )

            sys.exit(1)

        executed += 1

        print()
        print(
            "CONFIGURATION COMPLETE"
        )

    # --------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------

    remaining = 0

    for config in configurations:

        summary_path = get_summary_path(
            species=config["species"],
            method=config["method"],
            target=config["target"],
            seed=config["seed"],
        )

        if not summary_path.exists():
            remaining += 1

    print()
    print("=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)

    print(
        "Method:",
        args.method,
    )

    print(
        "Total configurations:",
        total,
    )

    print(
        "Already complete:",
        already_complete,
    )

    if args.dry_run:

        print(
            "Pending:",
            pending,
        )

        print(
            "Executed:",
            0,
        )

    else:

        print(
            "Executed this session:",
            executed,
        )

    print(
        "Failed:",
        failed,
    )

    print(
        "Remaining:",
        remaining,
    )

    if remaining == 0:

        print()
        print(
            f"{args.method.upper()} COMPLETE "
            f"{total}/{total}"
        )

    elif args.dry_run:

        print()
        print(
            "DRY RUN COMPLETE — "
            "NO TRAINING WAS EXECUTED"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

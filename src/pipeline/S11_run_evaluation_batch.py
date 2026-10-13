"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Evaluate the factorial and high-resolution control checkpoints.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------
# RUTAS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "training"
)


# --------------------------------------------------
# PROTOCOLO
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

CONTROL_METHOD = "control_48000"
CONTROL_TARGET = 48000


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evalúa automáticamente los mejores "
            "checkpoints del protocolo Q5."
        )
    )

    parser.add_argument(
        "--scope",
        choices=[
            "factorial",
            "control",
            "all",
        ],
        default="all",
        help=(
            "Qué parte del protocolo evaluar."
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="test",
        help=(
            "Split sobre el que ejecutar evaluación."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Muestra qué evaluaciones faltan "
            "sin ejecutar ninguna."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# CONFIGURACIONES FACTORIALES
# --------------------------------------------------

def build_factorial_configurations():

    configs = []

    for species in SPECIES_LIST:

        for method in METHOD_LIST:

            for target in TARGET_LIST:

                for seed in SEED_LIST:

                    configs.append(
                        {
                            "species": species,
                            "method": method,
                            "target": target,
                            "seed": seed,
                        }
                    )

    return configs


# --------------------------------------------------
# CONFIGURACIONES CONTROL
# --------------------------------------------------

def build_control_configurations():

    configs = []

    for species in SPECIES_LIST:

        for seed in SEED_LIST:

            configs.append(
                {
                    "species": species,
                    "method": CONTROL_METHOD,
                    "target": CONTROL_TARGET,
                    "seed": seed,
                }
            )

    return configs


# --------------------------------------------------
# CONSTRUIR LISTA SEGÚN SCOPE
# --------------------------------------------------

def build_configurations(scope):

    if scope == "factorial":
        return build_factorial_configurations()

    if scope == "control":
        return build_control_configurations()

    return (
        build_factorial_configurations()
        + build_control_configurations()
    )


# --------------------------------------------------
# DIRECTORIO DE UNA CORRIDA
# --------------------------------------------------

def get_run_dir(config):

    return (
        TRAINING_ROOT
        / config["species"]
        / config["method"]
        / str(config["target"])
        / f'seed_{config["seed"]}'
        / "no_aug"
    )


# --------------------------------------------------
# CHECKPOINT
# --------------------------------------------------

def get_checkpoint_path(config):

    return (
        get_run_dir(config)
        / "best_model.pt"
    )


# --------------------------------------------------
# RESULTADO DE EVALUACIÓN
# --------------------------------------------------

def get_evaluation_path(
    config,
    split,
):

    return (
        get_run_dir(config)
        / f"evaluation_{split}.json"
    )


# --------------------------------------------------
# COMANDO DE EVALUACIÓN
# --------------------------------------------------

def build_evaluation_command(
    checkpoint_path,
    split,
):

    return [
        sys.executable,
        "-m",
        "src.pipeline.S10_evaluate_pointnet2",
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        split,
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
        f"EVALUACIÓN {number}/{total}"
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
# MAIN
# --------------------------------------------------

def main():

    args = parse_args()

    configs = build_configurations(
        args.scope
    )

    total = len(configs)

    already_complete = 0
    pending = 0
    executed = 0
    failed = 0
    missing_checkpoint = 0

    print()
    print("Q5 EVALUATION BATCH")
    print("-------------------")

    print(
        "Scope:",
        args.scope,
    )

    print(
        "Split:",
        args.split,
    )

    print(
        "Total configurations:",
        total,
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    # --------------------------------------------------
    # RECORRER
    # --------------------------------------------------

    for number, config in enumerate(
        configs,
        start=1,
    ):

        checkpoint_path = (
            get_checkpoint_path(
                config
            )
        )

        evaluation_path = (
            get_evaluation_path(
                config,
                args.split,
            )
        )

        # ----------------------------------------------
        # YA EVALUADO
        # ----------------------------------------------

        if evaluation_path.exists():

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
        # CHECKPOINT FALTANTE
        # ----------------------------------------------

        if not checkpoint_path.exists():

            missing_checkpoint += 1

            print()
            print(
                "MISSING CHECKPOINT | "
                f'{config["species"]} | '
                f'{config["method"]} | '
                f'{config["target"]} | '
                f'seed {config["seed"]}'
            )

            print(
                checkpoint_path.relative_to(
                    PROJECT_ROOT
                )
            )

            continue

        # ----------------------------------------------
        # PENDIENTE
        # ----------------------------------------------

        pending += 1

        print_configuration(
            number,
            total,
            config,
        )

        print(
            "Checkpoint:"
        )

        print(
            checkpoint_path.relative_to(
                PROJECT_ROOT
            )
        )

        print(
            "Output:"
        )

        print(
            evaluation_path.relative_to(
                PROJECT_ROOT
            )
        )

        # ----------------------------------------------
        # DRY RUN
        # ----------------------------------------------

        if args.dry_run:
            continue

        # ----------------------------------------------
        # EJECUTAR
        # ----------------------------------------------

        command = (
            build_evaluation_command(
                checkpoint_path,
                args.split,
            )
        )

        print()
        print(
            "Starting evaluation..."
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
        # ERROR DEL EVALUADOR
        # ----------------------------------------------

        if result.returncode != 0:

            failed += 1

            print()
            print(
                "ERROR:"
            )

            print(
                "Evaluation failed with "
                f"return code {result.returncode}."
            )

            print(
                "Batch execution stopped."
            )

            sys.exit(
                result.returncode
            )

        # ----------------------------------------------
        # VALIDAR OUTPUT
        # ----------------------------------------------

        if not evaluation_path.exists():

            failed += 1

            print()
            print(
                "ERROR:"
            )

            print(
                "Evaluator returned successfully "
                "but evaluation file was not found:"
            )

            print(
                evaluation_path
            )

            sys.exit(1)

        executed += 1

        print()
        print(
            "EVALUATION COMPLETE"
        )

    # --------------------------------------------------
    # RECONTAR FALTANTES
    # --------------------------------------------------

    remaining = 0

    for config in configs:

        evaluation_path = (
            get_evaluation_path(
                config,
                args.split,
            )
        )

        if not evaluation_path.exists():
            remaining += 1

    # --------------------------------------------------
    # RESUMEN
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("EVALUATION BATCH SUMMARY")
    print("=" * 80)

    print(
        "Scope:",
        args.scope,
    )

    print(
        "Split:",
        args.split,
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
        "Missing checkpoints:",
        missing_checkpoint,
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
            f"EVALUATION COMPLETE "
            f"{total}/{total}"
        )

    elif args.dry_run:

        print()
        print(
            "DRY RUN COMPLETE — "
            "NO EVALUATION WAS EXECUTED"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

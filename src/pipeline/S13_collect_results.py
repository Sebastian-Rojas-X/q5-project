"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Consolidate the 160 formal training and test evaluation records.
"""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "training"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
)

OUTPUT_CSV = OUTPUT_DIR / "results_master.csv"


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


def build_configurations():

    configs = []

    # Diseño factorial principal: 150 corridas
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
                            "is_control": False,
                        }
                    )

    # Control práctico: 10 corridas
    for species in SPECIES_LIST:
        for seed in SEED_LIST:
            configs.append(
                {
                    "species": species,
                    "method": CONTROL_METHOD,
                    "target": CONTROL_TARGET,
                    "seed": seed,
                    "is_control": True,
                }
            )

    return configs


def get_run_dir(config):

    return (
        TRAINING_ROOT
        / config["species"]
        / config["method"]
        / str(config["target"])
        / f'seed_{config["seed"]}'
        / "no_aug"
    )


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_identity(
    config,
    run_data,
    eval_data,
):

    expected_species = config["species"]
    expected_method = config["method"]
    expected_target = config["target"]
    expected_seed = config["seed"]

    checks = [
        (
            "run species",
            run_data["species"],
            expected_species,
        ),
        (
            "run method",
            run_data["method"],
            expected_method,
        ),
        (
            "run target",
            run_data["target"],
            expected_target,
        ),
        (
            "run seed",
            run_data["train_seed"],
            expected_seed,
        ),
        (
            "eval species",
            eval_data["species"],
            expected_species,
        ),
        (
            "eval method",
            eval_data["method"],
            expected_method,
        ),
        (
            "eval target",
            eval_data["target"],
            expected_target,
        ),
        (
            "eval seed",
            eval_data["train_seed"],
            expected_seed,
        ),
    ]

    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"{name}: expected {expected}, "
                f"got {actual}"
            )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configs = build_configurations()

    rows = []

    missing = []

    for config in configs:

        run_dir = get_run_dir(config)

        run_summary_path = (
            run_dir
            / "run_summary.json"
        )

        evaluation_path = (
            run_dir
            / "evaluation_test.json"
        )

        if not run_summary_path.exists():
            missing.append(
                str(
                    run_summary_path.relative_to(
                        PROJECT_ROOT
                    )
                )
            )
            continue

        if not evaluation_path.exists():
            missing.append(
                str(
                    evaluation_path.relative_to(
                        PROJECT_ROOT
                    )
                )
            )
            continue

        run_data = load_json(
            run_summary_path
        )

        eval_data = load_json(
            evaluation_path
        )

        validate_identity(
            config,
            run_data,
            eval_data,
        )

        row = {
            # ------------------------------------------
            # Identificación
            # ------------------------------------------
            "species": config["species"],
            "method": config["method"],
            "target": config["target"],
            "seed": config["seed"],
            "is_control": config["is_control"],

            # ------------------------------------------
            # Configuración
            # ------------------------------------------
            "eval_seed": eval_data["eval_seed"],
            "epochs": run_data["epochs"],
            "batch_size_train": run_data["batch_size"],
            "batch_size_test": eval_data["batch_size"],
            "learning_rate": run_data[
                "initial_learning_rate"
            ],
            "augmentation_name": run_data[
                "augmentation_name"
            ],

            # ------------------------------------------
            # Tamaño de datasets
            # ------------------------------------------
            "train_samples": run_data["train_samples"],
            "val_samples": run_data["val_samples"],
            "test_samples": eval_data["samples"],

            # ------------------------------------------
            # Mejor resultado en VALIDACIÓN
            # ------------------------------------------
            "best_epoch": run_data["best_epoch"],
            "val_loss": run_data["best_val_loss"],
            "val_iou_stem": run_data["best_iou_stem"],
            "val_iou_leaf": run_data["best_iou_leaf"],
            "val_miou": run_data["best_miou"],

            # ------------------------------------------
            # Resultado formal en TEST
            # ------------------------------------------
            "test_loss": eval_data["loss"],
            "test_iou_stem": eval_data["iou_stem"],
            "test_iou_leaf": eval_data["iou_leaf"],
            "test_miou": eval_data["miou"],

            # ------------------------------------------
            # Costos computacionales
            # ------------------------------------------
            "training_seconds": run_data[
                "training_seconds"
            ],
            "training_peak_gpu_memory_mb": run_data[
                "peak_gpu_memory_mb"
            ],
            "evaluation_seconds": eval_data[
                "evaluation_seconds"
            ],
            "evaluation_peak_gpu_memory_mb": eval_data[
                "peak_gpu_memory_mb"
            ],

            # ------------------------------------------
            # Matrices de confusión
            # ------------------------------------------
            "val_cm_00": run_data[
                "best_confusion_matrix"
            ][0][0],
            "val_cm_01": run_data[
                "best_confusion_matrix"
            ][0][1],
            "val_cm_10": run_data[
                "best_confusion_matrix"
            ][1][0],
            "val_cm_11": run_data[
                "best_confusion_matrix"
            ][1][1],

            "test_cm_00": eval_data[
                "confusion_matrix"
            ][0][0],
            "test_cm_01": eval_data[
                "confusion_matrix"
            ][0][1],
            "test_cm_10": eval_data[
                "confusion_matrix"
            ][1][0],
            "test_cm_11": eval_data[
                "confusion_matrix"
            ][1][1],

            # ------------------------------------------
            # Trazabilidad
            # ------------------------------------------
            "run_summary_path": str(
                run_summary_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "evaluation_path": str(
                evaluation_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        }

        rows.append(row)

    # --------------------------------------------------
    # VALIDACIONES GLOBALES
    # --------------------------------------------------

    if missing:

        print()
        print("ERROR: missing files")

        for path in missing:
            print(path)

        raise SystemExit(1)

    if len(rows) != 160:
        raise RuntimeError(
            f"Expected 160 rows, got {len(rows)}"
        )

    # --------------------------------------------------
    # GUARDAR CSV
    # --------------------------------------------------

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    # --------------------------------------------------
    # RESUMEN
    # --------------------------------------------------

    factorial_count = sum(
        not row["is_control"]
        for row in rows
    )

    control_count = sum(
        row["is_control"]
        for row in rows
    )

    print()
    print("=" * 80)
    print("Q5 RESULTS COLLECTION")
    print("=" * 80)

    print(
        "Factorial rows:",
        factorial_count,
        "/ 150",
    )

    print(
        "Control rows:",
        control_count,
        "/ 10",
    )

    print(
        "Total rows:",
        len(rows),
        "/ 160",
    )

    print(
        "Output:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print("RESULTS MASTER COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

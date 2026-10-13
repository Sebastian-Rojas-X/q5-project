"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Train one frozen PointNet++ configuration and record its checkpoint and metrics.
"""

from pathlib import Path
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.pheno4d_processed_dataset import (
    Pheno4DProcessedDataset,
)

from src.models.pointnet2.pointnet2_sem_seg import (
    PointNet2SemanticSegmentation,
    PointNet2Loss,
)

from src.metrics.segmentation_metrics import (
    update_confusion_matrix,
    compute_segmentation_metrics,
)


# --------------------------------------------------
# CONSTANTES
# --------------------------------------------------

PROCESSED_ROOT = "data/processed"

DEFAULT_EVAL_SEED = 12345

NUM_CLASSES = 2

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 5e-4

DEFAULT_SCHEDULER_STEP_SIZE = 10
DEFAULT_SCHEDULER_GAMMA = 1.0


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento formal de PointNet++ "
            "para segmentación semántica Pheno4D."
        )
    )

    parser.add_argument(
        "--species",
        type=str,
        required=True,
        choices=[
            "maize",
            "tomato",
        ],
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "random",
            "voxel",
            "fps",
            "control_48000",
        ],
    )

    parser.add_argument(
        "--target",
        type=int,
        required=True,
        choices=[
            48000,
            16384,
            8192,
            4096,
            2048,
            1024,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
    )

    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=DEFAULT_SCHEDULER_STEP_SIZE,
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=DEFAULT_SCHEDULER_GAMMA,
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Activa augmentation geométrica "
            "solamente en entrenamiento."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# SEEDS
# --------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_batches = 0

    for (
        points,
        labels,
        metadata,
    ) in loader:

        points = points.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        predictions, _ = model(
            points
        )

        loss = criterion(
            predictions.reshape(
                -1,
                NUM_CLASSES,
            ),
            labels.reshape(
                -1,
            ),
        )

        loss.backward()

        optimizer.step()

        total_loss += float(
            loss.item()
        )

        total_batches += 1

    return (
        total_loss
        / total_batches
    )


# --------------------------------------------------
# VALIDACIÓN
# --------------------------------------------------

def validate(
    model,
    loader,
    criterion,
    device,
    eval_seed,
):

    # Guardar RNG del entrenamiento
    python_rng_state = (
        random.getstate()
    )

    numpy_rng_state = (
        np.random.get_state()
    )

    torch_cpu_rng_state = (
        torch.get_rng_state()
    )

    if torch.cuda.is_available():

        torch_cuda_rng_states = (
            torch.cuda.get_rng_state_all()
        )

    else:

        torch_cuda_rng_states = None

    try:

        # Seed exclusiva de evaluación
        random.seed(
            eval_seed
        )

        np.random.seed(
            eval_seed
        )

        torch.manual_seed(
            eval_seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed(
                eval_seed
            )

            torch.cuda.manual_seed_all(
                eval_seed
            )

        model.eval()

        total_loss = 0.0
        total_batches = 0

        confusion_matrix = torch.zeros(
            (
                NUM_CLASSES,
                NUM_CLASSES,
            ),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():

            for (
                points,
                labels,
                metadata,
            ) in loader:

                points = points.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                predictions, _ = model(
                    points
                )

                loss = criterion(
                    predictions.reshape(
                        -1,
                        NUM_CLASSES,
                    ),
                    labels.reshape(
                        -1,
                    ),
                )

                total_loss += float(
                    loss.item()
                )

                total_batches += 1

                predicted_labels = torch.argmax(
                    predictions,
                    dim=2,
                )

                update_confusion_matrix(
                    confusion_matrix,
                    predicted_labels,
                    labels,
                    num_classes=NUM_CLASSES,
                )

        mean_loss = (
            total_loss
            / total_batches
        )

        metrics = compute_segmentation_metrics(
            confusion_matrix
        )

    finally:

        # Restaurar RNG del entrenamiento
        random.setstate(
            python_rng_state
        )

        np.random.set_state(
            numpy_rng_state
        )

        torch.set_rng_state(
            torch_cpu_rng_state
        )

        if (
            torch.cuda.is_available()
            and torch_cuda_rng_states is not None
        ):

            torch.cuda.set_rng_state_all(
                torch_cuda_rng_states
            )

    return (
        mean_loss,
        metrics,
        confusion_matrix,
    )


# --------------------------------------------------
# HISTORIAL
# --------------------------------------------------

def save_history(
    history,
    output_path,
):

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "iou_stem",
        "iou_leaf",
        "miou",
        "epoch_seconds",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_args()

    # --------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------

    if args.epochs <= 0:
        raise ValueError(
            "--epochs debe ser > 0"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size debe ser > 0"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate debe ser > 0"
        )

    if args.scheduler_step_size <= 0:
        raise ValueError(
            "--scheduler-step-size debe ser > 0"
        )

    if not (
        0.0
        < args.scheduler_gamma
        <= 1.0
    ):
        raise ValueError(
            "--scheduler-gamma debe estar en (0, 1]"
        )

    # --------------------------------------------------
    # TRAIN SEED
    # --------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("POINTNET++ PHENO4D TRAINING")
    print("---------------------------")

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # CONFIGURACIÓN
    # --------------------------------------------------

    print()

    print(
        "Species:",
        args.species
    )

    print(
        "Method:",
        args.method
    )

    print(
        "Target:",
        args.target
    )

    print(
        "Train seed:",
        args.seed
    )

    print(
        "Eval seed:",
        args.eval_seed
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Initial learning rate:",
        args.learning_rate
    )

    print(
        "Train augmentation:",
        args.augment
    )

    print(
        "Validation augmentation:",
        False
    )

    print(
        "Scheduler:",
        "StepLR"
    )

    print(
        "Scheduler step size:",
        args.scheduler_step_size
    )

    print(
        "Scheduler gamma:",
        args.scheduler_gamma
    )

    # --------------------------------------------------
    # NOMBRE DEL PROTOCOLO
    # --------------------------------------------------

    augmentation_name = (
        "aug"
        if args.augment
        else "no_aug"
    )

    # --------------------------------------------------
    # OUTPUT DIRECTORY
    #
    # Ejemplo:
    #
    # data/results/training/
    #   maize/
    #     random/
    #       16384/
    #         seed_0/
    #           aug/
    #
    # o:
    #
    #           no_aug/
    # --------------------------------------------------

    output_dir = (
        Path(
            "data/results/training"
        )
        / args.species
        / args.method
        / str(
            args.target
        )
        / f"seed_{args.seed}"
        / augmentation_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    history_path = (
        output_dir
        / "history.csv"
    )

    summary_path = (
        output_dir
        / "run_summary.json"
    )

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------

    train_dataset = (
        Pheno4DProcessedDataset(
            processed_root=PROCESSED_ROOT,
            species=args.species,
            method=args.method,
            target=args.target,
            split="train",
            normalize=True,
            augment=args.augment,
        )
    )

    val_dataset = (
        Pheno4DProcessedDataset(
            processed_root=PROCESSED_ROOT,
            species=args.species,
            method=args.method,
            target=args.target,
            split="val",
            normalize=True,
            augment=False,
        )
    )

    print()

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Val samples:",
        len(val_dataset)
    )

    print(
        "Output protocol:",
        augmentation_name
    )

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------
    # MODELO
    # --------------------------------------------------

    model = (
        PointNet2SemanticSegmentation(
            num_classes=NUM_CLASSES
        ).to(device)
    )

    criterion = (
        PointNet2Loss()
    )

    # --------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
    )

    # --------------------------------------------------
    # SCHEDULER
    #
    # gamma = 1.0 equivale a LR fijo.
    # --------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    )

    # --------------------------------------------------
    # ESTADO
    # --------------------------------------------------

    history = []

    best_miou = float(
        "-inf"
    )

    best_epoch = None
    best_val_loss = None
    best_metrics = None
    best_confusion_matrix = None
    best_learning_rate = None

    # --------------------------------------------------
    # MEMORIA GPU
    # --------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # TIMING
    # --------------------------------------------------

    training_start = (
        time.perf_counter()
    )

    # --------------------------------------------------
    # LOOP
    # --------------------------------------------------

    print()
    print("Training...")
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        current_learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ------------------------------------------
        # TRAIN
        # ------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # ------------------------------------------
        # VALIDACIÓN
        # ------------------------------------------

        (
            val_loss,
            val_metrics,
            confusion_matrix,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            eval_seed=args.eval_seed,
        )

        # ------------------------------------------
        # TIEMPO
        # ------------------------------------------

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        # ------------------------------------------
        # HISTORIAL
        # ------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "learning_rate": (
                    current_learning_rate
                ),
                "train_loss": (
                    train_loss
                ),
                "val_loss": (
                    val_loss
                ),
                "iou_stem": (
                    val_metrics[
                        "iou_stem"
                    ]
                ),
                "iou_leaf": (
                    val_metrics[
                        "iou_leaf"
                    ]
                ),
                "miou": (
                    val_metrics[
                        "miou"
                    ]
                ),
                "epoch_seconds": (
                    epoch_seconds
                ),
            }
        )

        # ------------------------------------------
        # LOG
        # ------------------------------------------

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_learning_rate:.8f} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"IoU_stem={val_metrics['iou_stem']:.4f} | "
            f"IoU_leaf={val_metrics['iou_leaf']:.4f} | "
            f"mIoU={val_metrics['miou']:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

        # ------------------------------------------
        # CHECKPOINT POR mIoU
        # ------------------------------------------

        current_miou = (
            val_metrics[
                "miou"
            ]
        )

        if current_miou > best_miou:

            best_miou = (
                current_miou
            )

            best_epoch = (
                epoch
            )

            best_val_loss = (
                val_loss
            )

            best_metrics = (
                val_metrics.copy()
            )

            best_learning_rate = (
                current_learning_rate
            )

            best_confusion_matrix = (
                confusion_matrix
                .detach()
                .cpu()
                .tolist()
            )

            torch.save(
                {
                    "epoch": (
                        epoch
                    ),

                    "species": (
                        args.species
                    ),

                    "method": (
                        args.method
                    ),

                    "target": (
                        args.target
                    ),

                    "train_seed": (
                        args.seed
                    ),

                    "eval_seed": (
                        args.eval_seed
                    ),

                    "epochs": (
                        args.epochs
                    ),

                    "batch_size": (
                        args.batch_size
                    ),

                    "learning_rate": (
                        args.learning_rate
                    ),

                    "current_learning_rate": (
                        current_learning_rate
                    ),

                    "augmentation_name": (
                        augmentation_name
                    ),

                    "train_augmentation": (
                        args.augment
                    ),

                    "validation_augmentation": (
                        False
                    ),

                    "scheduler_type": (
                        "StepLR"
                    ),

                    "scheduler_step_size": (
                        args.scheduler_step_size
                    ),

                    "scheduler_gamma": (
                        args.scheduler_gamma
                    ),

                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),

                    "scheduler_state_dict": (
                        scheduler.state_dict()
                    ),

                    "train_loss": (
                        train_loss
                    ),

                    "val_loss": (
                        val_loss
                    ),

                    "iou_stem": (
                        val_metrics[
                            "iou_stem"
                        ]
                    ),

                    "iou_leaf": (
                        val_metrics[
                            "iou_leaf"
                        ]
                    ),

                    "miou": (
                        val_metrics[
                            "miou"
                        ]
                    ),

                    "confusion_matrix": (
                        best_confusion_matrix
                    ),
                },
                checkpoint_path,
            )

            print(
                "  -> best checkpoint saved by mIoU"
            )

        # ------------------------------------------
        # SCHEDULER
        # ------------------------------------------

        scheduler.step()

    # --------------------------------------------------
    # FINALIZAR
    # --------------------------------------------------

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    save_history(
        history=history,
        output_path=history_path,
    )

    if torch.cuda.is_available():

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    else:

        peak_gpu_memory_mb = None

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    summary = {

        "species": (
            args.species
        ),

        "method": (
            args.method
        ),

        "target": (
            args.target
        ),

        "train_seed": (
            args.seed
        ),

        "eval_seed": (
            args.eval_seed
        ),

        "epochs": (
            args.epochs
        ),

        "batch_size": (
            args.batch_size
        ),

        "initial_learning_rate": (
            args.learning_rate
        ),

        "augmentation_name": (
            augmentation_name
        ),

        "train_augmentation": (
            args.augment
        ),

        "validation_augmentation": (
            False
        ),

        "scheduler": {
            "type": (
                "StepLR"
            ),
            "step_size": (
                args.scheduler_step_size
            ),
            "gamma": (
                args.scheduler_gamma
            ),
        },

        "train_samples": (
            len(train_dataset)
        ),

        "val_samples": (
            len(val_dataset)
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_learning_rate": (
            best_learning_rate
        ),

        "best_val_loss": (
            best_val_loss
        ),

        "best_iou_stem": (
            best_metrics[
                "iou_stem"
            ]
        ),

        "best_iou_leaf": (
            best_metrics[
                "iou_leaf"
            ]
        ),

        "best_miou": (
            best_metrics[
                "miou"
            ]
        ),

        "best_confusion_matrix": (
            best_confusion_matrix
        ),

        "training_seconds": (
            training_seconds
        ),

        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),

        "checkpoint_path": str(
            checkpoint_path
        ),

        "history_path": str(
            history_path
        ),
    }

    with open(
        summary_path,
        "w",
    ) as json_file:

        json.dump(
            summary,
            json_file,
            indent=2,
        )

    # --------------------------------------------------
    # RESUMEN
    # --------------------------------------------------

    print()
    print("TRAINING SUMMARY")
    print("----------------")

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best learning rate:",
        f"{best_learning_rate:.8f}"
    )

    print(
        "Best val loss:",
        f"{best_val_loss:.6f}"
    )

    print(
        "Best IoU stem:",
        f"{best_metrics['iou_stem']:.6f}"
    )

    print(
        "Best IoU leaf:",
        f"{best_metrics['iou_leaf']:.6f}"
    )

    print(
        "Best mIoU:",
        f"{best_metrics['miou']:.6f}"
    )

    print(
        "Training time:",
        f"{training_seconds:.2f} s"
    )

    if peak_gpu_memory_mb is not None:

        print(
            "Peak GPU memory:",
            f"{peak_gpu_memory_mb:.2f} MB"
        )

    print()

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "History:",
        history_path
    )

    print(
        "Summary:",
        summary_path
    )

    assert checkpoint_path.exists()
    assert history_path.exists()
    assert summary_path.exists()

    print()
    print("TRAINING RUN OK")
    print()


if __name__ == "__main__":
    main()
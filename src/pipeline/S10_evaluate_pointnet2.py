"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Evaluate a frozen PointNet++ checkpoint on the requested data split.
"""

from pathlib import Path
import argparse
import json
import time

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
NUM_CLASSES = 2


# --------------------------------------------------
# ARGUMENTOS
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluación formal de checkpoints "
            "PointNet++ sobre Pheno4D."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al best_model.pt",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=[
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Si no se especifica, usa el batch size "
            "guardado en el checkpoint."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# SEED DE EVALUACIÓN
# --------------------------------------------------

def set_eval_seed(seed):
    """
    Fija la seed antes de evaluar.

    Esto hace determinista el FPS interno
    de PointNet++ durante validación/test.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# EVALUACIÓN
# --------------------------------------------------

def evaluate(
    model,
    loader,
    criterion,
    device,
    eval_seed,
):

    # --------------------------------------------------
    # DETERMINISMO DE EVALUACIÓN
    # --------------------------------------------------

    set_eval_seed(
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

    # --------------------------------------------------
    # TIMING
    # --------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

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

            # --------------------------------------
            # FORWARD
            # --------------------------------------

            predictions, _ = model(
                points
            )

            # --------------------------------------
            # LOSS
            # --------------------------------------

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

            # --------------------------------------
            # PREDICCIONES
            # --------------------------------------

            predicted_labels = torch.argmax(
                predictions,
                dim=2,
            )

            # --------------------------------------
            # MATRIZ DE CONFUSIÓN
            # --------------------------------------

            update_confusion_matrix(
                confusion_matrix,
                predicted_labels,
                labels,
                num_classes=NUM_CLASSES,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    evaluation_seconds = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------

    mean_loss = (
        total_loss
        / total_batches
    )

    metrics = compute_segmentation_metrics(
        confusion_matrix
    )

    return (
        mean_loss,
        metrics,
        confusion_matrix,
        evaluation_seconds,
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_args()

    # --------------------------------------------------
    # CHECKPOINT PATH
    # --------------------------------------------------

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "No existe el checkpoint:\n"
            f"{checkpoint_path}"
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
    print("POINTNET++ PHENO4D EVALUATION")
    print("-----------------------------")

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
    # CARGAR CHECKPOINT
    # --------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    species = checkpoint[
        "species"
    ]

    method = checkpoint[
        "method"
    ]

    target = checkpoint[
        "target"
    ]

    train_seed = checkpoint[
        "train_seed"
    ]

    eval_seed = checkpoint[
        "eval_seed"
    ]

    saved_batch_size = checkpoint[
        "batch_size"
    ]

    if args.batch_size is None:

        batch_size = (
            saved_batch_size
        )

    else:

        batch_size = (
            args.batch_size
        )

    print()
    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "Split:",
        args.split
    )

    print(
        "Species:",
        species
    )

    print(
        "Method:",
        method
    )

    print(
        "Target:",
        target
    )

    print(
        "Train seed:",
        train_seed
    )

    print(
        "Eval seed:",
        eval_seed
    )

    print(
        "Batch size:",
        batch_size
    )

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    dataset = Pheno4DProcessedDataset(
        processed_root=PROCESSED_ROOT,
        species=species,
        method=method,
        target=target,
        split=args.split,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print()
    print(
        "Samples:",
        len(dataset)
    )

    # --------------------------------------------------
    # MODELO
    # --------------------------------------------------

    model = PointNet2SemanticSegmentation(
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    criterion = PointNet2Loss()

    # --------------------------------------------------
    # MEMORIA GPU
    # --------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # EVALUAR
    # --------------------------------------------------

    (
        loss,
        metrics,
        confusion_matrix,
        evaluation_seconds,
    ) = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        eval_seed=eval_seed,
    )

    # --------------------------------------------------
    # MEMORIA GPU
    # --------------------------------------------------

    if torch.cuda.is_available():

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    else:

        peak_gpu_memory_mb = None

    # --------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------

    print()
    print("EVALUATION RESULTS")
    print("------------------")

    print(
        "Loss:",
        f"{loss:.6f}"
    )

    print(
        "IoU stem:",
        f"{metrics['iou_stem']:.6f}"
    )

    print(
        "IoU leaf:",
        f"{metrics['iou_leaf']:.6f}"
    )

    print(
        "mIoU:",
        f"{metrics['miou']:.6f}"
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix.detach().cpu()
    )

    print()

    print(
        "Evaluation time:",
        f"{evaluation_seconds:.4f} s"
    )

    if peak_gpu_memory_mb is not None:

        print(
            "Peak GPU memory:",
            f"{peak_gpu_memory_mb:.2f} MB"
        )

    # --------------------------------------------------
    # GUARDAR JSON
    # --------------------------------------------------

    output_dir = (
        checkpoint_path.parent
    )

    output_path = (
        output_dir
        / f"evaluation_{args.split}.json"
    )

    result = {
        "checkpoint_path": str(
            checkpoint_path
        ),

        "split": (
            args.split
        ),

        "species": (
            species
        ),

        "method": (
            method
        ),

        "target": (
            target
        ),

        "train_seed": (
            train_seed
        ),

        "eval_seed": (
            eval_seed
        ),

        "batch_size": (
            batch_size
        ),

        "samples": (
            len(dataset)
        ),

        "loss": (
            loss
        ),

        "iou_stem": (
            metrics[
                "iou_stem"
            ]
        ),

        "iou_leaf": (
            metrics[
                "iou_leaf"
            ]
        ),

        "miou": (
            metrics[
                "miou"
            ]
        ),

        "confusion_matrix": (
            confusion_matrix
            .detach()
            .cpu()
            .tolist()
        ),

        "evaluation_seconds": (
            evaluation_seconds
        ),

        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),
    }

    with open(
        output_path,
        "w",
    ) as json_file:

        json.dump(
            result,
            json_file,
            indent=2,
        )

    print()
    print(
        "Evaluation saved:",
        output_path
    )

    assert (
        output_path.exists()
    )

    # --------------------------------------------------
    # REPRODUCCIÓN DE VALIDACIÓN
    # --------------------------------------------------

    if args.split == "val":

        tolerance = 1e-6

        miou_difference = abs(
            metrics["miou"]
            - checkpoint[
                "miou"
            ]
        )

        stem_difference = abs(
            metrics["iou_stem"]
            - checkpoint[
                "iou_stem"
            ]
        )

        leaf_difference = abs(
            metrics["iou_leaf"]
            - checkpoint[
                "iou_leaf"
            ]
        )

        print()
        print(
            "Checkpoint mIoU difference:",
            f"{miou_difference:.10f}"
        )

        assert (
            miou_difference
            < tolerance
        )

        assert (
            stem_difference
            < tolerance
        )

        assert (
            leaf_difference
            < tolerance
        )

        print(
            "CHECKPOINT VALIDATION REPRODUCED OK"
        )

    print()
    print("EVALUATION RUN OK")
    print()


if __name__ == "__main__":
    main()
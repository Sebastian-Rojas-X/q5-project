"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Compute confusion matrices and the frozen stem, leaf, and mean IoU metrics.
"""

import torch


# --------------------------------------------------
# MATRIZ DE CONFUSIÓN
# --------------------------------------------------

def update_confusion_matrix(
    confusion_matrix,
    predictions,
    targets,
    num_classes=2,
):
    """
    Acumula una matriz de confusión.

    Parameters
    ----------
    confusion_matrix : torch.Tensor
        Matriz de forma:
            (num_classes, num_classes)

        Filas:
            clase real

        Columnas:
            clase predicha

    predictions : torch.Tensor
        Etiquetas predichas.
        Puede tener cualquier forma.

    targets : torch.Tensor
        Etiquetas reales.
        Debe tener la misma cantidad de elementos
        que predictions.

    num_classes : int
        Número de clases.

    Returns
    -------
    confusion_matrix : torch.Tensor
        Matriz actualizada.
    """

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    if predictions.numel() != targets.numel():
        raise ValueError(
            "predictions y targets deben tener "
            "el mismo número de elementos."
        )

    # --------------------------------------------------
    # VALIDAR RANGO
    # --------------------------------------------------

    valid_mask = (
        (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )

    predictions = predictions[
        valid_mask
    ]

    targets = targets[
        valid_mask
    ]

    # --------------------------------------------------
    # CODIFICACIÓN
    #
    # índice = real * num_classes + predicha
    # --------------------------------------------------

    indices = (
        targets * num_classes
        + predictions
    )

    counts = torch.bincount(
        indices,
        minlength=(
            num_classes
            * num_classes
        ),
    )

    counts = counts.reshape(
        num_classes,
        num_classes,
    )

    confusion_matrix += counts.to(
        confusion_matrix.device
    )

    return confusion_matrix


# --------------------------------------------------
# IoU
# --------------------------------------------------

def compute_iou_from_confusion_matrix(
    confusion_matrix,
):
    """
    Calcula IoU por clase y mIoU.

    Para cada clase:

        IoU = TP / (TP + FP + FN)

    Returns
    -------
    iou_per_class : torch.Tensor
        IoU de cada clase.

    miou : float
        Media de los IoU válidos.
    """

    confusion_matrix = (
        confusion_matrix.float()
    )

    # TP
    true_positive = torch.diag(
        confusion_matrix
    )

    # TP + FN
    actual = confusion_matrix.sum(
        dim=1
    )

    # TP + FP
    predicted = confusion_matrix.sum(
        dim=0
    )

    denominator = (
        actual
        + predicted
        - true_positive
    )

    iou_per_class = torch.empty(
        confusion_matrix.shape[0],
        dtype=torch.float32,
        device=confusion_matrix.device,
    )

    for class_index in range(
        confusion_matrix.shape[0]
    ):

        if denominator[
            class_index
        ] > 0:

            iou_per_class[
                class_index
            ] = (
                true_positive[
                    class_index
                ]
                / denominator[
                    class_index
                ]
            )

        else:

            iou_per_class[
                class_index
            ] = float("nan")

    valid_iou = iou_per_class[
        torch.isfinite(
            iou_per_class
        )
    ]

    if valid_iou.numel() == 0:

        miou = float("nan")

    else:

        miou = float(
            valid_iou.mean().item()
        )

    return (
        iou_per_class,
        miou,
    )


# --------------------------------------------------
# MÉTRICAS BINARIAS PHENO4D
# --------------------------------------------------

def compute_segmentation_metrics(
    confusion_matrix,
):
    """
    Devuelve las métricas del benchmark Pheno4D.

    Clases:

        0 = stem
        1 = leaf

    Returns
    -------
    metrics : dict

        {
            "iou_stem": ...,
            "iou_leaf": ...,
            "miou": ...
        }
    """

    if confusion_matrix.shape != (
        2,
        2,
    ):

        raise ValueError(
            "Para este benchmark se espera "
            "una matriz de confusión 2x2."
        )

    iou_per_class, miou = (
        compute_iou_from_confusion_matrix(
            confusion_matrix
        )
    )

    metrics = {
        "iou_stem": float(
            iou_per_class[0].item()
        ),
        "iou_leaf": float(
            iou_per_class[1].item()
        ),
        "miou": miou,
    }

    return metrics
"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Compute the frozen Euclidean Chamfer distance for original and reduced clouds.
"""

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(points_a, points_b):
    """
    Chamfer Distance simétrica general.

    CD(A,B) = 0.5 * (
        mean(A -> B)
        +
        mean(B -> A)
    )

    Usa distancia Euclidiana L2 NO elevada al cuadrado.

    Esta función NO hace ninguna suposición sobre
    la relación entre ambas nubes.
    """

    if points_a.ndim != 2 or points_a.shape[1] != 3:
        raise ValueError(
            "points_a debe tener forma (N, 3)."
        )

    if points_b.ndim != 2 or points_b.shape[1] != 3:
        raise ValueError(
            "points_b debe tener forma (M, 3)."
        )

    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError(
            "Las nubes no pueden estar vacías."
        )

    # A -> B
    tree_b = cKDTree(points_b)

    distances_a_to_b, _ = tree_b.query(
        points_a,
        k=1,
        workers=-1,
    )

    mean_a_to_b = float(
        np.mean(distances_a_to_b)
    )

    # B -> A
    tree_a = cKDTree(points_a)

    distances_b_to_a, _ = tree_a.query(
        points_b,
        k=1,
        workers=-1,
    )

    mean_b_to_a = float(
        np.mean(distances_b_to_a)
    )

    chamfer = 0.5 * (
        mean_a_to_b
        +
        mean_b_to_a
    )

    return (
        chamfer,
        mean_a_to_b,
        mean_b_to_a,
    )


def chamfer_distance_subset(
    original_points,
    reduced_points,
):
    """
    Chamfer Distance optimizada para nuestro benchmark.

    IMPORTANTE
    ----------
    Esta función solo debe usarse cuando reduced_points
    está formada exclusivamente por puntos que pertenecen
    exactamente a original_points.

    Esto se cumple para nuestros métodos:

        Random Sampling
        UVS / Voxel
        FPS

    porque todos conservan puntos reales de Pheno4D.

    Entonces:

        Reduced -> Original = 0

    y solo necesitamos calcular:

        Original -> Reduced

    Finalmente:

        Chamfer = 0.5 * (Original -> Reduced)

    Esta optimización evita construir un KD-Tree gigante
    sobre la nube original únicamente para recuperar cero.
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

    if (
        len(original_points) == 0
        or len(reduced_points) == 0
    ):
        raise ValueError(
            "Las nubes no pueden estar vacías."
        )

    # --------------------------------------------------
    # ORIGINAL -> REDUCED
    # --------------------------------------------------

    tree_reduced = cKDTree(
        reduced_points
    )

    distances_original_to_reduced, _ = (
        tree_reduced.query(
            original_points,
            k=1,
            workers=-1,
        )
    )

    original_to_reduced = float(
        np.mean(
            distances_original_to_reduced
        )
    )

    # --------------------------------------------------
    # REDUCED -> ORIGINAL
    #
    # Es exactamente cero por construcción.
    # --------------------------------------------------

    reduced_to_original = 0.0

    # --------------------------------------------------
    # CHAMFER SIMÉTRICA
    # --------------------------------------------------

    chamfer = 0.5 * (
        original_to_reduced
        +
        reduced_to_original
    )

    return (
        chamfer,
        original_to_reduced,
        reduced_to_original,
    )
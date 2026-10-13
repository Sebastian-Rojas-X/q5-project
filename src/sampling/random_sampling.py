"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Select original points using seeded random sampling without replacement.
"""

import numpy as np


def random_sampling(points, labels, n_points, seed=42):
    """
    Reduce una nube de puntos mediante muestreo aleatorio.

    Parámetros
    ----------
    points : np.ndarray
        Array de forma (N, 3) con coordenadas XYZ.

    labels : np.ndarray
        Array de forma (N,) con las etiquetas semánticas.

    n_points : int
        Número objetivo de puntos después del downsampling.

    seed : int
        Semilla usada SOLO para el sampling.
        Esto es independiente de las seeds de entrenamiento.

    Retorna
    -------
    sampled_points : np.ndarray
        Array de forma (n_points, 3).

    sampled_labels : np.ndarray
        Etiquetas correspondientes a los puntos seleccionados.
    """

    if len(points) != len(labels):
        raise ValueError(
            "points y labels deben tener la misma cantidad de elementos."
        )

    if n_points > len(points):
        raise ValueError(
            f"Se solicitaron {n_points} puntos, "
            f"pero la nube solo contiene {len(points)}."
        )

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(points),
        size=n_points,
        replace=False,
    )

    sampled_points = points[indices]
    sampled_labels = labels[indices]

    return sampled_points, sampled_labels
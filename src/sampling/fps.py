"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Select original points using deterministic external farthest point sampling.
"""

import torch


def farthest_point_sampling(points, labels, n_points, device="cuda"):
    """
    Farthest Point Sampling (FPS) usando PyTorch.

    Parámetros
    ----------
    points : np.ndarray
        Array (N, 3) con XYZ.

    labels : np.ndarray
        Array (N,) con etiquetas:
            0 = stem
            1 = leaf

    n_points : int
        Número objetivo de puntos.

    device : str
        "cuda" para GPU o "cpu".

    Retorna
    -------
    sampled_points : np.ndarray
        Array (n_points, 3).

    sampled_labels : np.ndarray
        Array (n_points,).
    """

    if len(points) != len(labels):
        raise ValueError(
            "points y labels deben tener la misma cantidad de elementos."
        )

    if n_points > len(points):
        raise ValueError(
            f"Se solicitaron {n_points} puntos, "
            f"pero la nube contiene {len(points)}."
        )

    # Convertimos puntos a tensor en GPU.
    xyz = torch.as_tensor(
        points,
        dtype=torch.float32,
        device=device,
    )

    n_total = xyz.shape[0]

    # Índices seleccionados por FPS.
    selected_indices = torch.zeros(
        n_points,
        dtype=torch.long,
        device=device,
    )

    # Distancia mínima de cada punto
    # respecto de cualquier punto ya seleccionado.
    min_distances = torch.full(
        (n_total,),
        float("inf"),
        device=device,
    )

    # Punto inicial.
    # Usamos índice 0 para que sea determinista.
    farthest_index = torch.tensor(
        0,
        dtype=torch.long,
        device=device,
    )

    for i in range(n_points):

        selected_indices[i] = farthest_index

        selected_point = xyz[farthest_index]

        # Distancia cuadrática desde todos los puntos
        # al último punto seleccionado.
        distances = torch.sum(
            (xyz - selected_point) ** 2,
            dim=1,
        )

        # Actualizamos la distancia al conjunto FPS.
        min_distances = torch.minimum(
            min_distances,
            distances,
        )

        # El siguiente punto es el que está más lejos
        # de todos los ya seleccionados.
        farthest_index = torch.argmax(min_distances)

    # Volvemos a CPU para indexar los arrays NumPy.
    selected_indices = (
        selected_indices
        .cpu()
        .numpy()
    )

    sampled_points = points[selected_indices]
    sampled_labels = labels[selected_indices]

    return sampled_points, sampled_labels
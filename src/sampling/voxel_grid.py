"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Select original points by voxel grid and adjust cardinality using FPS.
"""

import numpy as np

from fps import farthest_point_sampling


def _get_voxel_coordinates(points, voxel_size):
    """
    Calcula el índice 3D del voxel al que pertenece cada punto.

    La grilla se ancla en el mínimo XYZ de la nube.
    """

    min_bound = np.min(points, axis=0)

    voxel_coordinates = np.floor(
        (points - min_bound) / voxel_size
    ).astype(np.int64)

    return voxel_coordinates, min_bound


def count_occupied_voxels(points, voxel_size):
    """
    Cuenta cuántos voxels contienen al menos un punto.

    Esta función se usa durante la búsqueda del voxel_size.
    Aquí todavía NO necesitamos seleccionar el punto
    representativo de cada voxel.
    """

    voxel_coordinates, _ = _get_voxel_coordinates(
        points,
        voxel_size,
    )

    unique_voxels = np.unique(
        voxel_coordinates,
        axis=0,
    )

    return len(unique_voxels)


def voxel_grid_sampling(points, labels, voxel_size):
    """
    Uniform Voxel Sampling (UVS).

    Para cada voxel:

    1. Calcula el centro geométrico del cubo.
    2. Busca entre los puntos originales del voxel.
    3. Conserva el punto real más cercano a ese centro.
    4. Conserva también su etiqueta original.

    Esto evita crear puntos artificiales y sigue
    la estrategia UVS descrita por Li et al. (2023).

    Parámetros
    ----------
    points : np.ndarray
        Array (N, 3) con XYZ.

    labels : np.ndarray
        Array (N,) con etiquetas.

    voxel_size : float
        Longitud del lado del voxel.

    Retorna
    -------
    sampled_points : np.ndarray

    sampled_labels : np.ndarray
    """

    if len(points) != len(labels):
        raise ValueError(
            "points y labels deben tener la misma cantidad de elementos."
        )

    if voxel_size <= 0:
        raise ValueError(
            "voxel_size debe ser mayor que cero."
        )

    # --------------------------------------------------
    # 1. ASIGNAR CADA PUNTO A UN VOXEL
    # --------------------------------------------------

    voxel_coordinates, min_bound = _get_voxel_coordinates(
        points,
        voxel_size,
    )

    # --------------------------------------------------
    # 2. IDENTIFICAR VOXELS ÚNICOS
    #
    # inverse_indices indica a qué voxel único pertenece
    # cada punto original.
    # --------------------------------------------------

    unique_voxels, inverse_indices = np.unique(
        voxel_coordinates,
        axis=0,
        return_inverse=True,
    )

    # --------------------------------------------------
    # 3. CALCULAR CENTRO GEOMÉTRICO DE CADA VOXEL
    #
    # IMPORTANTE:
    # este NO es el centroide de los puntos contenidos.
    #
    # Es el centro geométrico del cubo:
    #
    # min_bound + (voxel_index + 0.5) * voxel_size
    # --------------------------------------------------

    voxel_centers = (
        min_bound
        + (unique_voxels + 0.5) * voxel_size
    )

    # Para cada punto obtenemos el centro del voxel
    # al que pertenece.
    centers_for_points = voxel_centers[inverse_indices]

    # --------------------------------------------------
    # 4. DISTANCIA DEL PUNTO AL CENTRO DEL VOXEL
    # --------------------------------------------------

    squared_distances = np.sum(
        (points - centers_for_points) ** 2,
        axis=1,
    )

    # --------------------------------------------------
    # 5. ELEGIR UN PUNTO REAL POR VOXEL
    #
    # Ordenamos primero por voxel y luego por distancia.
    # De esta forma, el primer punto de cada voxel será
    # el más cercano a su centro geométrico.
    # --------------------------------------------------

    order = np.lexsort(
        (
            squared_distances,
            inverse_indices,
        )
    )

    sorted_voxel_ids = inverse_indices[order]

    # El primer elemento siempre corresponde
    # al primer voxel del arreglo ordenado.
    first_mask = np.ones(
        len(order),
        dtype=bool,
    )

    first_mask[1:] = (
        sorted_voxel_ids[1:]
        != sorted_voxel_ids[:-1]
    )

    selected_indices = order[first_mask]

    sampled_points = points[selected_indices]
    sampled_labels = labels[selected_indices]

    return sampled_points, sampled_labels


def find_voxel_size_for_target(
    points,
    labels,
    target_points,
    min_voxel_size=0.01,
    max_voxel_size=10.0,
    max_iterations=30,
    device="cuda",
):
    """
    Busca un voxel_size que produzca AL MENOS target_points.

    El objetivo no es exigir que voxelización produzca
    exactamente N puntos, porque ese número depende de
    la ocupación espacial de los voxels.

    Política:

        nube original
              |
              v
        voxelización UVS
              |
              v
        M puntos, con M >= target
              |
              +---- M == target ---> resultado
              |
              +---- M > target ----> FPS ---> target

    De esta forma, la salida final contiene siempre
    exactamente target_points.

    Retorna
    -------
    voxel_size : float

    final_points : np.ndarray
        Exactamente target_points.

    final_labels : np.ndarray
        Etiquetas correspondientes.
    """

    if len(points) != len(labels):
        raise ValueError(
            "points y labels deben tener la misma cantidad de elementos."
        )

    if target_points <= 0:
        raise ValueError(
            "target_points debe ser mayor que cero."
        )

    if target_points > len(points):
        raise ValueError(
            f"Se solicitaron {target_points} puntos, "
            f"pero la nube contiene {len(points)}."
        )

    # --------------------------------------------------
    # 1. VERIFICAR LÍMITE INFERIOR
    # --------------------------------------------------

    count_min = count_occupied_voxels(
        points,
        min_voxel_size,
    )

    if count_min < target_points:
        raise ValueError(
            "min_voxel_size es demasiado grande: "
            f"solo produce {count_min} voxels para "
            f"un target de {target_points}."
        )

    # --------------------------------------------------
    # 2. ASEGURAR QUE EL LÍMITE SUPERIOR
    #    PRODUZCA MENOS DEL TARGET
    #
    # Si todavía produce demasiados puntos,
    # aumentamos automáticamente el voxel.
    # --------------------------------------------------

    count_max = count_occupied_voxels(
        points,
        max_voxel_size,
    )

    expansion = 0

    while count_max >= target_points:

        max_voxel_size *= 2.0

        count_max = count_occupied_voxels(
            points,
            max_voxel_size,
        )

        expansion += 1

        if expansion > 20:
            raise RuntimeError(
                "No fue posible encontrar un límite superior "
                "adecuado para voxel_size."
            )

    # --------------------------------------------------
    # 3. BÚSQUEDA BINARIA
    #
    # Guardamos siempre la mejor solución que tenga
    # M >= target.
    # --------------------------------------------------

    low = min_voxel_size
    high = max_voxel_size

    best_voxel_size = low
    best_count = count_min

    for iteration in range(max_iterations):

        voxel_size = (low + high) / 2.0

        current_count = count_occupied_voxels(
            points,
            voxel_size,
        )

        print(
            f"Iteración {iteration + 1:02d} | "
            f"voxel_size={voxel_size:.6f} | "
            f"points={current_count}"
        )

        # --------------------------------------------------
        # Queremos acercarnos al target DESDE ARRIBA.
        # --------------------------------------------------

        if current_count >= target_points:

            if current_count < best_count:
                best_voxel_size = voxel_size
                best_count = current_count

            # Podemos aumentar el voxel_size para
            # intentar reducir más puntos.
            low = voxel_size

        else:

            # Nos pasamos: hay menos puntos que target.
            # Hay que disminuir voxel_size.
            high = voxel_size

        # Si casualmente encontramos exactamente N,
        # podemos terminar la búsqueda.
        if current_count == target_points:

            best_voxel_size = voxel_size
            best_count = current_count
            break

    # --------------------------------------------------
    # 4. EJECUTAR UVS REAL CON EL MEJOR VOXEL_SIZE
    # --------------------------------------------------

    voxel_points, voxel_labels = voxel_grid_sampling(
        points,
        labels,
        best_voxel_size,
    )

    print()
    print(
        f"UVS produjo {len(voxel_points)} puntos "
        f"para target={target_points}"
    )

    # --------------------------------------------------
    # 5. FIJAR EXACTAMENTE N
    # --------------------------------------------------

    if len(voxel_points) > target_points:

        print(
            f"Aplicando FPS final: "
            f"{len(voxel_points)} -> {target_points}"
        )

        final_points, final_labels = farthest_point_sampling(
            voxel_points,
            voxel_labels,
            n_points=target_points,
            device=device,
        )

    elif len(voxel_points) == target_points:

        print(
            "No se requiere FPS final: "
            "UVS ya produjo exactamente el target."
        )

        final_points = voxel_points
        final_labels = voxel_labels

    else:

        # Esto no debería ocurrir porque la búsqueda
        # conserva una solución desde arriba.
        raise RuntimeError(
            "Error interno: UVS produjo menos puntos "
            "que el target."
        )

    # --------------------------------------------------
    # 6. VERIFICACIÓN FINAL
    # --------------------------------------------------

    if len(final_points) != target_points:
        raise RuntimeError(
            f"Se esperaban {target_points} puntos, "
            f"pero se obtuvieron {len(final_points)}."
        )

    return (
        best_voxel_size,
        final_points,
        final_labels,
    )
"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Load Pheno4D coordinates, remove ground, and map stem and leaf labels.
"""

from pathlib import Path
import numpy as np


def load_pheno4d_cloud(file_path, species):
    """
    Carga una nube etiquetada de Pheno4D y devuelve solo puntos de planta.

    Salida:
        points : array (N, 3) con XYZ
        labels : array (N,) con:
                 0 = stem
                 1 = leaf

    Decisión del proyecto:
        - eliminar soil
        - tomate: etiqueta en columna 4
        - maíz: usar columna 4 (Leaf Collar)
    """

    file_path = Path(file_path)

    # Cargamos todo el archivo como números.
    data = np.loadtxt(file_path)

    # Las primeras tres columnas siempre son XYZ.
    points = data[:, :3]

    if species == "tomato":
        # Tomate etiquetado:
        # X Y Z label
        raw_labels = data[:, 3].astype(int)

    elif species == "maize":
        # Maíz etiquetado:
        # X Y Z leaf_collar leaf_tip
        #
        # Para Q5 usamos Leaf Collar porque conserva
        # una clase explícita de tallo.
        raw_labels = data[:, 3].astype(int)

    else:
        raise ValueError(
            f"Especie desconocida: {species}. "
            "Use 'maize' o 'tomato'."
        )

    # Eliminamos soil.
    # Convención original:
    # 0 = soil
    plant_mask = raw_labels != 0

    points = points[plant_mask]
    raw_labels = raw_labels[plant_mask]

    # Remapeo semántico para nuestro benchmark:
    #
    # raw label 1     -> stem -> 0
    # raw label >= 2  -> leaf -> 1
    labels = np.zeros(len(raw_labels), dtype=np.int64)

    labels[raw_labels >= 2] = 1

    return points, labels


if __name__ == "__main__":

    test_file = "data/raw/Pheno4D/Tomato01/T01_0305_a.txt"

    points, labels = load_pheno4d_cloud(
        test_file,
        species="tomato",
    )

    print("Archivo:", test_file)
    print("Shape points:", points.shape)
    print("Shape labels:", labels.shape)

    print("Stem points:", np.sum(labels == 0))
    print("Leaf points:", np.sum(labels == 1))
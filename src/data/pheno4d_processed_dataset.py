"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Load and normalize precomputed point clouds for PointNet++.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------

VALID_SPECIES = {
    "maize",
    "tomato",
}

VALID_METHODS = {
    "random",
    "voxel",
    "fps",
    "control_48000",
}

VALID_TARGETS = {
    48000,
    16384,
    8192,
    4096,
    2048,
    1024,
}

VALID_SPLITS = {
    "train",
    "val",
    "test",
}


# --------------------------------------------------
# PARÁMETROS DE AUGMENTATION
#
# Se aplican SOLO durante entrenamiento.
#
# Rotación:
#   - alrededor del eje Z
#   - ángulo uniforme entre 0 y 360 grados
#
# Stretching:
#   - escala independiente por eje
#   - cada factor en [0.9, 1.1]
#
# Translation:
#   - después de normalizar
#   - desplazamiento independiente por eje
#   - rango [-0.1, 0.1]
# --------------------------------------------------

ROTATION_MIN = 0.0
ROTATION_MAX = 2.0 * np.pi

STRETCH_MIN = 0.90
STRETCH_MAX = 1.10

TRANSLATION_MIN = -0.10
TRANSLATION_MAX = 0.10


# --------------------------------------------------
# AUGMENTATION: ROTACIÓN EN Z
# --------------------------------------------------

def random_rotation_z(points):
    """
    Aplica una rotación aleatoria alrededor del eje Z.

    Parameters
    ----------
    points : np.ndarray
        Array de forma (N, 3).

    Returns
    -------
    np.ndarray
        Puntos rotados.
    """

    angle = np.random.uniform(
        ROTATION_MIN,
        ROTATION_MAX,
    )

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array(
        [
            [
                cos_angle,
                -sin_angle,
                0.0,
            ],
            [
                sin_angle,
                cos_angle,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    rotated_points = (
        points
        @ rotation_matrix.T
    )

    return rotated_points.astype(
        np.float32,
        copy=False,
    )


# --------------------------------------------------
# AUGMENTATION: STRETCHING
# --------------------------------------------------

def random_stretching(points):
    """
    Estira o comprime la nube de forma independiente
    en X, Y y Z.

    Cada eje recibe un factor aleatorio independiente
    entre STRETCH_MIN y STRETCH_MAX.

    Ejemplo:

        x *= 1.03
        y *= 0.94
        z *= 1.08

    Parameters
    ----------
    points : np.ndarray
        Array de forma (N, 3).

    Returns
    -------
    np.ndarray
        Puntos deformados.
    """

    scale_factors = np.random.uniform(
        STRETCH_MIN,
        STRETCH_MAX,
        size=3,
    ).astype(
        np.float32
    )

    stretched_points = (
        points
        * scale_factors
    )

    return stretched_points.astype(
        np.float32,
        copy=False,
    )


# --------------------------------------------------
# AUGMENTATION: TRANSLATION
# --------------------------------------------------

def random_translation(points):
    """
    Aplica una pequeña traslación aleatoria.

    IMPORTANTE:

    Esta operación se aplica DESPUÉS de normalizar.

    Si se aplicara antes de restar el centroide,
    la normalización eliminaría prácticamente
    toda la traslación.

    Parameters
    ----------
    points : np.ndarray
        Array de forma (N, 3).

    Returns
    -------
    np.ndarray
        Puntos trasladados.
    """

    translation = np.random.uniform(
        TRANSLATION_MIN,
        TRANSLATION_MAX,
        size=3,
    ).astype(
        np.float32
    )

    translated_points = (
        points
        + translation
    )

    return translated_points.astype(
        np.float32,
        copy=False,
    )


# --------------------------------------------------
# NORMALIZACIÓN
# --------------------------------------------------

def normalize_point_cloud(points):
    """
    Normaliza una nube de puntos usando:

    1. Restar el centroide.
    2. Dividir por la máxima distancia euclidiana
       al origen.

    Después de esta operación:

        centroid ~= [0, 0, 0]

    y:

        max(||p_i||) = 1

    Parameters
    ----------
    points : np.ndarray
        Array (N, 3).

    Returns
    -------
    np.ndarray
        Nube normalizada.
    """

    points = points.astype(
        np.float32,
        copy=True,
    )

    # ----------------------------------------------
    # CENTRAR
    # ----------------------------------------------

    centroid = np.mean(
        points,
        axis=0,
        keepdims=True,
    )

    points = (
        points
        - centroid
    )

    # ----------------------------------------------
    # RADIO MÁXIMO
    # ----------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1,
    )

    max_distance = np.max(
        distances
    )

    if (
        not np.isfinite(
            max_distance
        )
    ):
        raise ValueError(
            "La nube contiene un radio máximo "
            "no finito."
        )

    if max_distance <= 0.0:
        raise ValueError(
            "No se puede normalizar una nube "
            "con radio máximo <= 0."
        )

    # ----------------------------------------------
    # ESCALAR A ESFERA UNITARIA
    # ----------------------------------------------

    points = (
        points
        / max_distance
    )

    return points.astype(
        np.float32,
        copy=False,
    )


# --------------------------------------------------
# PIPELINE DE AUGMENTATION
# --------------------------------------------------

def augment_point_cloud(points):
    """
    Pipeline de augmentation usado durante train.

    Orden:

        1. Rotation Z
        2. Stretching
        3. Normalización
        4. Translation

    La normalización NO se hace aquí.

    Esta función solamente realiza las
    transformaciones geométricas que deben ocurrir
    antes de la normalización:

        rotation
        stretching

    La translation se aplica posteriormente en
    __getitem__(), después de normalizar.
    """

    points = random_rotation_z(
        points
    )

    points = random_stretching(
        points
    )

    return points


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class Pheno4DProcessedDataset(
    Dataset
):
    """
    Dataset para las nubes precomputadas de Pheno4D.

    Estructura esperada:

        data/processed/
            maize/
                random/
                    16384/
                        train/
                        val/
                        test/

                control_48000/
                    train/
                    val/
                    test/

    Cada archivo .npz debe contener:

        points : (N, 3)
        labels : (N,)

    Clases semánticas:

        0 -> stem
        1 -> leaf

    Parameters
    ----------
    processed_root : str o Path
        Raíz de los datos procesados.

    species : str
        maize o tomato.

    method : str
        random, voxel, fps o control_48000.

    target : int
        Número de puntos de la nube.

    split : str
        train, val o test.

    normalize : bool
        Si True, centra la nube y la lleva
        a radio máximo 1.

    augment : bool
        Si True, aplica augmentation geométrica.

        Por seguridad, solamente se permite
        cuando split == "train".
    """

    def __init__(
        self,
        processed_root,
        species,
        method,
        target,
        split,
        normalize=True,
        augment=False,
    ):

        super().__init__()

        # ------------------------------------------
        # VALIDACIONES
        # ------------------------------------------

        if species not in VALID_SPECIES:

            raise ValueError(
                f"Species inválida: {species}. "
                f"Opciones: {sorted(VALID_SPECIES)}"
            )

        if method not in VALID_METHODS:

            raise ValueError(
                f"Method inválido: {method}. "
                f"Opciones: {sorted(VALID_METHODS)}"
            )

        if target not in VALID_TARGETS:

            raise ValueError(
                f"Target inválido: {target}. "
                f"Opciones: {sorted(VALID_TARGETS)}"
            )

        if split not in VALID_SPLITS:

            raise ValueError(
                f"Split inválido: {split}. "
                f"Opciones: {sorted(VALID_SPLITS)}"
            )

        if (
            augment
            and split != "train"
        ):

            raise ValueError(
                "augment=True solamente está permitido "
                "para split='train'."
            )

        # ------------------------------------------
        # CONFIGURACIÓN
        # ------------------------------------------

        self.processed_root = Path(
            processed_root
        )

        self.species = species
        self.method = method
        self.target = target
        self.split = split

        self.normalize = bool(
            normalize
        )

        self.augment = bool(
            augment
        )

        # ------------------------------------------
        # DIRECTORIO
        # ------------------------------------------

        if self.method == "control_48000":

            if self.target != 48000:
                raise ValueError(
                    "control_48000 requiere target=48000."
                )

            self.data_dir = (
                self.processed_root
                / self.species
                / "control_48000"
                / self.split
            )

        else:

            if self.target == 48000:
                raise ValueError(
                    "target=48000 solamente está permitido "
                    "para method='control_48000'."
                )

            self.data_dir = (
                self.processed_root
                / self.species
                / self.method
                / str(
                    self.target
                )
                / self.split
            )

        if not self.data_dir.exists():

            raise FileNotFoundError(
                "No existe el directorio:\n"
                f"{self.data_dir}"
            )

        # ------------------------------------------
        # ARCHIVOS NPZ
        # ------------------------------------------

        self.files = sorted(
            self.data_dir.glob(
                "*.npz"
            )
        )

        if len(
            self.files
        ) == 0:

            raise RuntimeError(
                "No se encontraron archivos .npz en:\n"
                f"{self.data_dir}"
            )

    # --------------------------------------------------
    # LEN
    # --------------------------------------------------

    def __len__(
        self
    ):

        return len(
            self.files
        )

    # --------------------------------------------------
    # GET ITEM
    # --------------------------------------------------

    def __getitem__(
        self,
        index,
    ):

        file_path = (
            self.files[
                index
            ]
        )

        # ------------------------------------------
        # CARGAR NPZ
        # ------------------------------------------

        with np.load(
            file_path
        ) as data:

            if (
                "points"
                not in data
            ):

                raise KeyError(
                    f"'points' no existe en {file_path}"
                )

            if (
                "labels"
                not in data
            ):

                raise KeyError(
                    f"'labels' no existe en {file_path}"
                )

            points = (
                data[
                    "points"
                ]
                .astype(
                    np.float32,
                    copy=True,
                )
            )

            labels = (
                data[
                    "labels"
                ]
                .astype(
                    np.int64,
                    copy=True,
                )
            )

        # ------------------------------------------
        # VALIDAR POINTS
        # ------------------------------------------

        if (
            points.ndim != 2
            or points.shape[1] != 3
        ):

            raise ValueError(
                f"Forma inválida de points en "
                f"{file_path}: {points.shape}. "
                "Se esperaba (N, 3)."
            )

        # ------------------------------------------
        # VALIDAR LABELS
        # ------------------------------------------

        if labels.ndim != 1:

            raise ValueError(
                f"Forma inválida de labels en "
                f"{file_path}: {labels.shape}. "
                "Se esperaba (N,)."
            )

        # ------------------------------------------
        # MISMO N
        # ------------------------------------------

        if (
            len(points)
            != len(labels)
        ):

            raise ValueError(
                f"points y labels tienen distinto "
                f"número de elementos en {file_path}."
            )

        # ------------------------------------------
        # VALIDAR TARGET
        # ------------------------------------------

        if (
            len(points)
            != self.target
        ):

            raise ValueError(
                f"{file_path} contiene "
                f"{len(points)} puntos, "
                f"pero target={self.target}."
            )

        # ------------------------------------------
        # VALIDAR FINITOS
        # ------------------------------------------

        if not np.all(
            np.isfinite(
                points
            )
        ):

            raise ValueError(
                f"Se encontraron coordenadas "
                f"no finitas en {file_path}."
            )

        # ------------------------------------------
        # VALIDAR CLASES
        # ------------------------------------------

        unique_labels = set(
            np.unique(
                labels
            ).tolist()
        )

        if not unique_labels.issubset(
            {
                0,
                1,
            }
        ):

            raise ValueError(
                f"Etiquetas inválidas en "
                f"{file_path}: "
                f"{sorted(unique_labels)}"
            )

        # ------------------------------------------
        # AUGMENTATION PRE-NORMALIZACIÓN
        #
        # SOLO TRAIN.
        # ------------------------------------------

        if self.augment:

            points = augment_point_cloud(
                points
            )

        # ------------------------------------------
        # NORMALIZACIÓN
        # ------------------------------------------

        if self.normalize:

            points = normalize_point_cloud(
                points
            )

        # ------------------------------------------
        # TRANSLATION
        #
        # Se hace DESPUÉS de normalizar para que
        # el centrado no elimine el desplazamiento.
        #
        # SOLO TRAIN.
        # ------------------------------------------

        if self.augment:

            points = random_translation(
                points
            )

        # ------------------------------------------
        # CONVERTIR A TENSOR
        # ------------------------------------------

        points = torch.from_numpy(
            points
        ).float()

        labels = torch.from_numpy(
            labels
        ).long()

        # ------------------------------------------
        # PointNet++ espera:
        #
        # points = [C, N]
        #
        # donde C = 3:
        #
        # x, y, z
        # ------------------------------------------

        points = points.transpose(
            0,
            1,
        ).contiguous()

        # ------------------------------------------
        # METADATA
        # ------------------------------------------

        metadata = {
            "file": (
                file_path.name
            ),

            "species": (
                self.species
            ),

            "method": (
                self.method
            ),

            "target": (
                self.target
            ),

            "split": (
                self.split
            ),

            "normalized": (
                self.normalize
            ),

            "augmented": (
                self.augment
            ),
        }

        return (
            points,
            labels,
            metadata,
        )
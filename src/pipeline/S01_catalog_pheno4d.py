"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Catalog Pheno4D point clouds and their annotations for the frozen benchmark.
"""

from pathlib import Path
import csv


# Ruta a la carpeta del dataset Pheno4D.
DATASET_DIR = Path("data/raw/Pheno4D")

# Archivo de salida del catálogo.
OUTPUT_CSV = Path("data/catalog_pheno4d.csv")


def detect_species(folder_name):
    """
    Detecta la especie a partir del nombre de la carpeta.
    """
    if folder_name.startswith("Maize"):
        return "maize"

    if folder_name.startswith("Tomato"):
        return "tomato"

    return "unknown"


def is_labeled_file(file_path):
    """
    En Pheno4D, las nubes anotadas usan el sufijo '_a.txt'.
    """
    return file_path.name.endswith("_a.txt")


def extract_date(file_path):
    """
    Extrae el código temporal del nombre del archivo.

    Ejemplo:
        M01_0313_a.txt -> 0313
        T01_0305_a.txt -> 0305
    """
    parts = file_path.stem.split("_")

    if len(parts) >= 2:
        return parts[1]

    return ""


def count_columns(file_path):
    """
    Lee solamente la primera línea para detectar cuántas columnas
    tiene el archivo.
    """
    with open(file_path, "r") as file:
        first_line = file.readline().strip()

    if not first_line:
        return 0

    return len(first_line.split())


def count_labeled_points(file_path, species):
    """
    Cuenta puntos totales y puntos por clase semántica.

    Convención usada:
        0     -> soil
        1     -> stem
        >= 2  -> leaf

    Para tomate:
        la etiqueta está en la columna 4.

    Para maíz:
        usamos la columna 4 (Leaf Collar).
    """

    n_points = 0
    soil_points = 0
    stem_points = 0
    leaf_points = 0

    with open(file_path, "r") as file:

        for line in file:

            values = line.split()

            if not values:
                continue

            # En ambos casos usamos la columna 4.
            # En Python el índice correspondiente es 3.
            label = int(values[3])

            n_points += 1

            if label == 0:
                soil_points += 1

            elif label == 1:
                stem_points += 1

            elif label >= 2:
                leaf_points += 1

    return n_points, soil_points, stem_points, leaf_points


def main():

    rows = []

    # Recorremos las carpetas de plantas.
    for plant_dir in sorted(DATASET_DIR.iterdir()):

        if not plant_dir.is_dir():
            continue

        species = detect_species(plant_dir.name)

        # Recorremos todas las nubes .txt de esta planta.
        for file_path in sorted(plant_dir.glob("*.txt")):

            labeled = is_labeled_file(file_path)
            n_columns = count_columns(file_path)

            # Valores por defecto para nubes no etiquetadas.
            n_points = ""
            soil_points = ""
            stem_points = ""
            leaf_points = ""

            # Solo contamos clases en archivos etiquetados.
            if labeled:

                (
                    n_points,
                    soil_points,
                    stem_points,
                    leaf_points,
                ) = count_labeled_points(file_path, species)

            row = {
                "species": species,
                "plant_id": plant_dir.name,
                "date": extract_date(file_path),
                "labeled": labeled,
                "n_columns": n_columns,
                "n_points": n_points,
                "soil_points": soil_points,
                "stem_points": stem_points,
                "leaf_points": leaf_points,
                "path": str(file_path),
            }

            rows.append(row)

    # Creamos la carpeta de salida si no existe.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Guardamos el catálogo.
    with open(OUTPUT_CSV, "w", newline="") as csv_file:

        fieldnames = [
            "species",
            "plant_id",
            "date",
            "labeled",
            "n_columns",
            "n_points",
            "soil_points",
            "stem_points",
            "leaf_points",
            "path",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"Nubes encontradas: {len(rows)}")
    print(f"Catálogo guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

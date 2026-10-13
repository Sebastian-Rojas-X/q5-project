"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Create cloud manifests from the catalog and frozen plant-level splits.
"""

from pathlib import Path
import csv
import yaml


CATALOG_PATH = Path("data/catalog_pheno4d.csv")
SPLIT_PATH = Path("configs/splits.yaml")
OUTPUT_DIR = Path("data/manifests")


def main():
    # Cargar configuración de splits.
    with open(SPLIT_PATH, "r") as file:
        splits = yaml.safe_load(file)

    # Crear carpeta de salida.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Leer catálogo completo.
    with open(CATALOG_PATH, "r") as file:
        catalog = list(csv.DictReader(file))

    for species, split_config in splits.items():
        for split_name, plants in split_config.items():

            output_path = OUTPUT_DIR / f"{species}_{split_name}.csv"

            rows = []

            for row in catalog:
                # Solo la especie correspondiente.
                if row["species"] != species:
                    continue

                # Solo nubes etiquetadas.
                if row["labeled"] != "True":
                    continue

                # Solo plantas pertenecientes a este split.
                if row["plant_id"] not in plants:
                    continue

                rows.append({
                    "species": row["species"],
                    "plant_id": row["plant_id"],
                    "date": row["date"],
                    "path": row["path"],
                })

            with open(output_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "species",
                        "plant_id",
                        "date",
                        "path",
                    ],
                )

                writer.writeheader()
                writer.writerows(rows)

            print(f"{output_path}: {len(rows)} nubes")


if __name__ == "__main__":
    main()
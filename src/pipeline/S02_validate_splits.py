"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Validate the frozen plant-level training, validation, and test partitions.
"""

from pathlib import Path
import yaml


SPLIT_FILE = Path("configs/splits.yaml")


def check_species_split(species_name, split_config):
    train = set(split_config["train"])
    val = set(split_config["val"])
    test = set(split_config["test"])

    # Comprobamos que ninguna planta aparezca en más de un split.
    assert train.isdisjoint(val), f"{species_name}: leakage entre train y val"
    assert train.isdisjoint(test), f"{species_name}: leakage entre train y test"
    assert val.isdisjoint(test), f"{species_name}: leakage entre val y test"

    all_plants = train | val | test

    print(f"{species_name}")
    print(f"  train: {sorted(train)}")
    print(f"  val:   {sorted(val)}")
    print(f"  test:  {sorted(test)}")
    print(f"  total plants: {len(all_plants)}")
    print()


def main():
    with open(SPLIT_FILE, "r") as file:
        splits = yaml.safe_load(file)

    for species_name, split_config in splits.items():
        check_species_split(species_name, split_config)

    print("Split válido: no se detectó leakage entre plantas.")


if __name__ == "__main__":
    main()
"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Measure dedicated PointNet++ forward time and GPU memory usage.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader

from src.data.pheno4d_processed_dataset import (
    Pheno4DProcessedDataset,
)

from src.models.pointnet2.pointnet2_sem_seg import (
    PointNet2SemanticSegmentation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "training"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
)

RAW_OUTPUT = (
    OUTPUT_DIR
    / "inference_benchmark_raw.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "inference_benchmark_summary.csv"
)


SPECIES_LIST = [
    "maize",
    "tomato",
]

METHOD_LIST = [
    "random",
    "voxel",
    "fps",
]

TARGET_LIST = [
    16384,
    8192,
    4096,
    2048,
    1024,
]

SEED_LIST = [
    0,
    1,
    2,
    3,
    4,
]

CONTROL_METHOD = "control_48000"
CONTROL_TARGET = 48000

NUM_CLASSES = 2

DEFAULT_BATCH_SIZE = 2
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 20

EVAL_SEED = 12345


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark limpio de inferencia "
            "para PointNet++ en Q5."
        )
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--include-control",
        action="store_true",
        help=(
            "Incluye también control_48000."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def sample_std(values):

    if len(values) < 2:
        return 0.0

    return stdev(values)


def build_configurations(
    include_control,
):

    configs = []

    for species in SPECIES_LIST:
        for method in METHOD_LIST:
            for target in TARGET_LIST:
                for seed in SEED_LIST:

                    configs.append(
                        {
                            "species": species,
                            "method": method,
                            "target": target,
                            "seed": seed,
                            "is_control": False,
                        }
                    )

    if include_control:

        for species in SPECIES_LIST:
            for seed in SEED_LIST:

                configs.append(
                    {
                        "species": species,
                        "method": CONTROL_METHOD,
                        "target": CONTROL_TARGET,
                        "seed": seed,
                        "is_control": True,
                    }
                )

    return configs


def get_run_dir(config):

    return (
        TRAINING_ROOT
        / config["species"]
        / config["method"]
        / str(config["target"])
        / f'seed_{config["seed"]}'
        / "no_aug"
    )


def get_checkpoint_path(config):

    return (
        get_run_dir(config)
        / "best_model.pt"
    )


def load_checkpoint_state(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------
    # Soportar distintos formatos de checkpoint.
    # --------------------------------------------------

    if isinstance(
        checkpoint,
        torch.nn.Module,
    ):
        return checkpoint.state_dict()

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            f"Unsupported checkpoint type: "
            f"{type(checkpoint)}"
        )

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
    ]

    for key in candidate_keys:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(
                value,
                dict,
            ):
                return value

            if isinstance(
                value,
                torch.nn.Module,
            ):
                return value.state_dict()

    # --------------------------------------------------
    # Puede ser directamente state_dict.
    # --------------------------------------------------

    if all(
        isinstance(key, str)
        for key in checkpoint.keys()
    ):

        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:
            return checkpoint

    raise RuntimeError(
        "Could not identify model state_dict "
        f"in {checkpoint_path}"
    )


def build_model(
    checkpoint_path,
    device,
):

    model = PointNet2SemanticSegmentation(
        num_classes=NUM_CLASSES,
    )

    state_dict = load_checkpoint_state(
        checkpoint_path,
        device,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


def build_dataset(
    config,
):

    # --------------------------------------------------
    # Misma nube reducida usada durante evaluación test.
    # Sin augmentations.
    # --------------------------------------------------

    dataset = Pheno4DProcessedDataset(
        processed_root=PROJECT_ROOT / "data" / "processed",
        species=config["species"],
        method=config["method"],
        target=config["target"],
        split="test",
        normalize=True,
        augment=False,
    )

    return dataset


def get_first_full_batch(
    dataset,
    batch_size,
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    for points, labels, metadata in loader:

        return points

    raise RuntimeError(
        "Dataset returned no batches."
    )


def synchronize():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_forward(
    model,
    points,
    device,
    warmup,
    repeats,
):

    points = points.to(
        device,
        non_blocking=True,
    )

    # --------------------------------------------------
    # WARM-UP
    # --------------------------------------------------

    with torch.no_grad():

        for _ in range(warmup):

            _ = model(points)

    synchronize()

    # --------------------------------------------------
    # MEMORIA
    # --------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(
            device
        )

    # --------------------------------------------------
    # TIMING PURO DEL FORWARD
    # --------------------------------------------------

    times = []

    with torch.no_grad():

        for _ in range(repeats):

            synchronize()

            start = time.perf_counter()

            _ = model(points)

            synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            times.append(
                elapsed
            )

    # --------------------------------------------------
    # MEMORIA PICO
    # --------------------------------------------------

    if torch.cuda.is_available():

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 2)
        )

    else:

        peak_memory_mb = 0.0

    batch_size_real = int(
        points.shape[0]
    )

    seconds_per_batch = mean(
        times
    )

    seconds_per_cloud = (
        seconds_per_batch
        / batch_size_real
    )

    return {
        "batch_size_real": batch_size_real,
        "forward_seconds_mean": mean(times),
        "forward_seconds_std": sample_std(times),
        "forward_seconds_min": min(times),
        "forward_seconds_max": max(times),
        "forward_seconds_per_cloud":
            seconds_per_cloud,
        "peak_gpu_memory_mb":
            peak_memory_mb,
    }


def save_csv(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def build_summary(
    raw_rows,
):

    groups = {}

    for row in raw_rows:

        key = (
            row["species"],
            row["method"],
            row["target"],
        )

        groups.setdefault(
            key,
            [],
        )

        groups[key].append(
            row
        )

    summary_rows = []

    for key in sorted(groups):

        species, method, target = key

        rows = groups[key]

        per_cloud = [
            float(
                row[
                    "forward_seconds_per_cloud"
                ]
            )
            for row in rows
        ]

        memory = [
            float(
                row[
                    "peak_gpu_memory_mb"
                ]
            )
            for row in rows
        ]

        summary_rows.append(
            {
                "species": species,
                "method": method,
                "target": target,
                "n_seeds": len(rows),

                "inference_seconds_per_cloud_mean":
                    mean(per_cloud),

                "inference_seconds_per_cloud_std":
                    sample_std(per_cloud),

                "inference_ms_per_cloud_mean":
                    mean(per_cloud) * 1000.0,

                "inference_ms_per_cloud_std":
                    sample_std(per_cloud) * 1000.0,

                "inference_peak_gpu_memory_mb_mean":
                    mean(memory),

                "inference_peak_gpu_memory_mb_std":
                    sample_std(memory),

                "inference_peak_gpu_memory_mb_max":
                    max(memory),
            }
        )

    return summary_rows


def main():

    args = parse_args()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for the formal "
            "inference benchmark."
        )

    device = torch.device(
        "cuda"
    )

    torch.manual_seed(
        EVAL_SEED
    )

    torch.cuda.manual_seed_all(
        EVAL_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configs = build_configurations(
        include_control=args.include_control,
    )

    print()
    print("=" * 90)
    print("Q5 INFERENCE BENCHMARK")
    print("=" * 90)

    print(
        "Configurations:",
        len(configs),
    )

    print(
        "Warmup:",
        args.warmup,
    )

    print(
        "Repeats:",
        args.repeats,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Include control:",
        args.include_control,
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    print("=" * 90)

    if args.dry_run:

        missing = 0

        for config in configs:

            checkpoint = (
                get_checkpoint_path(
                    config
                )
            )

            exists = (
                checkpoint.exists()
            )

            print(
                f"{config['species']:<8} "
                f"{config['method']:<15} "
                f"{config['target']:>6} "
                f"seed={config['seed']} "
                f"checkpoint={exists}"
            )

            if not exists:
                missing += 1

        print()
        print(
            "Missing checkpoints:",
            missing,
        )

        print(
            "DRY RUN COMPLETE"
        )

        return

    raw_rows = []

    total = len(configs)

    for index, config in enumerate(
        configs,
        start=1,
    ):

        checkpoint_path = (
            get_checkpoint_path(
                config
            )
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                checkpoint_path
            )

        print()
        print("=" * 90)

        print(
            f"BENCHMARK {index}/{total}"
        )

        print(
            f"Species: {config['species']}"
        )

        print(
            f"Method:  {config['method']}"
        )

        print(
            f"Target:  {config['target']}"
        )

        print(
            f"Seed:    {config['seed']}"
        )

        print("=" * 90)

        dataset = build_dataset(
            config
        )

        points = get_first_full_batch(
            dataset,
            batch_size=args.batch_size,
        )

        model = build_model(
            checkpoint_path,
            device,
        )

        result = benchmark_forward(
            model=model,
            points=points,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        row = {
            "species": config["species"],
            "method": config["method"],
            "target": config["target"],
            "seed": config["seed"],

            "warmup": args.warmup,
            "repeats": args.repeats,

            **result,
        }

        raw_rows.append(
            row
        )

        print(
            "Inference:",
            f"{result['forward_seconds_per_cloud'] * 1000:.3f}",
            "ms/cloud",
        )

        print(
            "Peak GPU:",
            f"{result['peak_gpu_memory_mb']:.2f}",
            "MB",
        )

        # --------------------------------------------------
        # Liberar GPU entre configuraciones.
        # --------------------------------------------------

        del model
        del points
        del dataset

        torch.cuda.empty_cache()

    summary_rows = build_summary(
        raw_rows
    )

    save_csv(
        RAW_OUTPUT,
        raw_rows,
    )

    save_csv(
        SUMMARY_OUTPUT,
        summary_rows,
    )

    print()
    print("=" * 90)
    print("INFERENCE BENCHMARK SUMMARY")
    print("=" * 90)

    print(
        "Raw configurations:",
        len(raw_rows),
    )

    print(
        "Summary groups:",
        len(summary_rows),
    )

    print(
        "Raw output:",
        RAW_OUTPUT.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Summary output:",
        SUMMARY_OUTPUT.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "INFERENCE BENCHMARK COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()

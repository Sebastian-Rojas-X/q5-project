"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Generate technical trade-off figures, including time versus point count.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
    / "results_tradeoff.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "figures"
    / "tradeoff"
)


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

SPECIES_ORDER = [
    "maize",
    "tomato",
]

METHOD_ORDER = [
    "random",
    "voxel",
    "fps",
    "control_48000",
]

METHOD_LABELS = {
    "random": "Random",
    "voxel": "Voxel",
    "fps": "FPS",
    "control_48000": "Practical high-resolution control",
}

METHOD_STYLES = {
    "random": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 6.5,
    },
    "voxel": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 6.5,
    },
    "fps": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 7.0,
    },
    "control_48000": {
        "color": "#111111",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 2.0,
        "markersize": 6.5,
    },
}

TARGET_ORDER = [
    1024,
    2048,
    4096,
    8192,
    16384,
    48000,
]


# =============================================================================
# UTILITIES
# =============================================================================

def configure_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 450,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def ensure_output_dir():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_columns(
    df,
    required,
):
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing columns in results_tradeoff.csv:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    required = [
        "species",
        "method",
        "target",
        "test_miou_mean",
        "chamfer_mean",
        "hausdorff_h99_mean",
        "sampling_ms_mean",
        "inference_ms_per_cloud_mean",
        "training_seconds_mean",
        "training_peak_gpu_memory_mb_mean",
    ]

    require_columns(
        df,
        required,
    )

    numeric_columns = [
        "target",
        "test_miou_mean",
        "test_miou_std",
        "chamfer_mean",
        "chamfer_std",
        "hausdorff_h99_mean",
        "hausdorff_h99_std",
        "sampling_ms_mean",
        "sampling_ms_std",
        "inference_ms_per_cloud_mean",
        "inference_ms_per_cloud_std",
        "training_seconds_mean",
        "training_seconds_std",
        "training_peak_gpu_memory_mb_mean",
        "training_peak_gpu_memory_mb_std",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def validate_data(df):
    if len(df) != 32:
        raise RuntimeError(
            f"Expected 32 trade-off rows, got {len(df)}"
        )

    factorial = df[
        df["method"] != "control_48000"
    ]

    controls = df[
        df["method"] == "control_48000"
    ]

    if len(factorial) != 30:
        raise RuntimeError(
            f"Expected 30 factorial rows, got {len(factorial)}"
        )

    if len(controls) != 2:
        raise RuntimeError(
            f"Expected 2 practical high-resolution "
            f"control rows, got {len(controls)}"
        )

    if factorial[
        "chamfer_mean"
    ].isna().any():
        raise RuntimeError(
            "Unexpected missing Chamfer values "
            "in factorial configurations."
        )

    if factorial[
        "hausdorff_h99_mean"
    ].isna().any():
        raise RuntimeError(
            "Unexpected missing H99 values "
            "in factorial configurations."
        )

    if controls[
        "chamfer_mean"
    ].notna().any():
        raise RuntimeError(
            "Chamfer must be empty for the practical "
            "high-resolution control."
        )

    if controls[
        "hausdorff_h99_mean"
    ].notna().any():
        raise RuntimeError(
            "H99 must be empty for the practical "
            "high-resolution control."
        )


def format_target_label(target):
    if target == 48000:
        return "48k"

    if target >= 1000:
        return f"{int(target / 1000)}k"

    return str(
        int(target)
    )


def pretty_species_name(species):
    return species.capitalize()


def get_species_df(
    df,
    species,
):
    sdf = df[
        df["species"] == species
    ].copy()

    sdf["method"] = pd.Categorical(
        sdf["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    sdf = sdf.sort_values(
        by=[
            "method",
            "target",
        ]
    ).reset_index(
        drop=True
    )

    return sdf


def save_figure(
    fig,
    output_path,
):
    fig.savefig(
        output_path,
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def maybe_add_errorbars(
    ax,
    x,
    y,
    yerr,
    color,
):
    if yerr is None:
        return

    if np.all(
        np.isnan(yerr)
    ):
        return

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor=color,
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.7,
        zorder=1,
    )


# =============================================================================
# PLOTS VS TARGET
# =============================================================================

def plot_metric_vs_target(
    sdf,
    species,
    y_col,
    y_std_col,
    ylabel,
    title_suffix,
    filename_suffix,
    log_y=False,
    allow_control=True,
):
    fig, ax = plt.subplots(
        figsize=(6.8, 4.8)
    )

    for method in METHOD_ORDER:
        if (
            not allow_control
            and method == "control_48000"
        ):
            continue

        mdf = sdf[
            sdf["method"] == method
        ].copy()

        if mdf.empty:
            continue

        if y_col not in mdf.columns:
            continue

        mdf = mdf.dropna(
            subset=[y_col]
        )

        if mdf.empty:
            continue

        x = mdf[
            "target"
        ].to_numpy(
            dtype=float
        )

        y = mdf[
            y_col
        ].to_numpy(
            dtype=float
        )

        yerr = None

        if (
            y_std_col
            and y_std_col in mdf.columns
        ):
            yerr = mdf[
                y_std_col
            ].to_numpy(
                dtype=float
            )

        style = METHOD_STYLES[
            method
        ]

        ax.plot(
            x,
            y,
            label=METHOD_LABELS[
                method
            ],
            color=style[
                "color"
            ],
            marker=style[
                "marker"
            ],
            linestyle=style[
                "linestyle"
            ],
            linewidth=style[
                "linewidth"
            ],
            markersize=style[
                "markersize"
            ],
            zorder=2,
        )

        maybe_add_errorbars(
            ax=ax,
            x=x,
            y=y,
            yerr=yerr,
            color=style[
                "color"
            ],
        )

    ax.set_title(
        f"{pretty_species_name(species)} — "
        f"{title_suffix}"
    )

    ax.set_xlabel(
        "Number of points"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticks(
        TARGET_ORDER
    )

    ax.set_xticklabels(
        [
            f"{target:,}"
            for target
            in TARGET_ORDER
        ]
    )

    if log_y:
        ax.set_yscale(
            "log"
        )

    ax.grid(
        True,
        which="major",
        axis="both",
    )

    ax.legend(
        ncol=1,
        loc="best",
    )

    output_path = (
        OUTPUT_DIR
        / f"{species}_{filename_suffix}.png"
    )

    save_figure(
        fig,
        output_path,
    )

    print(
        output_path.relative_to(
            PROJECT_ROOT
        )
    )


# =============================================================================
# TRADE-OFF SCATTERS
# =============================================================================

def plot_tradeoff_scatter(
    sdf,
    species,
    x_col,
    x_std_col,
    xlabel,
    title_suffix,
    filename_suffix,
    log_x=False,
):
    fig, ax = plt.subplots(
        figsize=(6.8, 4.8)
    )

    for method in METHOD_ORDER:
        mdf = sdf[
            sdf["method"] == method
        ].copy()

        if mdf.empty:
            continue

        mdf = mdf.dropna(
            subset=[
                x_col,
                "test_miou_mean",
                "target",
            ]
        )

        if mdf.empty:
            continue

        mdf = mdf.sort_values(
            "target"
        )

        x = mdf[
            x_col
        ].to_numpy(
            dtype=float
        )

        y = mdf[
            "test_miou_mean"
        ].to_numpy(
            dtype=float
        )

        sizes = []

        for target in mdf[
            "target"
        ]:
            target = int(
                target
            )

            if target == 48000:
                sizes.append(
                    95
                )

            elif target == 16384:
                sizes.append(
                    80
                )

            elif target == 8192:
                sizes.append(
                    72
                )

            elif target == 4096:
                sizes.append(
                    66
                )

            elif target == 2048:
                sizes.append(
                    60
                )

            else:
                sizes.append(
                    55
                )

        style = METHOD_STYLES[
            method
        ]

        ax.plot(
            x,
            y,
            color=style[
                "color"
            ],
            linestyle=style[
                "linestyle"
            ],
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )

        ax.scatter(
            x,
            y,
            s=sizes,
            color=style[
                "color"
            ],
            marker=style[
                "marker"
            ],
            edgecolors="white",
            linewidths=0.8,
            label=METHOD_LABELS[
                method
            ],
            zorder=3,
        )

        if (
            x_std_col
            and x_std_col in mdf.columns
        ):
            xerr = mdf[
                x_std_col
            ].to_numpy(
                dtype=float
            )

            if not np.all(
                np.isnan(xerr)
            ):
                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    fmt="none",
                    ecolor=style[
                        "color"
                    ],
                    elinewidth=0.9,
                    capsize=2.5,
                    alpha=0.6,
                    zorder=2,
                )

        for xi, yi, target in zip(
            x,
            y,
            mdf["target"],
        ):
            ax.annotate(
                format_target_label(
                    int(target)
                ),
                (
                    xi,
                    yi,
                ),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                alpha=0.9,
            )

    ax.set_title(
        f"{pretty_species_name(species)} — "
        f"{title_suffix}"
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Test mIoU"
    )

    if log_x:
        ax.set_xscale(
            "log"
        )

    ax.grid(
        True,
        which="major",
        axis="both",
    )

    ax.legend(
        ncol=1,
        loc="best",
    )

    output_path = (
        OUTPUT_DIR
        / f"{species}_{filename_suffix}.png"
    )

    save_figure(
        fig,
        output_path,
    )

    print(
        output_path.relative_to(
            PROJECT_ROOT
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    configure_matplotlib()
    ensure_output_dir()

    df = load_data()

    validate_data(
        df
    )

    print()
    print("=" * 90)
    print("Q5 FIGURE GENERATION")
    print("=" * 90)

    print(
        "Input :",
        INPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Output:",
        OUTPUT_DIR.relative_to(
            PROJECT_ROOT
        ),
    )

    print()

    generated = 0

    for species in SPECIES_ORDER:
        sdf = get_species_df(
            df,
            species,
        )

        # 1. mIoU vs target
        plot_metric_vs_target(
            sdf=sdf,
            species=species,
            y_col="test_miou_mean",
            y_std_col="test_miou_std",
            ylabel="Test mIoU",
            title_suffix=(
                "Segmentation performance vs number of points"
            ),
            filename_suffix="miou_vs_target",
            log_y=False,
            allow_control=True,
        )

        generated += 1

        # 2. Chamfer vs target
        plot_metric_vs_target(
            sdf=sdf,
            species=species,
            y_col="chamfer_mean",
            y_std_col="chamfer_std",
            ylabel="Chamfer distance",
            title_suffix=(
                "Mean geometric distortion vs number of points"
            ),
            filename_suffix="chamfer_vs_target",
            log_y=False,
            allow_control=False,
        )

        generated += 1

        # 3. Robust Hausdorff H99 vs target
        plot_metric_vs_target(
            sdf=sdf,
            species=species,
            y_col="hausdorff_h99_mean",
            y_std_col="hausdorff_h99_std",
            ylabel="Robust Hausdorff H99",
            title_suffix=(
                "Tail geometric distortion vs number of points"
            ),
            filename_suffix="hausdorff_h99_vs_target",
            log_y=False,
            allow_control=False,
        )

        generated += 1

        # 4. Sampling time vs target
        plot_metric_vs_target(
            sdf=sdf,
            species=species,
            y_col="sampling_ms_mean",
            y_std_col="sampling_ms_std",
            ylabel="Sampling time (ms)",
            title_suffix=(
                "Sampling cost vs number of points"
            ),
            filename_suffix="sampling_time_vs_target",
            log_y=True,
            allow_control=True,
        )

        generated += 1

        # 5. Inference time vs target
        plot_metric_vs_target(
            sdf=sdf,
            species=species,
            y_col="inference_ms_per_cloud_mean",
            y_std_col="inference_ms_per_cloud_std",
            ylabel="Inference time per cloud (ms)",
            title_suffix=(
                "Inference cost vs number of points"
            ),
            filename_suffix="inference_time_vs_target",
            log_y=False,
            allow_control=True,
        )

        generated += 1

        # 6. GPU memory vs target
        plot_metric_vs_target(
            sdf=sdf,
            species=species,
            y_col="training_peak_gpu_memory_mb_mean",
            y_std_col="training_peak_gpu_memory_mb_std",
            ylabel="Peak GPU memory (MB)",
            title_suffix=(
                "Training memory usage vs number of points"
            ),
            filename_suffix="gpu_memory_vs_target",
            log_y=False,
            allow_control=True,
        )

        generated += 1

        # 7. mIoU vs sampling time
        plot_tradeoff_scatter(
            sdf=sdf,
            species=species,
            x_col="sampling_ms_mean",
            x_std_col="sampling_ms_std",
            xlabel="Sampling time (ms)",
            title_suffix=(
                "Accuracy–sampling trade-off"
            ),
            filename_suffix="miou_vs_sampling_time",
            log_x=True,
        )

        generated += 1

        # 8. mIoU vs inference time
        plot_tradeoff_scatter(
            sdf=sdf,
            species=species,
            x_col="inference_ms_per_cloud_mean",
            x_std_col="inference_ms_per_cloud_std",
            xlabel="Inference time per cloud (ms)",
            title_suffix=(
                "Accuracy–inference trade-off"
            ),
            filename_suffix="miou_vs_inference_time",
            log_x=False,
        )

        generated += 1

    expected = (
        len(SPECIES_ORDER)
        * 8
    )

    if generated != expected:
        raise RuntimeError(
            f"Expected {expected} figures, "
            f"generated {generated}"
        )

    print()
    print("=" * 90)

    print(
        f"PNG generated: "
        f"{generated} / {expected}"
    )

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
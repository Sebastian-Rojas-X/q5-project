"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Generate the five final publication figures from frozen trade-off results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    / "tradeoff_publication"
)


# =============================================================================
# EXPERIMENT CONSTANTS
# =============================================================================

SPECIES_ORDER = [
    "maize",
    "tomato",
]

SPECIES_LABELS = {
    "maize": "Maize",
    "tomato": "Tomato",
}

FACTORIAL_METHODS = [
    "random",
    "voxel",
    "fps",
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
    "control_48000": "Control 48k",
}

TARGETS = [
    1024,
    2048,
    4096,
    8192,
    16384,
]

TARGET_LABELS = {
    1024: "1k",
    2048: "2k",
    4096: "4k",
    8192: "8k",
    16384: "16k",
    48000: "48k",
}

METHOD_STYLES = {
    "random": {
        "color": "#0072B2",
        "marker": "o",
    },
    "voxel": {
        "color": "#E69F00",
        "marker": "s",
    },
    "fps": {
        "color": "#009E73",
        "marker": "^",
    },
    "control_48000": {
        "color": "#303030",
        "marker": "D",
    },
}

DOT_OFFSETS = {
    "random": -0.18,
    "voxel": 0.00,
    "fps": 0.18,
}

PANEL_LABELS = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]


# =============================================================================
# MANUAL LABEL OFFSETS FOR FIGURES 3 AND 4
# Units are matplotlib "offset points": (dx, dy)
# Edit these if you want to move a specific label manually.
# =============================================================================

LABEL_OFFSETS = {
    "figure_3": {
        "maize": {
            "random": {
                1024: (-18, -2),
                2048: (-18, 6),
                4096: (-20, -10),
                8192: (-20, 6),
                16384: (-22, -10),
            },
            "voxel": {
                1024: (8, 4),
                2048: (10, -6),
                4096: (10, -8),
                8192: (10, -2),
                16384: (10, -10),
            },
            "fps": {
                1024: (8, 6),
                2048: (10, 6),
                4096: (10, -6),
                8192: (8, 6),
                16384: (8, 6),
            },
            "control_48000": {
                48000: (-40, 0),
            },
        },
        "tomato": {
            "random": {
                1024: (-18, -8),
                2048: (-12, 6), #2048: (-20, -2),
                4096: (8, -12),
                8192: (8, 6),
                16384: (-22, -12),
            },
            "voxel": {
                1024: (8, -10),
                2048: (-12, 6),
                4096: (8, -8),
                8192: (8, 6),
                16384: (8, -10),
            },
            "fps": {
                1024: (8, 6),
                2048: (8, 6),
                4096: (8, 6),
                8192: (8, 2),
                16384: (8, 0),
            },
            "control_48000": {
                48000: (-40, 0),
            },
        },
    },
    "figure_4": {
        "maize": {
            "random": {
                1024: (-12, -10),
                2048: (-18, -2),
                4096: (-25, 12),
                8192: (-18, 6),
                16384: (-18, -10),
            },
            "voxel": {
                1024: (10, 6),
                2048: (10, 5),
                4096: (10, -10),
                8192: (10, -10),
                16384: (10, -4),
            },
            "fps": {
                1024: (8, 6),
                2048: (8, 8),
                4096: (8, 0),
                8192: (8, 6),
                16384: (8, 6),
            },
            "control_48000": {
                48000: (10, 2),
            },
        },
        "tomato": {
            "random": {
                1024: (-12, -12),
                2048: (-17, 0),
                4096: (-20, 6),
                8192: (-18, 6),
                16384: (-18, -10),
            },
            "voxel": {
                1024: (8, -10),
                2048: (8, -10),
                4096: (8, 6),
                8192: (8, -10),
                16384: (8, -4),
            },
            "fps": {
                1024: (8, 6),
                2048: (8, 6),
                4096: (8, 6),
                8192: (8, 6),
                16384: (8, 0),
            },
            "control_48000": {
                48000: (10, 2),
            },
        },
    },
}


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

def configure_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 450,

        "font.family": "serif",
        "font.size": 9.5,

        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,

        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "grid.alpha": 0.16,
        "grid.linewidth": 0.6,

        "legend.frameon": False,
        "legend.fontsize": 8.5,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =============================================================================
# DATA
# =============================================================================

def ensure_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def ensure_output_dir():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_columns(df, required):
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns in results_tradeoff.csv: "
            + ", ".join(missing)
        )


def load_data():
    ensure_exists(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required = [
        "species",
        "method",
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

    require_columns(df, required)

    numeric_columns = [
        column
        for column in df.columns
        if (
            column == "target"
            or column.endswith("_mean")
            or column.endswith("_std")
        )
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def validate_data(df):
    if len(df) != 32:
        raise RuntimeError(
            f"Expected 32 rows, got {len(df)}"
        )

    for species in SPECIES_ORDER:

        control = df[
            (df["species"] == species)
            & (df["method"] == "control_48000")
        ]

        if len(control) != 1:
            raise RuntimeError(
                f"Expected one control row for {species}, "
                f"got {len(control)}"
            )

        if int(control.iloc[0]["target"]) != 48000:
            raise RuntimeError(
                f"Unexpected control target for {species}"
            )

        for method in FACTORIAL_METHODS:
            rows = df[
                (df["species"] == species)
                & (df["method"] == method)
            ]

            if len(rows) != 5:
                raise RuntimeError(
                    f"Expected 5 rows for {species}/{method}, "
                    f"got {len(rows)}"
                )

            observed_targets = set(
                rows["target"]
                .astype(int)
                .tolist()
            )

            if observed_targets != set(TARGETS):
                raise RuntimeError(
                    f"Unexpected targets for {species}/{method}: "
                    f"{sorted(observed_targets)}"
                )

    factorial = df[
        df["method"].isin(FACTORIAL_METHODS)
    ]

    if factorial["chamfer_mean"].isna().any():
        raise RuntimeError(
            "Missing Chamfer values in factorial rows."
        )

    if factorial["hausdorff_h99_mean"].isna().any():
        raise RuntimeError(
            "Missing H99 values in factorial rows."
        )

    controls = df[
        df["method"] == "control_48000"
    ]

    if controls["chamfer_mean"].notna().any():
        raise RuntimeError(
            "Control rows must not contain Chamfer."
        )

    if controls["hausdorff_h99_mean"].notna().any():
        raise RuntimeError(
            "Control rows must not contain H99."
        )


def species_subset(df, species):
    return (
        df[
            df["species"] == species
        ]
        .copy()
        .reset_index(drop=True)
    )


def factorial_subset(species_df, method):
    return (
        species_df[
            species_df["method"] == method
        ]
        .sort_values("target")
        .copy()
    )


def control_row(species_df):
    rows = species_df[
        species_df["method"] == "control_48000"
    ]

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one control row."
        )

    return rows.iloc[0]


# =============================================================================
# COMMON FIGURE HELPERS
# =============================================================================

def add_panel_label(ax, label):
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def configure_target_axis(ax):
    positions = np.arange(len(TARGETS))

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [TARGET_LABELS[target] for target in TARGETS]
    )

    ax.set_xlim(
        -0.45,
        len(TARGETS) - 0.55,
    )

    ax.set_xlabel("Input points")


def build_method_handles(
    include_control=True,
    control_as_reference=False,
):
    handles = []

    for method in FACTORIAL_METHODS:
        style = METHOD_STYLES[method]

        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.8,
                markersize=6,
                label=METHOD_LABELS[method],
            )
        )

    if include_control:
        style = METHOD_STYLES["control_48000"]

        if control_as_reference:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.4,
                    label=METHOD_LABELS["control_48000"],
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="None",
                    markersize=6,
                    label=METHOD_LABELS["control_48000"],
                )
            )

    return handles


def add_global_legend(
    fig,
    handles,
    ncol,
    y,
):
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        columnspacing=1.7,
        handletextpad=0.6,
    )


def add_control_reference(ax, mean, std):
    color = METHOD_STYLES["control_48000"]["color"]

    mean = float(mean)
    std = float(std)

    ax.axhspan(
        mean - std,
        mean + std,
        color=color,
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )

    ax.axhline(
        mean,
        color=color,
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )


def save_figure(fig, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    fig.savefig(
        png_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(png_path.relative_to(PROJECT_ROOT))
    print(pdf_path.relative_to(PROJECT_ROOT))

    return [png_path, pdf_path]


# =============================================================================
# LABEL HELPERS FOR FIGURE 3 / 4
# =============================================================================

def default_annotation_offset(method, target):
    target = int(target)

    if method == "random":
        mapping = {
            1024: (-16, -8),
            2048: (-16, 2),
            4096: (-16, -6),
            8192: (-16, 6),
            16384: (-16, -8),
        }
        return mapping[target]

    if method == "voxel":
        mapping = {
            1024: (8, 4),
            2048: (8, -4),
            4096: (8, -8),
            8192: (8, 4),
            16384: (8, -6),
        }
        return mapping[target]

    if method == "fps":
        mapping = {
            1024: (8, 6),
            2048: (8, 6),
            4096: (8, 0),
            8192: (8, 6),
            16384: (8, 6),
        }
        return mapping[target]

    return (8, 2)


def get_label_offset(
    figure_key,
    species,
    method,
    target,
):
    target = int(target)

    try:
        return LABEL_OFFSETS[figure_key][species][method][target]
    except KeyError:
        return default_annotation_offset(
            method,
            target,
        )


def annotate_point_label(
    ax,
    x,
    y,
    text,
    color,
    dx,
    dy,
):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.4,
        color=color,
        fontweight="semibold",
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )


# =============================================================================
# CATEGORICAL CURVES
# =============================================================================

def plot_performance_curve(ax, species_df):
    x = np.arange(len(TARGETS))

    for method in FACTORIAL_METHODS:
        rows = factorial_subset(species_df, method)

        mean = rows["test_miou_mean"].to_numpy(dtype=float)
        std = rows["test_miou_std"].to_numpy(dtype=float)

        style = METHOD_STYLES[method]

        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            elinewidth=0.8,
            capsize=2.2,
            alpha=0.95,
            zorder=3,
        )

    control = control_row(species_df)

    add_control_reference(
        ax,
        control["test_miou_mean"],
        control["test_miou_std"],
    )

    configure_target_axis(ax)
    ax.set_ylabel("Test mIoU ↑")
    ax.grid(axis="y")


def plot_geometric_curve(
    ax,
    species_df,
    mean_column,
    std_column,
    y_label,
):
    x = np.arange(len(TARGETS))

    for method in FACTORIAL_METHODS:
        rows = factorial_subset(species_df, method)

        mean = rows[mean_column].to_numpy(dtype=float)
        std = rows[std_column].to_numpy(dtype=float)

        style = METHOD_STYLES[method]

        lower = np.maximum(0.0, mean - std)
        upper = mean + std

        ax.fill_between(
            x,
            lower,
            upper,
            color=style["color"],
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.8,
            markersize=5.8,
            zorder=3,
        )

    configure_target_axis(ax)
    ax.set_ylabel(y_label)
    ax.grid(axis="y")


# =============================================================================
# TRADE-OFF SCATTER
# =============================================================================

def plot_accuracy_efficiency_scatter(
    ax,
    species_df,
    species,
    figure_key,
    x_column,
    x_label,
    log_x,
):
    for method in FACTORIAL_METHODS:
        rows = factorial_subset(species_df, method)

        x = rows[x_column].to_numpy(dtype=float)
        y = rows["test_miou_mean"].to_numpy(dtype=float)
        yerr = rows["test_miou_std"].to_numpy(dtype=float)

        style = METHOD_STYLES[method]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.75,
            capsize=2,
            alpha=0.45,
            zorder=1,
        )

        ax.scatter(
            x,
            y,
            s=54,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        for _, row in rows.iterrows():
            target = int(row["target"])

            dx, dy = get_label_offset(
                figure_key=figure_key,
                species=species,
                method=method,
                target=target,
            )

            annotate_point_label(
                ax=ax,
                x=row[x_column],
                y=row["test_miou_mean"],
                text=TARGET_LABELS[target],
                color=style["color"],
                dx=dx,
                dy=dy,
            )

    control = control_row(species_df)
    control_style = METHOD_STYLES["control_48000"]

    ax.errorbar(
        [control[x_column]],
        [control["test_miou_mean"]],
        yerr=[control["test_miou_std"]],
        fmt="none",
        ecolor=control_style["color"],
        elinewidth=0.8,
        capsize=2.2,
        alpha=0.55,
        zorder=1,
    )

    ax.scatter(
        [control[x_column]],
        [control["test_miou_mean"]],
        marker=control_style["marker"],
        color=control_style["color"],
        s=66,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    control_dx, control_dy = get_label_offset(
        figure_key=figure_key,
        species=species,
        method="control_48000",
        target=48000,
    )

    annotate_point_label(
        ax=ax,
        x=control[x_column],
        y=control["test_miou_mean"],
        text="Control",
        color=control_style["color"],
        dx=control_dx,
        dy=control_dy,
    )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Test mIoU ↑")
    ax.grid(alpha=0.16)


# =============================================================================
# RESOURCE DOT PLOTS
# =============================================================================

def plot_resource_dots(
    ax,
    species_df,
    mean_column,
    std_column,
    y_label,
    log_y=False,
):
    base_x = np.arange(len(TARGETS))

    for method in FACTORIAL_METHODS:
        rows = factorial_subset(species_df, method)

        x = base_x + DOT_OFFSETS[method]
        y = rows[mean_column].to_numpy(dtype=float)
        yerr = rows[std_column].to_numpy(dtype=float)

        style = METHOD_STYLES[method]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            markersize=5.5,
            elinewidth=0.75,
            capsize=2,
            alpha=0.9,
            linestyle="None",
            zorder=3,
        )

    control = control_row(species_df)

    add_control_reference(
        ax,
        control[mean_column],
        control[std_column],
    )

    configure_target_axis(ax)
    ax.set_ylabel(y_label)

    if log_y:
        ax.set_yscale("log")

    ax.grid(axis="y")


# =============================================================================
# FIGURE 1
# =============================================================================

def make_figure_1(df):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 3.8),
    )

    for index, species in enumerate(SPECIES_ORDER):
        ax = axes[index]
        sdf = species_subset(df, species)

        plot_performance_curve(ax, sdf)

        ax.set_title(SPECIES_LABELS[species])
        add_panel_label(ax, PANEL_LABELS[index])

    handles = build_method_handles(
        include_control=True,
        control_as_reference=True,
    )

    add_global_legend(
        fig,
        handles,
        ncol=4,
        y=1.02,
    )

    fig.subplots_adjust(
        top=0.78,
        bottom=0.16,
        left=0.08,
        right=0.98,
        wspace=0.20,
    )

    return save_figure(
        fig,
        "figure_1_performance_vs_points",
    )


# =============================================================================
# FIGURE 2
# =============================================================================

def make_figure_2(df):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.6, 7.0),
    )

    panels = [
        (
            "maize",
            "chamfer_mean",
            "chamfer_std",
            "Chamfer distance ↓",
        ),
        (
            "tomato",
            "chamfer_mean",
            "chamfer_std",
            "Chamfer distance ↓",
        ),
        (
            "maize",
            "hausdorff_h99_mean",
            "hausdorff_h99_std",
            "Robust Hausdorff H99 ↓",
        ),
        (
            "tomato",
            "hausdorff_h99_mean",
            "hausdorff_h99_std",
            "Robust Hausdorff H99 ↓",
        ),
    ]

    for index, panel in enumerate(panels):
        (
            species,
            mean_column,
            std_column,
            y_label,
        ) = panel

        ax = axes.flat[index]
        sdf = species_subset(df, species)

        plot_geometric_curve(
            ax=ax,
            species_df=sdf,
            mean_column=mean_column,
            std_column=std_column,
            y_label=y_label,
        )

        ax.set_title(SPECIES_LABELS[species])
        add_panel_label(ax, PANEL_LABELS[index])

    handles = build_method_handles(
        include_control=False
    )

    add_global_legend(
        fig,
        handles,
        ncol=3,
        y=1.01,
    )

    fig.subplots_adjust(
        top=0.89,
        bottom=0.09,
        left=0.09,
        right=0.98,
        wspace=0.22,
        hspace=0.33,
    )

    return save_figure(
        fig,
        "figure_2_geometry_vs_points",
    )


# =============================================================================
# FIGURE 3
# =============================================================================

def make_figure_3(df):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.0),
    )

    for index, species in enumerate(SPECIES_ORDER):
        ax = axes[index]
        sdf = species_subset(df, species)

        plot_accuracy_efficiency_scatter(
            ax=ax,
            species_df=sdf,
            species=species,
            figure_key="figure_3",
            x_column="inference_ms_per_cloud_mean",
            x_label="Inference time per cloud (ms) ↓",
            log_x=False,
        )

        ax.set_title(SPECIES_LABELS[species])
        add_panel_label(ax, PANEL_LABELS[index])

    handles = build_method_handles(
        include_control=True,
        control_as_reference=False,
    )

    add_global_legend(
        fig,
        handles,
        ncol=4,
        y=1.02,
    )

    fig.subplots_adjust(
        top=0.78,
        bottom=0.16,
        left=0.08,
        right=0.98,
        wspace=0.21,
    )

    return save_figure(
        fig,
        "figure_3_accuracy_vs_inference_time",
    )


# =============================================================================
# FIGURE 4
# =============================================================================

def make_figure_4(df):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.0),
    )

    for index, species in enumerate(SPECIES_ORDER):
        ax = axes[index]
        sdf = species_subset(df, species)

        plot_accuracy_efficiency_scatter(
            ax=ax,
            species_df=sdf,
            species=species,
            figure_key="figure_4",
            x_column="sampling_ms_mean",
            x_label="Sampling time (ms) ↓",
            log_x=True,
        )

        ax.set_title(SPECIES_LABELS[species])
        add_panel_label(ax, PANEL_LABELS[index])

    handles = build_method_handles(
        include_control=True,
        control_as_reference=False,
    )

    add_global_legend(
        fig,
        handles,
        ncol=4,
        y=1.02,
    )

    fig.subplots_adjust(
        top=0.78,
        bottom=0.16,
        left=0.08,
        right=0.98,
        wspace=0.21,
    )

    return save_figure(
        fig,
        "figure_4_accuracy_vs_sampling_time",
    )


# =============================================================================
# FIGURE 5
# =============================================================================

def make_figure_5(df):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.6, 7.0),
    )

    panels = [
        (
            "maize",
            "training_seconds_mean",
            "training_seconds_std",
            "Training time (s) ↓",
            False,
        ),
        (
            "tomato",
            "training_seconds_mean",
            "training_seconds_std",
            "Training time (s) ↓",
            False,
        ),
        (
            "maize",
            "training_peak_gpu_memory_mb_mean",
            "training_peak_gpu_memory_mb_std",
            "Peak GPU memory (MB) ↓",
            True,
        ),
        (
            "tomato",
            "training_peak_gpu_memory_mb_mean",
            "training_peak_gpu_memory_mb_std",
            "Peak GPU memory (MB) ↓",
            True,
        ),
    ]

    for index, panel in enumerate(panels):
        (
            species,
            mean_column,
            std_column,
            y_label,
            log_y,
        ) = panel

        ax = axes.flat[index]
        sdf = species_subset(df, species)

        plot_resource_dots(
            ax=ax,
            species_df=sdf,
            mean_column=mean_column,
            std_column=std_column,
            y_label=y_label,
            log_y=log_y,
        )

        ax.set_title(SPECIES_LABELS[species])
        add_panel_label(ax, PANEL_LABELS[index])

    handles = build_method_handles(
        include_control=True,
        control_as_reference=True,
    )

    add_global_legend(
        fig,
        handles,
        ncol=4,
        y=1.01,
    )

    fig.subplots_adjust(
        top=0.89,
        bottom=0.09,
        left=0.09,
        right=0.98,
        wspace=0.22,
        hspace=0.33,
    )

    return save_figure(
        fig,
        "figure_5_resource_profile",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    configure_matplotlib()
    ensure_output_dir()

    df = load_data()
    validate_data(df)

    print()
    print("=" * 90)
    print("Q5 PUBLICATION-READY FIGURE GENERATION")
    print("=" * 90)

    print(
        "Input :",
        INPUT_CSV.relative_to(PROJECT_ROOT),
    )

    print(
        "Output:",
        OUTPUT_DIR.relative_to(PROJECT_ROOT),
    )

    print()

    generated = []

    generated.extend(make_figure_1(df))
    generated.extend(make_figure_2(df))
    generated.extend(make_figure_3(df))
    generated.extend(make_figure_4(df))
    generated.extend(make_figure_5(df))

    expected_files = 10

    if len(generated) != expected_files:
        raise RuntimeError(
            f"Expected {expected_files} files, "
            f"got {len(generated)}"
        )

    print()
    print("=" * 90)
    print("Files generated:", len(generated), "/", expected_files)
    print("PUBLICATION-READY FIGURES COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
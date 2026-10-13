"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Build paper tables from frozen segmentation, geometry, cost, and statistics.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "paper_tables"
)

TRADEOFF_CSV = (
    ANALYSIS_DIR
    / "results_tradeoff.csv"
)

STATISTICAL_SUMMARY_CSV = (
    ANALYSIS_DIR
    / "statistical_summary.csv"
)

CONTROL_COMPARISONS_CSV = (
    ANALYSIS_DIR
    / "operating_points.csv"
)


# =============================================================================
# LABELS
# =============================================================================

SPECIES_LABELS = {
    "maize": "Maize",
    "tomato": "Tomato",
}

METHOD_LABELS = {
    "random": "Random",
    "voxel": "Voxel",
    "fps": "FPS",
    "control_48000": "Control práctico de alta resolución",
}


# =============================================================================
# HELPERS
# =============================================================================

def ensure_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )


def fmt_mean_std(
    mean,
    std,
    digits=4,
):
    if pd.isna(mean):
        return ""

    if pd.isna(std):
        return (
            f"{float(mean):.{digits}f}"
        )

    return (
        f"{float(mean):.{digits}f} "
        f"$\\pm$ "
        f"{float(std):.{digits}f}"
    )


def fmt_number(
    value,
    digits=4,
):
    if pd.isna(value):
        return ""

    return (
        f"{float(value):.{digits}f}"
    )


def fmt_signed(
    value,
    digits=4,
):
    if pd.isna(value):
        return ""

    return (
        f"{float(value):+.{digits}f}"
    )


def fmt_ms(value):
    if pd.isna(value):
        return ""

    value = float(value)

    if value < 1:
        return f"{value:.3f}"

    if value < 100:
        return f"{value:.2f}"

    return f"{value:.1f}"


def fmt_pct(value):
    if pd.isna(value):
        return ""

    return (
        f"{float(value):.1f}"
    )


def fmt_p(value):
    if pd.isna(value):
        return ""

    value = float(value)

    if value < 0.001:
        return f"{value:.2e}"

    return f"{value:.4f}"


def bool_from_value(value):
    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def method_label(method):
    return METHOD_LABELS.get(
        method,
        method,
    )


def species_label(species):
    return SPECIES_LABELS.get(
        species,
        species,
    )


def write_latex_table(
    df,
    path,
    caption,
    label,
    column_format=None,
):
    latex = df.to_latex(
        index=False,
        escape=False,
        na_rep="",
        column_format=column_format,
        caption=caption,
        label=label,
        position="t",
    )

    path.write_text(
        latex,
        encoding="utf-8",
    )


# =============================================================================
# TABLE 1
# SEGMENTATION PERFORMANCE
# =============================================================================

def build_table_1():
    ensure_exists(
        TRADEOFF_CSV
    )

    df = pd.read_csv(
        TRADEOFF_CSV
    )

    required = [
        "species",
        "method",
        "target",
        "test_miou_mean",
        "test_miou_std",
        "test_iou_stem_mean",
        "test_iou_stem_std",
        "test_iou_leaf_mean",
        "test_iou_leaf_std",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Table 1 missing columns: "
            + ", ".join(missing)
        )

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "Species":
                species_label(
                    row["species"]
                ),

            "Method":
                method_label(
                    row["method"]
                ),

            "Points":
                int(
                    row["target"]
                ),

            "mIoU":
                fmt_mean_std(
                    row[
                        "test_miou_mean"
                    ],
                    row[
                        "test_miou_std"
                    ],
                ),

            "IoU Stem":
                fmt_mean_std(
                    row[
                        "test_iou_stem_mean"
                    ],
                    row[
                        "test_iou_stem_std"
                    ],
                ),

            "IoU Leaf":
                fmt_mean_std(
                    row[
                        "test_iou_leaf_mean"
                    ],
                    row[
                        "test_iou_leaf_std"
                    ],
                ),

        })

    out = pd.DataFrame(
        rows
    )

    out = out.sort_values(
        [
            "Species",
            "Method",
            "Points",
        ]
    )

    csv_path = (
        OUTPUT_DIR
        / "table_1_segmentation_performance.csv"
    )

    tex_path = (
        OUTPUT_DIR
        / "table_1_segmentation_performance.tex"
    )

    out.to_csv(
        csv_path,
        index=False,
    )

    write_latex_table(
        df=out,
        path=tex_path,
        caption=(
            "Semantic segmentation performance on the Pheno4D "
            "test set. Values are reported as mean $\\pm$ "
            "standard deviation across five training seeds."
        ),
        label="tab:segmentation_performance",
        column_format="lllccc",
    )

    return out


# =============================================================================
# TABLE 2
# GEOMETRIC FIDELITY + COMPUTATIONAL COST
# =============================================================================

def build_table_2():
    ensure_exists(
        TRADEOFF_CSV
    )

    df = pd.read_csv(
        TRADEOFF_CSV
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
        "inference_peak_gpu_memory_mb_mean",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Table 2 missing columns: "
            + ", ".join(missing)
        )

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "Species":
                species_label(
                    row["species"]
                ),

            "Method":
                method_label(
                    row["method"]
                ),

            "Points":
                int(
                    row["target"]
                ),

            "mIoU":
                fmt_number(
                    row[
                        "test_miou_mean"
                    ],
                    4,
                ),

            "Chamfer":
                fmt_number(
                    row[
                        "chamfer_mean"
                    ],
                    4,
                ),

            "H99":
                fmt_number(
                    row[
                        "hausdorff_h99_mean"
                    ],
                    4,
                ),

            "Sampling (ms)":
                fmt_ms(
                    row[
                        "sampling_ms_mean"
                    ]
                ),

            "Inference (ms/cloud)":
                fmt_ms(
                    row[
                        "inference_ms_per_cloud_mean"
                    ]
                ),

            "Training (s)":
                fmt_number(
                    row[
                        "training_seconds_mean"
                    ],
                    1,
                ),

            "Train GPU (MB)":
                fmt_number(
                    row[
                        "training_peak_gpu_memory_mb_mean"
                    ],
                    1,
                ),

            "Infer GPU (MB)":
                fmt_number(
                    row[
                        "inference_peak_gpu_memory_mb_mean"
                    ],
                    1,
                ),
        })

    out = pd.DataFrame(
        rows
    )

    out = out.sort_values(
        [
            "Species",
            "Method",
            "Points",
        ]
    )

    csv_path = (
        OUTPUT_DIR
        / "table_2_cost_fidelity.csv"
    )

    tex_path = (
        OUTPUT_DIR
        / "table_2_cost_fidelity.tex"
    )

    out.to_csv(
        csv_path,
        index=False,
    )

    write_latex_table(
        df=out,
        path=tex_path,
        caption=(
            "Trade-off between segmentation performance, geometric "
            "fidelity, and computational cost. Chamfer summarizes "
            "mean geometric coverage error, whereas H99 is the "
            "99th percentile of original-to-reduced nearest-neighbor "
            "distances. Geometry metrics are not reported for the "
            "control práctico de alta resolución because the original "
            "post-ground cloud is retained as the geometric reference."
        ),
        label="tab:cost_fidelity",
        column_format="lllcccccccc",
    )

    return out


# =============================================================================
# TABLE 3
# SAMPLER STATISTICS
# =============================================================================

def build_table_3():
    ensure_exists(
        STATISTICAL_SUMMARY_CSV
    )

    df = pd.read_csv(
        STATISTICAL_SUMMARY_CSV
    )

    required = [
        "analysis_level",
        "species",
        "method",
        "comparator",
        "target",
        "test",
        "statistic",
        "p_value_raw",
        "p_value_adjusted",
        "effect_name",
        "effect_value",
        "significant",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Table 3 missing columns: "
            + ", ".join(missing)
        )

    sampler_df = df[
        df[
            "analysis_level"
        ].isin([
            "species_target",
            "global",
            "global_pairwise",
        ])
    ].copy()

    if len(sampler_df) != 14:
        raise RuntimeError(
            "Expected 14 sampler-statistics rows "
            f"(10 species-target + 1 global + 3 pairwise), "
            f"got {len(sampler_df)}"
        )

    rows = []

    for _, row in sampler_df.iterrows():
        level = row[
            "analysis_level"
        ]

        if level == "species_target":
            scope = (
                f"{species_label(row['species'])}, "
                f"{int(float(row['target']))} pts"
            )

            comparison = (
                "Random / Voxel / FPS"
            )

        elif level == "global":
            scope = "Global"

            comparison = (
                "Random / Voxel / FPS"
            )

        elif level == "global_pairwise":
            scope = "Global"

            comparison = (
                f"{method_label(row['method'])} vs "
                f"{method_label(row['comparator'])}"
            )

        else:
            raise RuntimeError(
                f"Unexpected analysis level: {level}"
            )

        raw_p = fmt_p(
            row[
                "p_value_raw"
            ]
        )

        adjusted_p = fmt_p(
            row[
                "p_value_adjusted"
            ]
        )

        if level == "global_pairwise":
            displayed_p = (
                adjusted_p
            )

            test_label = (
                "Wilcoxon + Holm"
            )

        else:
            displayed_p = (
                raw_p
            )

            test_label = (
                "Friedman"
            )

        if pd.isna(
            row[
                "effect_value"
            ]
        ):
            effect = ""

        else:
            effect_name = str(
                row[
                    "effect_name"
                ]
            )

            if effect_name == "kendalls_w":
                effect = (
                    "Kendall W = "
                    + fmt_number(
                        row[
                            "effect_value"
                        ],
                        3,
                    )
                )

            elif (
                effect_name
                == "rank_biserial_r"
            ):
                effect = (
                    "r_rb = "
                    + fmt_signed(
                        row[
                            "effect_value"
                        ],
                        3,
                    )
                )

            else:
                effect = (
                    f"{effect_name} = "
                    + fmt_number(
                        row[
                            "effect_value"
                        ],
                        3,
                    )
                )

        significant = bool_from_value(
            row[
                "significant"
            ]
        )

        rows.append({
            "Scope":
                scope,

            "Comparison":
                comparison,

            "Test":
                test_label,

            "Statistic":
                fmt_number(
                    row[
                        "statistic"
                    ],
                    3,
                ),

            "p-value":
                displayed_p,

            "Effect":
                effect,

            "Result":
                (
                    "Significant"
                    if significant
                    else
                    "Not significant"
                ),
        })

    out = pd.DataFrame(
        rows
    )

    csv_path = (
        OUTPUT_DIR
        / "table_3_statistics.csv"
    )

    tex_path = (
        OUTPUT_DIR
        / "table_3_statistics.tex"
    )

    out.to_csv(
        csv_path,
        index=False,
    )

    write_latex_table(
        df=out,
        path=tex_path,
        caption=(
            "Statistical comparison of the three external sampling "
            "strategies using test mIoU. Friedman tests evaluate the "
            "three samplers jointly, and global pairwise contrasts use "
            "paired Wilcoxon signed-rank tests with Holm correction. "
            "Kendall's W and rank-biserial correlation are reported "
            "as effect sizes."
        ),
        label="tab:statistics",
        column_format="lllcccc",
    )

    return out


# =============================================================================
# TABLE 4
# CONFIGURATIONS VS PRACTICAL HIGH-RESOLUTION CONTROL
# =============================================================================

def build_table_4():
    ensure_exists(
        CONTROL_COMPARISONS_CSV
    )

    df = pd.read_csv(
        CONTROL_COMPARISONS_CSV
    )

    required = [
        "species",
        "method",
        "target",
        "test_miou_mean",
        "control_miou_mean",
        "delta_miou_mean_config_minus_control",
        "n_seed_differences_positive",
        "n_seed_differences_negative",
        "n_seed_differences_zero",
        "p_value_raw_exact",
        "p_value_holm",
        "rank_biserial_r",
        "holm_significant_alpha_0_05",
        "sampling_ms_mean",
        "inference_ms_per_cloud_mean",
        "inference_saving_pct",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Table 4 missing columns: "
            + ", ".join(missing)
        )

    if len(df) != 30:
        raise RuntimeError(
            f"Expected 30 control comparisons, got {len(df)}"
        )

    rows = []

    for _, row in df.iterrows():
        significant = bool_from_value(
            row[
                "holm_significant_alpha_0_05"
            ]
        )

        signs = (
            f"{int(row['n_seed_differences_positive'])}/"
            f"{int(row['n_seed_differences_negative'])}/"
            f"{int(row['n_seed_differences_zero'])}"
        )

        rows.append({
            "Species":
                species_label(
                    row[
                        "species"
                    ]
                ),

            "Method":
                method_label(
                    row[
                        "method"
                    ]
                ),

            "Points":
                int(
                    row[
                        "target"
                    ]
                ),

            "mIoU":
                fmt_number(
                    row[
                        "test_miou_mean"
                    ],
                    4,
                ),

            "Control mIoU":
                fmt_number(
                    row[
                        "control_miou_mean"
                    ],
                    4,
                ),

            "$\\Delta$mIoU":
                fmt_signed(
                    row[
                        "delta_miou_mean_config_minus_control"
                    ],
                    4,
                ),

            "+/-/0 seeds":
                signs,

            "Exact p":
                fmt_p(
                    row[
                        "p_value_raw_exact"
                    ]
                ),

            "Holm p":
                fmt_p(
                    row[
                        "p_value_holm"
                    ]
                ),

            "$r_{rb}$":
                fmt_signed(
                    row[
                        "rank_biserial_r"
                    ],
                    3,
                ),

            "Sampling (ms)":
                fmt_ms(
                    row[
                        "sampling_ms_mean"
                    ]
                ),

            "Inference (ms)":
                fmt_ms(
                    row[
                        "inference_ms_per_cloud_mean"
                    ]
                ),

            "Inference saving (\\%)":
                fmt_pct(
                    row[
                        "inference_saving_pct"
                    ]
                ),

            "Significant":
                (
                    "Yes"
                    if significant
                    else "No"
                ),
        })

    out = pd.DataFrame(
        rows
    )

    out = out.sort_values(
        [
            "Species",
            "Method",
            "Points",
        ]
    )

    csv_path = (
        OUTPUT_DIR
        / "table_4_control_comparisons.csv"
    )

    tex_path = (
        OUTPUT_DIR
        / "table_4_control_comparisons.tex"
    )

    out.to_csv(
        csv_path,
        index=False,
    )

    write_latex_table(
        df=out,
        path=tex_path,
        caption=(
            "Paired comparison of each downsampling configuration "
            "against the control práctico de alta resolución. "
            "$\\Delta$mIoU is defined as configuration minus control. "
            "Exact two-sided Wilcoxon signed-rank tests use the five "
            "paired training seeds, with Holm correction over the "
            "15 comparisons within each species. The sign-count column "
            "reports positive/negative/zero paired differences. "
            "Non-significant results do not establish equivalence."
        ),
        label="tab:control_comparisons",
        column_format="lllccccccccccc",
    )

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print("BUILD Q5 PAPER TABLES")
    print("=" * 90)

    table_1 = build_table_1()

    print(
        "Table 1 rows:",
        len(table_1),
        "/ 32",
    )

    table_2 = build_table_2()

    print(
        "Table 2 rows:",
        len(table_2),
        "/ 32",
    )

    table_3 = build_table_3()

    print(
        "Table 3 rows:",
        len(table_3),
        "/ 14",
    )

    table_4 = build_table_4()

    print(
        "Table 4 rows:",
        len(table_4),
        "/ 30",
    )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "Tables generated:"
    )

    print(
        "  table_1_segmentation_performance.csv/.tex"
    )

    print(
        "  table_2_cost_fidelity.csv/.tex"
    )

    print(
        "  table_3_statistics.csv/.tex"
    )

    print(
        "  table_4_control_comparisons.csv/.tex"
    )

    print()
    print(
        "No statistical equivalence claim is made."
    )

    print(
        "No operating point is selected automatically."
    )

    print()
    print(
        "PAPER TABLES COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()

"""
Author: Sebastián Rojas
Email: srojas.hz@gmail.com
Year: 2026

Description:
Consolidate the frozen Friedman, Wilcoxon, Holm, and effect-size results.
"""

import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "analysis"
)

FRIEDMAN_BY_TARGET = (
    ANALYSIS_DIR
    / "friedman_test_miou.csv"
)

POSTHOC_BY_TARGET = (
    ANALYSIS_DIR
    / "posthoc_test_miou.csv"
)

FRIEDMAN_GLOBAL = (
    ANALYSIS_DIR
    / "friedman_global_test_miou.csv"
)

POSTHOC_GLOBAL = (
    ANALYSIS_DIR
    / "posthoc_global_test_miou.csv"
)

CONTROL_COMPARISONS = (
    ANALYSIS_DIR
    / "operating_points.csv"
)

OUTPUT_CSV = (
    ANALYSIS_DIR
    / "statistical_summary.csv"
)

OUTPUT_TXT = (
    ANALYSIS_DIR
    / "statistical_summary.txt"
)


ALPHA = 0.05

EXPECTED_SPECIES = {
    "maize",
    "tomato",
}

EXPECTED_TARGETS = {
    1024,
    2048,
    4096,
    8192,
    16384,
}

EXPECTED_METHODS = {
    "random",
    "voxel",
    "fps",
}

EXPECTED_CONTROL_COMPARISONS = 30
EXPECTED_CONTROL_COMPARISONS_PER_SPECIES = 15


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def as_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def optional_float(value):
    if (
        value is None
        or str(value).strip() == ""
    ):
        return ""

    return float(value)


def empty_summary_row():
    return {
        "analysis_level": "",
        "species": "",
        "method": "",
        "comparator": "",
        "target": "",
        "metric": "",
        "test": "",
        "n_blocks": "",
        "statistic": "",
        "p_value_raw": "",
        "p_value_adjusted": "",
        "correction": "",
        "alpha": "",
        "significant": "",
        "effect_name": "",
        "effect_value": "",
        "mean_difference_config_minus_reference": "",
        "median_difference_config_minus_reference": "",
        "observed_direction": "",
        "n_positive": "",
        "n_negative": "",
        "n_zero": "",
        "interpretation": "",
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_friedman_by_target(rows):
    if len(rows) != 10:
        raise RuntimeError(
            f"Expected 10 species-target Friedman rows, "
            f"got {len(rows)}"
        )

    seen = set()

    for row in rows:
        key = (
            row["species"],
            int(row["target"]),
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate Friedman species-target row: {key}"
            )

        seen.add(key)

        if row["species"] not in EXPECTED_SPECIES:
            raise RuntimeError(
                f"Unexpected species in Friedman: "
                f"{row['species']}"
            )

        if int(row["target"]) not in EXPECTED_TARGETS:
            raise RuntimeError(
                f"Unexpected target in Friedman: "
                f"{row['target']}"
            )


def validate_global_friedman(rows):
    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one global Friedman row."
        )


def validate_global_posthoc(rows):
    if len(rows) != 3:
        raise RuntimeError(
            f"Expected 3 global pairwise rows, "
            f"got {len(rows)}"
        )


def validate_control_comparisons(rows):
    if len(rows) != EXPECTED_CONTROL_COMPARISONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROL_COMPARISONS} "
            f"control comparisons, got {len(rows)}"
        )

    counts = Counter(
        row["species"]
        for row in rows
    )

    for species in EXPECTED_SPECIES:
        if (
            counts[species]
            != EXPECTED_CONTROL_COMPARISONS_PER_SPECIES
        ):
            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_CONTROL_COMPARISONS_PER_SPECIES} "
                f"control comparisons for {species}, "
                f"got {counts[species]}"
            )

    seen = set()

    for row in rows:
        species = row["species"]
        method = row["method"]
        target = int(row["target"])

        key = (
            species,
            method,
            target,
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate control comparison: {key}"
            )

        seen.add(key)

        if species not in EXPECTED_SPECIES:
            raise RuntimeError(
                f"Unexpected species: {species}"
            )

        if method not in EXPECTED_METHODS:
            raise RuntimeError(
                f"Unexpected method: {method}"
            )

        if target not in EXPECTED_TARGETS:
            raise RuntimeError(
                f"Unexpected target: {target}"
            )

        if row["control_method"] != "control_48000":
            raise RuntimeError(
                "Unexpected control identifier for "
                f"{key}: {row['control_method']}"
            )

        if int(row["control_target"]) != 48000:
            raise RuntimeError(
                "Unexpected control target for "
                f"{key}: {row['control_target']}"
            )


# ============================================================
# BUILD SUMMARY ROWS
# ============================================================

def build_species_target_rows(
    friedman_rows,
    posthoc_rows,
):
    output = []

    for row in friedman_rows:
        species = row["species"]
        target = int(row["target"])

        relevant_posthoc = [
            item
            for item in posthoc_rows
            if (
                item["species"] == species
                and int(item["target"]) == target
            )
        ]

        significant_pairwise = sum(
            as_bool(
                item["significant_holm"]
            )
            for item in relevant_posthoc
        )

        statistic = float(
            row["statistic"]
        )

        n_blocks = int(
            row["n_blocks"]
        )

        # Friedman:
        # W = chi2 / [n * (k - 1)]
        # k = 3 samplers
        kendalls_w = (
            statistic
            / (
                n_blocks
                * 2
            )
        )

        significant = as_bool(
            row["significant"]
        )

        summary = empty_summary_row()

        summary.update({
            "analysis_level":
                "species_target",

            "species":
                species,

            "method":
                "all_samplers",

            "comparator":
                "",

            "target":
                target,

            "metric":
                "test_miou",

            "test":
                "friedman",

            "n_blocks":
                n_blocks,

            "statistic":
                statistic,

            "p_value_raw":
                float(
                    row["p_value"]
                ),

            "p_value_adjusted":
                "",

            "correction":
                "none",

            "alpha":
                float(
                    row["alpha"]
                ),

            "significant":
                significant,

            "effect_name":
                "kendalls_w",

            "effect_value":
                kendalls_w,

            "interpretation":
                (
                    "Omnibus difference detected among samplers; "
                    "no pairwise contrast survived Holm correction."
                    if (
                        significant
                        and relevant_posthoc
                        and significant_pairwise == 0
                    )
                    else
                    (
                        "No statistically significant omnibus "
                        "difference detected among samplers."
                        if not significant
                        else
                        "Statistically significant omnibus "
                        "difference detected among samplers."
                    )
                ),
        })

        output.append(
            summary
        )

    return output


def build_global_friedman_row(
    global_rows,
    global_posthoc_rows,
):
    row = global_rows[0]

    significant_pairs = sum(
        as_bool(
            item["significant_holm"]
        )
        for item in global_posthoc_rows
    )

    summary = empty_summary_row()

    summary.update({
        "analysis_level":
            "global",

        "species":
            "all",

        "method":
            "all_samplers",

        "comparator":
            "",

        "target":
            "all",

        "metric":
            "test_miou",

        "test":
            "friedman",

        "n_blocks":
            int(
                row["n_blocks"]
            ),

        "statistic":
            float(
                row["statistic"]
            ),

        "p_value_raw":
            float(
                row["p_value"]
            ),

        "p_value_adjusted":
            "",

        "correction":
            "none",

        "alpha":
            float(
                row["alpha"]
            ),

        "significant":
            as_bool(
                row["significant"]
            ),

        "effect_name":
            "kendalls_w",

        "effect_value":
            float(
                row["kendalls_w"]
            ),

        "interpretation":
            (
                "A statistically significant global sampler "
                "effect was detected; all three global "
                "pairwise contrasts survived Holm correction."
                if significant_pairs == 3
                else
                "A statistically significant global sampler "
                "effect was detected."
            ),
    })

    return summary


def build_global_pairwise_rows(
    global_posthoc_rows,
):
    output = []

    for row in global_posthoc_rows:
        method_a = row["method_a"]
        method_b = row["method_b"]

        difference = float(
            row[
                "mean_difference_a_minus_b"
            ]
        )

        if difference > 0:
            direction = (
                f"{method_a}_higher"
            )
        elif difference < 0:
            direction = (
                f"{method_b}_higher"
            )
        else:
            direction = (
                "no_mean_difference"
            )

        summary = empty_summary_row()

        summary.update({
            "analysis_level":
                "global_pairwise",

            "species":
                "all",

            "method":
                method_a,

            "comparator":
                method_b,

            "target":
                "all",

            "metric":
                "test_miou",

            "test":
                "wilcoxon_signed_rank",

            "n_blocks":
                int(
                    row["n_blocks"]
                ),

            "statistic":
                float(
                    row[
                        "wilcoxon_statistic"
                    ]
                ),

            "p_value_raw":
                float(
                    row["p_value_raw"]
                ),

            "p_value_adjusted":
                float(
                    row["p_value_holm"]
                ),

            "correction":
                "holm",

            "alpha":
                ALPHA,

            "significant":
                as_bool(
                    row[
                        "significant_holm"
                    ]
                ),

            "effect_name":
                "rank_biserial_r",

            "effect_value":
                float(
                    row[
                        "rank_biserial_r"
                    ]
                ),

            "mean_difference_config_minus_reference":
                difference,

            "observed_direction":
                direction,

            "interpretation":
                (
                    "Pairwise difference remained statistically "
                    "significant after Holm correction."
                    if as_bool(
                        row[
                            "significant_holm"
                        ]
                    )
                    else
                    "Pairwise difference did not remain "
                    "statistically significant after Holm "
                    "correction."
                ),
        })

        output.append(
            summary
        )

    return output


def build_control_comparison_rows(
    control_rows,
):
    output = []

    for row in control_rows:
        adjusted_p = float(
            row["p_value_holm"]
        )

        significant = as_bool(
            row[
                "holm_significant_alpha_0_05"
            ]
        )

        delta_mean = float(
            row[
                "delta_miou_mean_config_minus_control"
            ]
        )

        if significant:
            if delta_mean < 0:
                interpretation = (
                    "A statistically significant mIoU loss "
                    "relative to the practical high-resolution "
                    "control was detected after Holm correction."
                )

            elif delta_mean > 0:
                interpretation = (
                    "A statistically significant mIoU improvement "
                    "relative to the practical high-resolution "
                    "control was detected after Holm correction."
                )

            else:
                interpretation = (
                    "A statistically significant paired difference "
                    "was detected, with zero mean difference."
                )

        else:
            interpretation = (
                "No statistically significant difference was "
                "detected relative to the practical "
                "high-resolution control after Holm correction. "
                "This does not demonstrate equivalence."
            )

        summary = empty_summary_row()

        summary.update({
            "analysis_level":
                "control_comparison",

            "species":
                row["species"],

            "method":
                row["method"],

            "comparator":
                "control_48000",

            "target":
                int(
                    row["target"]
                ),

            "metric":
                "test_miou",

            "test":
                "wilcoxon_signed_rank_exact",

            "n_blocks":
                int(
                    row["n_pairs"]
                ),

            "statistic":
                float(
                    row[
                        "wilcoxon_statistic"
                    ]
                ),

            "p_value_raw":
                float(
                    row[
                        "p_value_raw_exact"
                    ]
                ),

            "p_value_adjusted":
                adjusted_p,

            "correction":
                "holm_within_species_15_comparisons",

            "alpha":
                ALPHA,

            "significant":
                significant,

            "effect_name":
                "matched_pairs_rank_biserial_r",

            "effect_value":
                float(
                    row[
                        "rank_biserial_r"
                    ]
                ),

            "mean_difference_config_minus_reference":
                delta_mean,

            "median_difference_config_minus_reference":
                float(
                    row[
                        "delta_miou_median_config_minus_control"
                    ]
                ),

            "observed_direction":
                row[
                    "observed_direction"
                ],

            "n_positive":
                int(
                    row[
                        "n_seed_differences_positive"
                    ]
                ),

            "n_negative":
                int(
                    row[
                        "n_seed_differences_negative"
                    ]
                ),

            "n_zero":
                int(
                    row[
                        "n_seed_differences_zero"
                    ]
                ),

            "interpretation":
                interpretation,
        })

        output.append(
            summary
        )

    return output


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(rows):
    if not rows:
        raise RuntimeError(
            "No statistical summary rows generated."
        )

    with OUTPUT_CSV.open(
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


# ============================================================
# SAVE HUMAN-READABLE TXT
# ============================================================

def save_txt(
    friedman_rows,
    posthoc_rows,
    global_rows,
    global_posthoc_rows,
    control_rows,
):
    lines = []

    lines.append(
        "Q5 STATISTICAL SUMMARY"
    )

    lines.append(
        "=" * 88
    )

    lines.append("")

    # --------------------------------------------------------
    # 1. FRIEDMAN BY SPECIES / TARGET
    # --------------------------------------------------------

    lines.append(
        "1. FRIEDMAN BY SPECIES AND TARGET"
    )

    lines.append(
        "-" * 88
    )

    significant_groups = 0

    for row in friedman_rows:
        statistic = float(
            row["statistic"]
        )

        n_blocks = int(
            row["n_blocks"]
        )

        kendalls_w = (
            statistic
            / (
                n_blocks
                * 2
            )
        )

        significant = as_bool(
            row["significant"]
        )

        if significant:
            significant_groups += 1

        lines.append(
            (
                f"{row['species']:<8} "
                f"target={int(row['target']):>5} | "
                f"chi2={statistic:.4f} | "
                f"p={float(row['p_value']):.6f} | "
                f"W={kendalls_w:.3f} | "
                f"significant={significant}"
            )
        )

    lines.append("")

    lines.append(
        f"Significant omnibus groups: "
        f"{significant_groups}/10"
    )

    # --------------------------------------------------------
    # 2. PER-TARGET POST-HOC
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "2. POST-HOC IN SIGNIFICANT SPECIES-TARGET GROUPS"
    )

    lines.append(
        "-" * 88
    )

    significant_posthoc = 0

    for row in posthoc_rows:
        if as_bool(
            row["significant_holm"]
        ):
            significant_posthoc += 1

    lines.append(
        "Pairwise contrasts surviving Holm correction: "
        f"{significant_posthoc}"
    )

    if significant_posthoc == 0:
        lines.append(
            "No pairwise species-target contrast survived "
            "Holm correction."
        )

    # --------------------------------------------------------
    # 3. GLOBAL FRIEDMAN
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "3. GLOBAL FRIEDMAN"
    )

    lines.append(
        "-" * 88
    )

    global_row = global_rows[0]

    lines.append(
        (
            f"chi2="
            f"{float(global_row['statistic']):.4f} | "
            f"p="
            f"{float(global_row['p_value']):.10f} | "
            f"Kendall W="
            f"{float(global_row['kendalls_w']):.4f} | "
            f"significant="
            f"{global_row['significant']}"
        )
    )

    # --------------------------------------------------------
    # 4. GLOBAL POST-HOC
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "4. GLOBAL POST-HOC"
    )

    lines.append(
        "-" * 88
    )

    for row in global_posthoc_rows:
        delta = float(
            row[
                "mean_difference_a_minus_b"
            ]
        )

        lines.append(
            (
                f"{row['method_a']} vs "
                f"{row['method_b']} | "
                f"raw p="
                f"{float(row['p_value_raw']):.10f} | "
                f"Holm p="
                f"{float(row['p_value_holm']):.10f} | "
                f"r_rb="
                f"{float(row['rank_biserial_r']):+.3f} | "
                f"delta(a-b)="
                f"{delta:+.4f} | "
                f"significant="
                f"{row['significant_holm']}"
            )
        )

    # --------------------------------------------------------
    # 5. CONTROL COMPARISONS
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "5. CONFIGURATIONS VS PRACTICAL HIGH-RESOLUTION CONTROL"
    )

    lines.append(
        "-" * 88
    )

    lines.append(
        "Test: paired two-sided exact Wilcoxon signed-rank."
    )

    lines.append(
        "Correction: Holm across 15 comparisons separately "
        "within each species."
    )

    lines.append(
        "Effect size: matched-pairs rank-biserial correlation."
    )

    lines.append(
        "Difference convention: configuration - control."
    )

    lines.append(
        "IMPORTANT: non-significant does not imply equivalence."
    )

    lines.append(
        "With n=5 non-zero untied pairs, the minimum possible "
        "two-sided exact Wilcoxon p-value is 0.0625."
    )

    for species in sorted(
        EXPECTED_SPECIES
    ):
        lines.append("")
        lines.append(
            species.upper()
        )

        species_rows = [
            row
            for row in control_rows
            if row["species"] == species
        ]

        species_rows = sorted(
            species_rows,
            key=lambda row: (
                row["method"],
                int(row["target"]),
            ),
        )

        significant_count = 0
        observed_improvements = 0
        observed_losses = 0

        for row in species_rows:
            delta = float(
                row[
                    "delta_miou_mean_config_minus_control"
                ]
            )

            if delta > 0:
                observed_improvements += 1

            elif delta < 0:
                observed_losses += 1

            if as_bool(
                row[
                    "holm_significant_alpha_0_05"
                ]
            ):
                significant_count += 1

            lines.append(
                (
                    f"  {row['method']:<6} "
                    f"{int(row['target']):>5} | "
                    f"delta="
                    f"{delta:+.4f} | "
                    f"p_exact="
                    f"{float(row['p_value_raw_exact']):.4f} | "
                    f"p_Holm="
                    f"{float(row['p_value_holm']):.4f} | "
                    f"r_rb="
                    f"{float(row['rank_biserial_r']):+.3f} | "
                    f"signs="
                    f"{row['n_seed_differences_positive']}/"
                    f"{row['n_seed_differences_negative']}/"
                    f"{row['n_seed_differences_zero']}"
                )
            )

        lines.append(
            (
                f"  Observed mean improvements: "
                f"{observed_improvements}/15"
            )
        )

        lines.append(
            (
                f"  Observed mean losses: "
                f"{observed_losses}/15"
            )
        )

        lines.append(
            (
                f"  Holm-significant comparisons: "
                f"{significant_count}/15"
            )
        )

    # --------------------------------------------------------
    # 6. INTERPRETATION
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "6. INTERPRETATION"
    )

    lines.append(
        "-" * 88
    )

    lines.append(
        (
            "The global Friedman analysis detects a sampler effect "
            "on test mIoU across the full factorial benchmark."
        )
    )

    lines.append(
        (
            "At the species-target level, only a subset of omnibus "
            "tests are significant and no corresponding pairwise "
            "contrast survives Holm correction."
        )
    )

    lines.append(
        (
            "The comparisons against the practical high-resolution "
            "control are strongly resolution-limited by n=5 paired "
            "seeds. Therefore, lack of statistical significance must "
            "not be interpreted as statistical equivalence."
        )
    )

    lines.append(
        (
            "Observed delta mIoU and rank-biserial effect sizes should "
            "be interpreted jointly with computational cost; "
            "no automatic operating point "
            "is selected from non-significance alone."
        )
    )

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    friedman_rows = load_csv(
        FRIEDMAN_BY_TARGET
    )

    posthoc_rows = load_csv(
        POSTHOC_BY_TARGET
    )

    global_rows = load_csv(
        FRIEDMAN_GLOBAL
    )

    global_posthoc_rows = load_csv(
        POSTHOC_GLOBAL
    )

    control_rows = load_csv(
        CONTROL_COMPARISONS
    )

    validate_friedman_by_target(
        friedman_rows
    )

    validate_global_friedman(
        global_rows
    )

    validate_global_posthoc(
        global_posthoc_rows
    )

    validate_control_comparisons(
        control_rows
    )

    output_rows = []

    output_rows.extend(
        build_species_target_rows(
            friedman_rows,
            posthoc_rows,
        )
    )

    output_rows.append(
        build_global_friedman_row(
            global_rows,
            global_posthoc_rows,
        )
    )

    output_rows.extend(
        build_global_pairwise_rows(
            global_posthoc_rows
        )
    )

    output_rows.extend(
        build_control_comparison_rows(
            control_rows
        )
    )

    expected_output_rows = (
        10
        + 1
        + 3
        + 30
    )

    if (
        len(output_rows)
        != expected_output_rows
    ):
        raise RuntimeError(
            f"Expected {expected_output_rows} "
            f"summary rows, got "
            f"{len(output_rows)}"
        )

    save_csv(
        output_rows
    )

    save_txt(
        friedman_rows,
        posthoc_rows,
        global_rows,
        global_posthoc_rows,
        control_rows,
    )

    control_significant = sum(
        as_bool(
            row[
                "holm_significant_alpha_0_05"
            ]
        )
        for row in control_rows
    )

    print()
    print("=" * 90)
    print("Q5 STATISTICAL SUMMARY")
    print("=" * 90)

    print(
        "Species-target Friedman rows:",
        len(friedman_rows),
        "/ 10",
    )

    print(
        "Global Friedman rows:",
        len(global_rows),
        "/ 1",
    )

    print(
        "Global post-hoc rows:",
        len(global_posthoc_rows),
        "/ 3",
    )

    print(
        "Control-comparison rows:",
        len(control_rows),
        "/ 30",
    )

    print(
        "Control comparisons significant after Holm:",
        control_significant,
        "/ 30",
    )

    print(
        "Summary rows:",
        len(output_rows),
        "/ 44",
    )

    print(
        "CSV:",
        OUTPUT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "TXT:",
        OUTPUT_TXT.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "No operating point selected automatically."
    )

    print(
        "Non-significance is not interpreted as equivalence."
    )

    print()
    print(
        "STATISTICAL SUMMARY COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
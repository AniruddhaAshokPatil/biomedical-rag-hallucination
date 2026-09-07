import csv
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


EVAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
)


INTEGRATED_PATH = (
    EVAL_DIR
    / "integrated_final_results.json"
)


DECISION_STATS_PATH = (
    EVAL_DIR
    / "final_test_decision_statistics.json"
)


HALL_STATS_PATH = (
    EVAL_DIR
    / "final_test_hallucination_statistics.json"
)


RETRIEVAL_STATS_PATH = (
    EVAL_DIR
    / "final_test_retrieval_statistics.json"
)


ERROR_PATH = (
    EVAL_DIR
    / "exploratory_error_analysis.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dissertation_tables"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 2. HELPERS

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def write_csv(
    path,
    fieldnames,
    rows
):

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def pct(value):

    if value is None:

        return "-"


    return (
        f"{value * 100:.2f}%"
    )


def pp(value):

    if value is None:

        return "-"


    return (
        f"{value * 100:+.2f} pp"
    )


def f4(value):

    if value is None:

        return "-"


    return (
        f"{value:.4f}"
    )


def format_p(value):

    if value is None:

        return "-"


    value = float(
        value
    )


    if value < 0.001:

        return (
            f"{value:.3e}"
        )


    return (
        f"{value:.6f}"
    )


def pretty_system_name(
    system_name
):

    names = {

        "baseline_llm":
            "Baseline LLM",

        "basic_rag":
            "Basic RAG",

        "advanced_rag":
            "Advanced RAG",

        "gold_context_control":
            "Gold-context control"
    }


    return names.get(
        system_name,
        system_name
    )


# 3. LOAD RESULTS

print("=" * 78)

print(
    "SECTION 48 - PREPARE DISSERTATION RESULT TABLES"
)

print("=" * 78)


integrated = load_json(
    INTEGRATED_PATH
)


decision_stats = load_json(
    DECISION_STATS_PATH
)


hall_stats = load_json(
    HALL_STATS_PATH
)


retrieval_stats = load_json(
    RETRIEVAL_STATS_PATH
)


error_analysis = load_json(
    ERROR_PATH
)


systems = (
    integrated[
        "systems"
    ]
)


SYSTEM_ORDER = [

    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


# 4. TABLE 1  PRIMARY SYSTEM PERFORMANCE

print(
    "\n[1] PRIMARY SYSTEM PERFORMANCE TABLE"
)


table1 = []


for system in SYSTEM_ORDER:

    row = (
        systems[
            system
        ]
    )


    table_row = {

        "System":
            row[
                "display_name"
            ],

        "Decision accuracy":
            pct(
                row[
                    "decision_accuracy"
                ]
            ),

        "Macro-F1":
            f4(
                row[
                    "decision_macro_f1"
                ]
            ),

        "V3 hallucination rate":
            pct(
                row[
                    "v3_hallucination_rate"
                ]
            ),

        "Groundedness":
            pct(
                row[
                    "v3_groundedness"
                ]
            ),

        "Contradiction rate":
            pct(
                row[
                    "v3_contradiction_rate"
                ]
            ),

        "Human-calibrated hallucination":
            pct(
                row[
                    "human_calibrated_hallucination_rate"
                ]
            ),

        "Calibration 95% CI":
            (
                f"{pct(row['human_calibrated_ci_low'])}"
                f"–"
                f"{pct(row['human_calibrated_ci_high'])}"
            )
    }


    table1.append(
        table_row
    )


    print(
        table_row
    )


write_csv(
    OUTPUT_DIR
    / "table_1_primary_system_performance.csv",
    list(
        table1[
            0
        ].keys()
    ),
    table1
)


# 5. TABLE 2  RETRIEVAL PERFORMANCE


print(
    "\n[2] RETRIEVAL PERFORMANCE TABLE"
)


table2 = []


for system in [

    "basic_rag",
    "advanced_rag"

]:

    row = (
        systems[
            system
        ]
    )


    table_row = {

        "System":
            row[
                "display_name"
            ],

        "Source Hit@1":
            pct(
                row[
                    "source_hit_at_1"
                ]
            ),

        "Source Hit@3":
            pct(
                row[
                    "source_hit_at_3"
                ]
            ),

        "Source Hit@5":
            pct(
                row[
                    "source_hit_at_5"
                ]
            ),

        "Source Hit@10":
            pct(
                row[
                    "source_hit_at_10"
                ]
            ),

        "MRR@10":
            f4(
                row[
                    "retrieval_mrr_at_10"
                ]
            ),

        "Gold-context Recall@1":
            pct(
                row[
                    "gold_context_recall_at_1"
                ]
            ),

        "Gold-context Recall@3":
            pct(
                row[
                    "gold_context_recall_at_3"
                ]
            ),

        "Gold-context Recall@5":
            pct(
                row[
                    "gold_context_recall_at_5"
                ]
            ),

        "Gold-context Recall@10":
            pct(
                row[
                    "gold_context_recall_at_10"
                ]
            )
    }


    table2.append(
        table_row
    )


    print(
        table_row
    )


write_csv(
    OUTPUT_DIR
    / "table_2_retrieval_performance.csv",
    list(
        table2[
            0
        ].keys()
    ),
    table2
)


# 6. TABLE 3  DECISION STATISTICAL COMPARISONS

print(
    "\n[3] DECISION STATISTICAL COMPARISONS"
)


table3 = []


decision_comparisons = (
    decision_stats[
        "comparisons"
    ]
)


if not isinstance(
    decision_comparisons,
    list
):

    raise TypeError(
        "Expected decision comparisons "
        "to be stored as a list."
    )


for record in decision_comparisons:

    system_a = (
        record[
            "system_a"
        ]
    )


    system_b = (
        record[
            "system_b"
        ]
    )


    accuracy = (
        record[
            "accuracy"
        ]
    )


    macro_f1 = (
        record[
            "macro_f1"
        ]
    )


    mcnemar = (
        record[
            "mcnemar"
        ]
    )


    accuracy_ci = (
        accuracy[
            "bootstrap_95_ci"
        ]
    )


    macro_ci = (
        macro_f1[
            "bootstrap_95_ci"
        ]
    )


    table_row = {

        "Comparison":
            (
                f"{pretty_system_name(system_b)} "
                f"vs {pretty_system_name(system_a)}"
            ),

        "Accuracy A":
            pct(
                accuracy[
                    "system_a"
                ]
            ),

        "Accuracy B":
            pct(
                accuracy[
                    "system_b"
                ]
            ),

        "Accuracy difference B-A":
            pp(
                accuracy[
                    "difference"
                ]
            ),

        "Accuracy 95% CI":
            (
                f"{pp(accuracy_ci[0])} to "
                f"{pp(accuracy_ci[1])}"
            ),

        "Macro-F1 A":
            f4(
                macro_f1[
                    "system_a"
                ]
            ),

        "Macro-F1 B":
            f4(
                macro_f1[
                    "system_b"
                ]
            ),

        "Macro-F1 difference B-A":
            (
                f"{macro_f1['difference']:+.4f}"
            ),

        "Macro-F1 95% CI":
            (
                f"{macro_ci[0]:+.4f} to "
                f"{macro_ci[1]:+.4f}"
            ),

        "McNemar raw p":
            format_p(
                mcnemar[
                    "exact_p_raw"
                ]
            ),

        "McNemar Holm p":
            format_p(
                mcnemar[
                    "exact_p_holm"
                ]
            ),

        "Paired odds ratio B/A":
            (
                f"{record['paired_odds_ratio_b_over_a']:.4f}"
            )
    }


    table3.append(
        table_row
    )


    print(
        table_row
    )


write_csv(
    OUTPUT_DIR
    / "table_3_decision_statistical_comparisons.csv",
    list(
        table3[
            0
        ].keys()
    ),
    table3
)


# 7. TABLE 4  HALLUCINATION STATISTICAL COMPARISONS

print(
    "\n[4] HALLUCINATION STATISTICAL COMPARISONS"
)


table4 = []


hall_comparisons = (
    hall_stats[
        "comparisons"
    ]
)


if not isinstance(
    hall_comparisons,
    list
):

    raise TypeError(
        "Expected hallucination comparisons "
        "to be stored as a list."
    )


for record in hall_comparisons:

    system_a = (
        record[
            "system_a"
        ]
    )


    system_b = (
        record[
            "system_b"
        ]
    )


    primary = (
        record[
            "primary_micro_claim_hallucination"
        ]
    )


    answer_level = (
        record[
            "paired_answer_level"
        ]
    )


    zero_claim = (
        record[
            "zero_claim_analysis"
        ]
    )


    ci = (
        primary[
            "question_cluster_bootstrap_95_ci"
        ]
    )


    answer_ci = (
        answer_level[
            "bootstrap_mean_difference_95_ci"
        ]
    )


    table_row = {

        "Comparison":
            (
                f"{pretty_system_name(system_b)} "
                f"vs {pretty_system_name(system_a)}"
            ),

        "Micro hallucination A":
            pct(
                primary[
                    "system_a"
                ]
            ),

        "Micro hallucination B":
            pct(
                primary[
                    "system_b"
                ]
            ),

        "Absolute difference B-A":
            pp(
                primary[
                    "absolute_difference"
                ]
            ),

        "Cluster bootstrap 95% CI":
            (
                f"{pp(ci[0])} to "
                f"{pp(ci[1])}"
            ),

        "Relative reduction B vs A":
            pct(
                primary[
                    "relative_reduction_b_vs_a"
                ]
            ),

        "Answer-level N":
            answer_level[
                "n_pairs_with_scored_claims"
            ],

        "Answer-level mean difference":
            (
                f"{answer_level['mean_difference']:+.4f}"
            ),

        "Answer-level 95% CI":
            (
                f"{answer_ci[0]:+.4f} to "
                f"{answer_ci[1]:+.4f}"
            ),

        "Wilcoxon raw p":
            format_p(
                answer_level[
                    "wilcoxon_p_raw"
                ]
            ),

        "Wilcoxon Holm p":
            format_p(
                answer_level[
                    "wilcoxon_p_holm"
                ]
            ),

        "Zero-claim McNemar raw p":
            format_p(
                zero_claim[
                    "mcnemar_p_raw"
                ]
            ),

        "Zero-claim McNemar Holm p":
            format_p(
                zero_claim[
                    "mcnemar_p_holm"
                ]
            )
    }


    table4.append(
        table_row
    )


    print(
        table_row
    )


write_csv(
    OUTPUT_DIR
    / "table_4_hallucination_statistical_comparisons.csv",
    list(
        table4[
            0
        ].keys()
    ),
    table4
)


# 8. TABLE 5 RETRIEVAL STATISTICAL COMPARISONS

print(
    "\n[5] RETRIEVAL STATISTICAL COMPARISONS"
)


table5 = []


source_hit_results = (
    retrieval_stats[
        "source_hit_analysis"
    ][
        "results"
    ]
)


for record in source_hit_results:

    k = (
        record[
            "k"
        ]
    )


    table_row = {

        "Metric":
            f"Source Hit@{k}",

        "Basic":
            pct(
                record[
                    "basic_rate"
                ]
            ),

        "Advanced":
            pct(
                record[
                    "advanced_rate"
                ]
            ),

        "Advanced - Basic":
            pp(
                record[
                    "advanced_minus_basic"
                ]
            ),

        "95% CI":
            "-",

        "Raw p":
            format_p(
                record[
                    "mcnemar_p_raw"
                ]
            ),

        "Holm-adjusted p":
            format_p(
                record[
                    "mcnemar_p_holm"
                ]
            )
    }


    table5.append(
        table_row
    )


    print(
        table_row
    )


continuous_results = (
    retrieval_stats[
        "continuous_metric_analysis"
    ][
        "results"
    ]
)


for record in continuous_results:

    ci = (
        record[
            "paired_bootstrap_95_ci"
        ]
    )


    metric_name = (
        record[
            "metric"
        ]
    )


    if metric_name.startswith(
        "gold_recall"
    ):

        basic_display = pct(
            record[
                "basic_mean"
            ]
        )


        advanced_display = pct(
            record[
                "advanced_mean"
            ]
        )


        difference_display = pp(
            record[
                "advanced_minus_basic"
            ]
        )


        ci_display = (
            f"{pp(ci[0])} to "
            f"{pp(ci[1])}"
        )


    else:

        basic_display = f4(
            record[
                "basic_mean"
            ]
        )


        advanced_display = f4(
            record[
                "advanced_mean"
            ]
        )


        difference_display = (
            f"{record['advanced_minus_basic']:+.4f}"
        )


        ci_display = (
            f"{ci[0]:+.4f} to "
            f"{ci[1]:+.4f}"
        )


    table_row = {

        "Metric":
            metric_name,

        "Basic":
            basic_display,

        "Advanced":
            advanced_display,

        "Advanced - Basic":
            difference_display,

        "95% CI":
            ci_display,

        "Raw p":
            format_p(
                record[
                    "wilcoxon_p_raw"
                ]
            ),

        "Holm-adjusted p":
            format_p(
                record[
                    "wilcoxon_p_holm"
                ]
            )
    }


    table5.append(
        table_row
    )


    print(
        table_row
    )


write_csv(
    OUTPUT_DIR
    / "table_5_retrieval_statistical_comparisons.csv",
    list(
        table5[
            0
        ].keys()
    ),
    table5
)


# 9. TABLE 6  EXPLORATORY ERROR ANALYSIS

print(
    "\n[6] EXPLORATORY ERROR TABLE"
)


rag_errors = (
    error_analysis[
        "rag_error_summary"
    ]
)


table6 = []


for system in [

    "basic_rag",
    "advanced_rag"

]:

    record = (
        rag_errors[
            system
        ]
    )


    table_row = {

        "System":
            pretty_system_name(
                system
            ),

        "Retrieval misses @5":
            record[
                "retrieval_misses_at_5"
            ],

        "Decision errors":
            record[
                "decision_errors"
            ],

        "Decision errors despite source hit":
            record[
                "decision_errors_with_source_hit_at_5"
            ],

        "% decision errors despite source hit":
            pct(
                record[
                    "proportion_of_decision_errors_despite_source_hit"
                ]
            ),

        "Answers with hallucinated claim":
            record[
                "answers_with_at_least_one_hallucinated_claim"
            ],

        "Hallucination answers despite source hit":
            record[
                "hallucination_answers_with_source_hit_at_5"
            ],

        "Gold Recall@5 | correct":
            f4(
                record[
                    "mean_gold_context_recall_at_5_when_decision_correct"
                ]
            ),

        "Gold Recall@5 | wrong":
            f4(
                record[
                    "mean_gold_context_recall_at_5_when_decision_wrong"
                ]
            )
    }


    table6.append(
        table_row
    )


    print(
        table_row
    )


write_csv(
    OUTPUT_DIR
    / "table_6_exploratory_error_analysis.csv",
    list(
        table6[
            0
        ].keys()
    ),
    table6
)

# 10. CREATE MARKDOWN SUMMARY

print(
    "\n[7] CREATING MARKDOWN SUMMARY"
)


markdown_path = (
    OUTPUT_DIR
    / "final_results_tables.md"
)


lines = []


lines.append(
    "# Final Test Results Tables\n\n"
)


# TABLE 1

lines.append(
    "## Table 1. Primary system performance\n\n"
)


lines.append(
    "| System | Accuracy | Macro-F1 | "
    "V3 hallucination | Groundedness | "
    "Contradiction | Human-calibrated hallucination |\n"
)


lines.append(
    "|---|---:|---:|---:|---:|---:|---:|\n"
)


for system in SYSTEM_ORDER:

    row = (
        systems[
            system
        ]
    )


    lines.append(
        "| "
        f"{row['display_name']} | "
        f"{pct(row['decision_accuracy'])} | "
        f"{row['decision_macro_f1']:.4f} | "
        f"{pct(row['v3_hallucination_rate'])} | "
        f"{pct(row['v3_groundedness'])} | "
        f"{pct(row['v3_contradiction_rate'])} | "
        f"{pct(row['human_calibrated_hallucination_rate'])} "
        "|\n"
    )


# TABLE 2

lines.append(
    "\n## Table 2. Retrieval performance\n\n"
)


lines.append(
    "| System | Hit@1 | Hit@3 | Hit@5 | Hit@10 | "
    "MRR@10 | Gold Recall@5 |\n"
)


lines.append(
    "|---|---:|---:|---:|---:|---:|---:|\n"
)


for system in [

    "basic_rag",
    "advanced_rag"

]:

    row = (
        systems[
            system
        ]
    )


    lines.append(
        "| "
        f"{row['display_name']} | "
        f"{pct(row['source_hit_at_1'])} | "
        f"{pct(row['source_hit_at_3'])} | "
        f"{pct(row['source_hit_at_5'])} | "
        f"{pct(row['source_hit_at_10'])} | "
        f"{row['retrieval_mrr_at_10']:.4f} | "
        f"{pct(row['gold_context_recall_at_5'])} "
        "|\n"
    )


# KEY CONFIRMATORY FINDINGS


lines.append(
    "\n## Key confirmatory findings\n\n"
)


lines.append(
    "- Basic RAG increased decision accuracy from "
    "32.20% to 62.20% relative to the standalone "
    "LLM and reduced the frozen V3 micro-claim "
    "hallucination rate from 61.08% to 8.49%.\n"
)


lines.append(
    "- Advanced RAG achieved 60.60% decision accuracy "
    "and a 7.85% V3 hallucination rate.\n"
)


lines.append(
    "- Advanced RAG did not show a statistically "
    "detectable decision-accuracy advantage over "
    "Basic RAG: difference -1.60 percentage points, "
    "95% CI -5.40 to +2.40 percentage points.\n"
)


lines.append(
    "- Advanced RAG did not show a statistically "
    "detectable hallucination-rate advantage over "
    "Basic RAG: difference -0.64 percentage points, "
    "95% cluster-bootstrap CI -2.33 to +1.08 "
    "percentage points.\n"
)


lines.append(
    "- At the generation cutoff, Source Hit@5 was "
    "99.00% for both Basic and Advanced RAG.\n"
)


# EXPLORATORY SUMMARY

lines.append(
    "\n## Exploratory residual-error summary\n\n"
)


for system in [

    "basic_rag",
    "advanced_rag"

]:

    record = (
        rag_errors[
            system
        ]
    )


    display = pretty_system_name(
        system
    )


    lines.append(
        f"- **{display}:** "
        f"{record['decision_errors']} decision errors; "
        f"{record['decision_errors_with_source_hit_at_5']} "
        "occurred despite Source Hit@5. "
        f"Mean Gold-context Recall@5 was "
        f"{record['mean_gold_context_recall_at_5_when_decision_correct']:.4f} "
        "for correct decisions and "
        f"{record['mean_gold_context_recall_at_5_when_decision_wrong']:.4f} "
        "for incorrect decisions.\n"
    )


lines.append(
    "\n*The residual-error analysis is exploratory "
    "and should not be presented as a pre-specified "
    "confirmatory endpoint.*\n"
)


with markdown_path.open(
    "w",
    encoding="utf-8"
) as file:

    file.writelines(
        lines
    )


print(
    markdown_path
)


# 11. FINAL VALIDATION

print(
    "\n[8] VALIDATING TABLE COMPLETENESS"
)


decision_missing = sum(

    1

    for row
    in table3

    if (
        row[
            "McNemar raw p"
        ]
        ==
        "-"
        or
        row[
            "McNemar Holm p"
        ]
        ==
        "-"
    )
)


hall_missing = sum(

    1

    for row
    in table4

    if (
        row[
            "Absolute difference B-A"
        ]
        ==
        "-"
        or
        row[
            "Relative reduction B vs A"
        ]
        ==
        "-"
    )
)


print(
    f"Decision rows missing p-values : "
    f"{decision_missing}"
)


print(
    f"Hallucination rows missing effects: "
    f"{hall_missing}"
)


if decision_missing != 0:

    raise RuntimeError(
        "Decision table still contains "
        "missing statistical p-values."
    )


if hall_missing != 0:

    raise RuntimeError(
        "Hallucination table still contains "
        "missing primary effect sizes."
    )


print(
    "PASS: Statistical dissertation "
    "tables are complete."
)


# 12. FILES SAVED

print(
    "\n[9] FILES SAVED"
)


for path in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        path
    )


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 48 COMPLETE - "
    "DISSERTATION TABLES PREPARED"
)


print(
    "=" * 78
)
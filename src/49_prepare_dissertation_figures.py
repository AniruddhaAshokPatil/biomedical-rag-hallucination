import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


ERROR_PATH = (
    EVAL_DIR
    / "exploratory_error_analysis.json"
)


FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "dissertation_figures"
)


FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 2. SETTINGS

SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


SYSTEM_LABELS = {
    "baseline_llm":
        "Baseline LLM",

    "basic_rag":
        "Basic RAG",

    "advanced_rag":
        "Advanced RAG",

    "gold_context_control":
        "Gold-context"
}


DPI = 300


# 3. HELPERS

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


def save_figure(
    figure,
    base_name
):

    png_path = (
        FIGURE_DIR
        / f"{base_name}.png"
    )


    pdf_path = (
        FIGURE_DIR
        / f"{base_name}.pdf"
    )


    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )


    figure.savefig(
        pdf_path,
        bbox_inches="tight"
    )


    plt.close(
        figure
    )


    print(
        f"  {png_path}"
    )


    print(
        f"  {pdf_path}"
    )


def add_bar_labels(
    axis,
    bars,
    percentage=False
):

    for bar in bars:

        value = (
            bar.get_height()
        )


        if percentage:

            label = (
                f"{value * 100:.1f}%"
            )

        else:

            label = (
                f"{value:.3f}"
            )


        axis.annotate(
            label,
            xy=(
                bar.get_x()
                +
                bar.get_width()
                /
                2,
                value
            ),
            xytext=(
                0,
                4
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )


# 4. LOAD FINAL RESULTS

print("=" * 78)

print(
    "SECTION 49 - PREPARE DISSERTATION FIGURES"
)

print("=" * 78)


print(
    """
These figures visualize existing frozen results only.

No new metric, statistical test, model judgment,
or system comparison is introduced.
"""
)


integrated = load_json(
    INTEGRATED_PATH
)


error_analysis = load_json(
    ERROR_PATH
)


systems = (
    integrated[
        "systems"
    ]
)


print(
    "[1] FINAL RESULT ARTIFACTS LOADED"
)


print(
    f"Systems : "
    f"{len(systems)}"
)


print(
    f"Figure output directory:\n"
    f"{FIGURE_DIR}"
)


# 5. FIGURE 1
# DECISION ACCURACY AND MACRO F1


print(
    "\n[2] FIGURE 1 - DECISION PERFORMANCE"
)


labels = [

    SYSTEM_LABELS[
        system
    ]

    for system
    in SYSTEM_ORDER
]


accuracy = np.asarray(

    [
        systems[
            system
        ][
            "decision_accuracy"
        ]

        for system
        in SYSTEM_ORDER
    ],

    dtype=float
)


macro_f1 = np.asarray(

    [
        systems[
            system
        ][
            "decision_macro_f1"
        ]

        for system
        in SYSTEM_ORDER
    ],

    dtype=float
)


x = np.arange(
    len(
        labels
    )
)


width = 0.36


figure = plt.figure(
    figsize=(
        9,
        5.5
    )
)


axis = figure.add_axes(
    [
        0.10,
        0.17,
        0.86,
        0.76
    ]
)


accuracy_bars = axis.bar(
    x
    -
    width
    /
    2,
    accuracy,
    width,
    label="Decision accuracy"
)


f1_bars = axis.bar(
    x
    +
    width
    /
    2,
    macro_f1,
    width,
    label="Macro-F1"
)


axis.set_ylabel(
    "Score"
)


axis.set_ylim(
    0,
    0.82
)


axis.set_xticks(
    x
)


axis.set_xticklabels(
    labels,
    rotation=15,
    ha="right"
)


axis.set_title(
    "Final test decision performance"
)


axis.legend(
    frameon=False
)


axis.grid(
    axis="y",
    alpha=0.25
)


add_bar_labels(
    axis,
    accuracy_bars,
    percentage=True
)


add_bar_labels(
    axis,
    f1_bars,
    percentage=False
)


save_figure(
    figure,
    "figure_1_decision_performance"
)


# 6. FIGURE 2
# PRIMARY V3 HALLUCINATION RATE


print(
    "\n[3] FIGURE 2 - PRIMARY V3 HALLUCINATION RATE"
)


v3_rates = np.asarray(

    [
        systems[
            system
        ][
            "v3_hallucination_rate"
        ]

        for system
        in SYSTEM_ORDER
    ],

    dtype=float
)


figure = plt.figure(
    figsize=(
        8.5,
        5.5
    )
)


axis = figure.add_axes(
    [
        0.11,
        0.17,
        0.85,
        0.76
    ]
)


bars = axis.bar(
    x,
    v3_rates
)


axis.set_ylabel(
    "Micro-claim hallucination rate"
)


axis.set_ylim(
    0,
    0.70
)


axis.set_xticks(
    x
)


axis.set_xticklabels(
    labels,
    rotation=15,
    ha="right"
)


axis.set_title(
    "Frozen V3 hallucination rate on the final test set"
)


axis.grid(
    axis="y",
    alpha=0.25
)


add_bar_labels(
    axis,
    bars,
    percentage=True
)


save_figure(
    figure,
    "figure_2_v3_hallucination_rate"
)

# 7. FIGURE 3
# HUMAN-CALIBRATED SENSITIVITY ANALYSIS


print(
    "\n[4] FIGURE 3 - HUMAN-CALIBRATED SENSITIVITY"
)


calibrated_rates = np.asarray(

    [
        systems[
            system
        ][
            "human_calibrated_hallucination_rate"
        ]

        for system
        in SYSTEM_ORDER
    ],

    dtype=float
)


ci_low = np.asarray(

    [
        systems[
            system
        ][
            "human_calibrated_ci_low"
        ]

        for system
        in SYSTEM_ORDER
    ],

    dtype=float
)


ci_high = np.asarray(

    [
        systems[
            system
        ][
            "human_calibrated_ci_high"
        ]

        for system
        in SYSTEM_ORDER
    ],

    dtype=float
)


lower_error = (
    calibrated_rates
    -
    ci_low
)


upper_error = (
    ci_high
    -
    calibrated_rates
)


figure = plt.figure(
    figsize=(
        8.5,
        5.5
    )
)


axis = figure.add_axes(
    [
        0.11,
        0.17,
        0.85,
        0.76
    ]
)


axis.errorbar(
    x,
    calibrated_rates,
    yerr=np.vstack(
        [
            lower_error,
            upper_error
        ]
    ),
    fmt="o",
    capsize=5
)


axis.set_ylabel(
    "Human-calibrated hallucination rate"
)


axis.set_ylim(
    0,
    0.52
)


axis.set_xticks(
    x
)


axis.set_xticklabels(
    labels,
    rotation=15,
    ha="right"
)


axis.set_title(
    "Human-calibrated hallucination sensitivity analysis"
)


axis.grid(
    axis="y",
    alpha=0.25
)


for index, rate in enumerate(
    calibrated_rates
):

    axis.annotate(
        f"{rate * 100:.2f}%",
        xy=(
            index,
            rate
        ),
        xytext=(
            7,
            0
        ),
        textcoords="offset points",
        va="center",
        fontsize=9
    )


axis.text(
    0.99,
    0.02,
    "Sensitivity analysis; error bars show 95% bootstrap CIs",
    transform=axis.transAxes,
    ha="right",
    va="bottom",
    fontsize=8
)


save_figure(
    figure,
    "figure_3_human_calibrated_hallucination"
)


# 8. FIGURE 4
# SOURCE HIT@K


print(
    "\n[5] FIGURE 4 - SOURCE HIT@K"
)


k_values = np.asarray(
    [
        1,
        3,
        5,
        10
    ]
)


basic_hit = np.asarray(

    [
        systems[
            "basic_rag"
        ][
            "source_hit_at_1"
        ],

        systems[
            "basic_rag"
        ][
            "source_hit_at_3"
        ],

        systems[
            "basic_rag"
        ][
            "source_hit_at_5"
        ],

        systems[
            "basic_rag"
        ][
            "source_hit_at_10"
        ]
    ],

    dtype=float
)


advanced_hit = np.asarray(

    [
        systems[
            "advanced_rag"
        ][
            "source_hit_at_1"
        ],

        systems[
            "advanced_rag"
        ][
            "source_hit_at_3"
        ],

        systems[
            "advanced_rag"
        ][
            "source_hit_at_5"
        ],

        systems[
            "advanced_rag"
        ][
            "source_hit_at_10"
        ]
    ],

    dtype=float
)


figure = plt.figure(
    figsize=(
        8,
        5.2
    )
)


axis = figure.add_axes(
    [
        0.11,
        0.14,
        0.85,
        0.79
    ]
)


axis.plot(
    k_values,
    basic_hit,
    marker="o",
    label="Basic RAG"
)


axis.plot(
    k_values,
    advanced_hit,
    marker="o",
    label="Advanced RAG"
)


axis.set_xlabel(
    "Retrieval depth (k)"
)


axis.set_ylabel(
    "Source Hit@k"
)


axis.set_ylim(
    0.95,
    1.002
)


axis.set_xticks(
    k_values
)


axis.set_title(
    "Correct-source retrieval on the final test set"
)


axis.legend(
    frameon=False
)


axis.grid(
    alpha=0.25
)


save_figure(
    figure,
    "figure_4_source_hit_at_k"
)


# 9. FIGURE 5
# GOLD-CONTEXT RECALL@K


print(
    "\n[6] FIGURE 5 - GOLD-CONTEXT RECALL@K"
)


basic_recall = np.asarray(

    [
        systems[
            "basic_rag"
        ][
            "gold_context_recall_at_1"
        ],

        systems[
            "basic_rag"
        ][
            "gold_context_recall_at_3"
        ],

        systems[
            "basic_rag"
        ][
            "gold_context_recall_at_5"
        ],

        systems[
            "basic_rag"
        ][
            "gold_context_recall_at_10"
        ]
    ],

    dtype=float
)


advanced_recall = np.asarray(

    [
        systems[
            "advanced_rag"
        ][
            "gold_context_recall_at_1"
        ],

        systems[
            "advanced_rag"
        ][
            "gold_context_recall_at_3"
        ],

        systems[
            "advanced_rag"
        ][
            "gold_context_recall_at_5"
        ],

        systems[
            "advanced_rag"
        ][
            "gold_context_recall_at_10"
        ]
    ],

    dtype=float
)


figure = plt.figure(
    figsize=(
        8,
        5.2
    )
)


axis = figure.add_axes(
    [
        0.11,
        0.14,
        0.85,
        0.79
    ]
)


axis.plot(
    k_values,
    basic_recall,
    marker="o",
    label="Basic RAG"
)


axis.plot(
    k_values,
    advanced_recall,
    marker="o",
    label="Advanced RAG"
)


axis.set_xlabel(
    "Retrieval depth (k)"
)


axis.set_ylabel(
    "Gold-context Recall@k"
)


axis.set_ylim(
    0.25,
    0.90
)


axis.set_xticks(
    k_values
)


axis.set_title(
    "Gold-context evidence coverage"
)


axis.legend(
    frameon=False
)


axis.grid(
    alpha=0.25
)


save_figure(
    figure,
    "figure_5_gold_context_recall_at_k"
)


# 10. FIGURE 6
# EXPLORATORY RESIDUAL ERROR PROFILE

print(
    "\n[7] FIGURE 6 - EXPLORATORY RESIDUAL ERRORS"
)


rag_errors = (
    error_analysis[
        "rag_error_summary"
    ]
)


error_labels = [
    "Retrieval\nmisses @5",
    "Decision errors\ndespite hit",
    "Hallucination answers\ndespite hit"
]


basic_error_values = np.asarray(
    [
        rag_errors[
            "basic_rag"
        ][
            "retrieval_misses_at_5"
        ],

        rag_errors[
            "basic_rag"
        ][
            "decision_errors_with_source_hit_at_5"
        ],

        rag_errors[
            "basic_rag"
        ][
            "hallucination_answers_with_source_hit_at_5"
        ]
    ],
    dtype=float
)


advanced_error_values = np.asarray(
    [
        rag_errors[
            "advanced_rag"
        ][
            "retrieval_misses_at_5"
        ],

        rag_errors[
            "advanced_rag"
        ][
            "decision_errors_with_source_hit_at_5"
        ],

        rag_errors[
            "advanced_rag"
        ][
            "hallucination_answers_with_source_hit_at_5"
        ]
    ],
    dtype=float
)


error_x = np.arange(
    len(
        error_labels
    )
)


width = 0.36


figure = plt.figure(
    figsize=(
        8.5,
        5.5
    )
)


axis = figure.add_axes(
    [
        0.11,
        0.18,
        0.85,
        0.75
    ]
)


basic_bars = axis.bar(
    error_x
    -
    width
    /
    2,
    basic_error_values,
    width,
    label="Basic RAG"
)


advanced_bars = axis.bar(
    error_x
    +
    width
    /
    2,
    advanced_error_values,
    width,
    label="Advanced RAG"
)


axis.set_ylabel(
    "Number of test questions"
)


axis.set_ylim(
    0,
    220
)


axis.set_xticks(
    error_x
)


axis.set_xticklabels(
    error_labels
)


axis.set_title(
    "Exploratory residual-error patterns"
)


axis.legend(
    frameon=False
)


axis.grid(
    axis="y",
    alpha=0.25
)


for bars in [
    basic_bars,
    advanced_bars
]:

    for bar in bars:

        value = int(
            bar.get_height()
        )


        axis.annotate(
            str(
                value
            ),
            xy=(
                bar.get_x()
                +
                bar.get_width()
                /
                2,
                value
            ),
            xytext=(
                0,
                4
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )


axis.text(
    0.99,
    0.02,
    "Exploratory analysis",
    transform=axis.transAxes,
    ha="right",
    va="bottom",
    fontsize=8
)


save_figure(
    figure,
    "figure_6_exploratory_residual_errors"
)


# 11. FIGURE MANIFEST

print(
    "\n[8] CREATING FIGURE MANIFEST"
)


manifest = {
    "analysis":
        "dissertation_figures",

    "presentation_only":
        True,

    "new_metrics":
        False,

    "new_statistical_tests":
        False,

    "figures": {
        "figure_1_decision_performance": {
            "title":
                "Final test decision performance",

            "role":
                "confirmatory"
        },

        "figure_2_v3_hallucination_rate": {
            "title":
                "Frozen V3 hallucination rate",

            "role":
                "confirmatory"
        },

        "figure_3_human_calibrated_hallucination": {
            "title":
                "Human-calibrated hallucination sensitivity analysis",

            "role":
                "sensitivity"
        },

        "figure_4_source_hit_at_k": {
            "title":
                "Correct-source retrieval",

            "role":
                "confirmatory"
        },

        "figure_5_gold_context_recall_at_k": {
            "title":
                "Gold-context evidence coverage",

            "role":
                "confirmatory"
        },

        "figure_6_exploratory_residual_errors": {
            "title":
                "Exploratory residual-error patterns",

            "role":
                "exploratory"
        }
    }
}


manifest_path = (
    FIGURE_DIR
    / "figure_manifest.json"
)


with manifest_path.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=2
    )


print(
    manifest_path
)


# 12. VALIDATE FILES

print(
    "\n[9] VALIDATING GENERATED FIGURES"
)


expected_base_names = [
    "figure_1_decision_performance",
    "figure_2_v3_hallucination_rate",
    "figure_3_human_calibrated_hallucination",
    "figure_4_source_hit_at_k",
    "figure_5_gold_context_recall_at_k",
    "figure_6_exploratory_residual_errors"
]


missing_files = []


for base_name in expected_base_names:

    for extension in [
        ".png",
        ".pdf"
    ]:

        path = (
            FIGURE_DIR
            / (
                base_name
                +
                extension
            )
        )


        if not path.exists():

            missing_files.append(
                str(
                    path
                )
            )


print(
    f"Expected figure files : "
    f"{len(expected_base_names) * 2}"
)


print(
    f"Missing figure files  : "
    f"{len(missing_files)}"
)


if missing_files:

    raise RuntimeError(
        "Some dissertation figure files "
        "were not generated."
    )


print(
    "PASS: All dissertation figures generated."
)


# 13. FINAL OUTPUT

print(
    "\n[10] FILES CREATED"
)


for path in sorted(
    FIGURE_DIR.iterdir()
):

    print(
        path
    )


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 49 COMPLETE - "
    "DISSERTATION FIGURES PREPARED"
)


print(
    "=" * 78
)

import csv
import hashlib
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


EVALUATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
)


FROZEN_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)


DECISION_SUMMARY_PATH = (
    EVALUATION_DIR
    / "final_test_decision_summary.json"
)


DECISION_STATS_PATH = (
    EVALUATION_DIR
    / "final_test_decision_statistics.json"
)


HALLUCINATION_SUMMARY_PATH = (
    EVALUATION_DIR
    / "final_test_hallucination_v3_summary.json"
)


HALLUCINATION_STATS_PATH = (
    EVALUATION_DIR
    / "final_test_hallucination_statistics.json"
)


CALIBRATION_PATH = (
    EVALUATION_DIR
    / "final_test_human_calibrated_hallucination_sensitivity.json"
)


RETRIEVAL_SUMMARY_PATH = (
    EVALUATION_DIR
    / "final_test_retrieval_summary.json"
)


RETRIEVAL_STATS_PATH = (
    EVALUATION_DIR
    / "final_test_retrieval_statistics.json"
)


OUTPUT_JSON_PATH = (
    EVALUATION_DIR
    / "integrated_final_results.json"
)


OUTPUT_CSV_PATH = (
    EVALUATION_DIR
    / "integrated_final_system_scorecard.csv"
)


# 2. SYSTEMS

SYSTEM_ORDER = [

    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


DISPLAY_NAMES = {

    "baseline_llm":
        "Baseline LLM",

    "basic_rag":
        "Basic RAG",

    "advanced_rag":
        "Advanced RAG",

    "gold_context_control":
        "Gold-context control"
}


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


def save_json(
    data,
    path
):

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def sha256_file(path):

    hasher = hashlib.sha256()


    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )


            if not chunk:

                break


            hasher.update(
                chunk
            )


    return hasher.hexdigest()


def relative_path(path):

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/"
    )


def recursive_find(
    obj,
    candidate_keys
):

    """
    Find the first numeric value whose exact key
    matches one of candidate_keys.

    Used only to tolerate minor schema differences
    between previously completed sections.
    """

    if isinstance(
        obj,
        dict
    ):

        for key in candidate_keys:

            if key in obj:

                value = obj[
                    key
                ]


                if isinstance(
                    value,
                    (
                        int,
                        float
                    )
                ):

                    return value


        for value in obj.values():

            result = recursive_find(
                value,
                candidate_keys
            )


            if result is not None:

                return result


    elif isinstance(
        obj,
        list
    ):

        for value in obj:

            result = recursive_find(
                value,
                candidate_keys
            )


            if result is not None:

                return result


    return None


def require_number(
    value,
    description
):

    if value is None:

        raise KeyError(
            f"Could not locate metric: "
            f"{description}"
        )


    return float(
        value
    )


def format_optional(
    value,
    decimals=4
):

    if value is None:

        return "-"


    return (
        f"{value:.{decimals}f}"
    )


# 4. START

print("=" * 78)

print(
    "SECTION 46 - INTEGRATED FINAL TEST RESULTS"
)

print("=" * 78)


print(
    """
This section performs integration only.

It does NOT:
- run another LLM judge
- change metric definitions
- run new hypothesis tests
- tune any system
- select a system using test performance
"""
)


# 5. LOAD ALL FINAL RESULT FILES


print(
    "[1] LOADING FINAL RESULT ARTIFACTS"
)


protocol = load_json(
    FROZEN_PROTOCOL_PATH
)


decision_summary = load_json(
    DECISION_SUMMARY_PATH
)


decision_stats = load_json(
    DECISION_STATS_PATH
)


hallucination_summary = load_json(
    HALLUCINATION_SUMMARY_PATH
)


hallucination_stats = load_json(
    HALLUCINATION_STATS_PATH
)


calibration = load_json(
    CALIBRATION_PATH
)


retrieval_summary = load_json(
    RETRIEVAL_SUMMARY_PATH
)


retrieval_stats = load_json(
    RETRIEVAL_STATS_PATH
)


input_paths = [

    FROZEN_PROTOCOL_PATH,
    DECISION_SUMMARY_PATH,
    DECISION_STATS_PATH,
    HALLUCINATION_SUMMARY_PATH,
    HALLUCINATION_STATS_PATH,
    CALIBRATION_PATH,
    RETRIEVAL_SUMMARY_PATH,
    RETRIEVAL_STATS_PATH
]


for path in input_paths:

    print(
        "PASS: "
        f"{relative_path(path)}"
    )


# 6. VERIFY FROZEN PROTOCOL

print(
    "\n[2] VERIFYING ANALYSIS SCOPE"
)


protocol_data = (
    protocol[
        "protocol"
    ]
)


if (
    protocol_data[
        "status"
    ]
    !=
    "FROZEN_BEFORE_FINAL_TEST"
):

    raise RuntimeError(
        "Unexpected frozen protocol status."
    )


primary_metrics = (
    protocol_data[
        "primary_metrics"
    ]
)


print(
    "Frozen primary metrics:"
)


for metric in primary_metrics:

    print(
        f"  - {metric}"
    )


required_primary_metrics = {

    "decision accuracy",
    "decision macro-F1",
    "hallucination rate",
    "groundedness",
    "contradiction rate",
    "retrieval Source Hit@k",
    "retrieval MRR@10",
    "Gold-context Recall@k"
}


if (
    set(
        primary_metrics
    )
    !=
    required_primary_metrics
):

    raise RuntimeError(
        "Frozen primary metric set "
        "does not match expected protocol."
    )


print(
    "PASS: Section 46 adds no "
    "new primary endpoint."
)


# 7. VERIFY TEST QUESTION COUNTS

print(
    "\n[3] VERIFYING TEST SAMPLE"
)


question_counts = {

    "decision":
        decision_summary.get(
            "questions"
        ),

    "hallucination":
        hallucination_summary.get(
            "questions"
        ),

    "decision_statistics":
        decision_stats.get(
            "questions"
        ),

    "hallucination_statistics":
        hallucination_stats.get(
            "questions"
        ),

    "retrieval":
        retrieval_summary.get(
            "questions"
        ),

    "retrieval_statistics":
        retrieval_stats.get(
            "questions"
        )
}


for name, count in (
    question_counts.items()
):

    print(
        f"{name:<28}: "
        f"{count}"
    )


    if count != 500:

        raise ValueError(
            f"{name} does not contain "
            "500 test questions."
        )


print(
    "PASS: All primary analyses use "
    "the same 500-question test set."
)


# 8. DECISION METRICS

print(
    "\n[4] EXTRACTING DECISION METRICS"
)


decision_systems = (
    decision_summary[
        "systems"
    ]
)


decision_metrics = {}


for system in SYSTEM_ORDER:

    if system not in decision_systems:

        raise KeyError(
            f"Missing decision system: "
            f"{system}"
        )


    record = (
        decision_systems[
            system
        ]
    )


    accuracy = require_number(

        recursive_find(
            record,
            [
                "accuracy",
                "decision_accuracy"
            ]
        ),

        f"{system} accuracy"
    )


    macro_f1 = require_number(

        recursive_find(
            record,
            [
                "macro_f1",
                "macro_f1_score",
                "macro-F1"
            ]
        ),

        f"{system} macro-F1"
    )


    decision_metrics[
        system
    ] = {

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1
    }


    print(
        f"{DISPLAY_NAMES[system]:<24} "
        f"accuracy={accuracy:.4f} "
        f"macro-F1={macro_f1:.4f}"
    )


# 9. HALLUCINATION METRICS

print(
    "\n[5] EXTRACTING FROZEN V3 CLAIM METRICS"
)


hallucination_systems = (
    hallucination_summary[
        "system_summaries"
    ]
)


hallucination_metrics = {}


for system in SYSTEM_ORDER:

    if system not in hallucination_systems:

        raise KeyError(
            f"Missing hallucination "
            f"system: {system}"
        )


    record = (
        hallucination_systems[
            system
        ]
    )


    total_claims = int(

        require_number(

            recursive_find(
                record,
                [
                    "total_claims"
                ]
            ),

            f"{system} total claims"
        )
    )


    supported_claims = int(

        require_number(

            recursive_find(
                record,
                [
                    "supported_claims"
                ]
            ),

            f"{system} supported claims"
        )
    )


    unsupported_claims = int(

        require_number(

            recursive_find(
                record,
                [
                    "unsupported_claims"
                ]
            ),

            f"{system} unsupported claims"
        )
    )


    contradicted_claims = int(

        require_number(

            recursive_find(
                record,
                [
                    "contradicted_claims"
                ]
            ),

            f"{system} contradicted claims"
        )
    )


    if (
        supported_claims
        +
        unsupported_claims
        +
        contradicted_claims
        !=
        total_claims
    ):

        raise RuntimeError(
            f"Claim partition mismatch "
            f"for {system}."
        )


    micro_hallucination_rate = (
        (
            unsupported_claims
            +
            contradicted_claims
        )
        /
        total_claims
    )


    groundedness = (
        supported_claims
        /
        total_claims
    )


    contradiction_rate = (
        contradicted_claims
        /
        total_claims
    )


    zero_claim_answers = recursive_find(
        record,
        [
            "zero_claim_answers",
            "zero_claim_count"
        ]
    )


    if zero_claim_answers is None:

        zero_claim_rate = None

    else:

        zero_claim_answers = int(
            zero_claim_answers
        )


        zero_claim_rate = (
            zero_claim_answers
            /
            500
        )


    frozen_micro_rate = recursive_find(
        record,
        [
            "micro_claim_hallucination_rate"
        ]
    )


    if (
        frozen_micro_rate is not None
        and
        abs(
            float(
                frozen_micro_rate
            )
            -
            micro_hallucination_rate
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Frozen hallucination-rate "
            f"mismatch for {system}."
        )


    hallucination_metrics[
        system
    ] = {

        "total_claims":
            total_claims,

        "supported_claims":
            supported_claims,

        "unsupported_claims":
            unsupported_claims,

        "contradicted_claims":
            contradicted_claims,

        "micro_hallucination_rate":
            micro_hallucination_rate,

        "groundedness":
            groundedness,

        "contradiction_rate":
            contradiction_rate,

        "zero_claim_answers":
            zero_claim_answers,

        "zero_claim_rate":
            zero_claim_rate
    }


    print(
        f"{DISPLAY_NAMES[system]:<24} "
        f"hall={micro_hallucination_rate:.4f} "
        f"ground={groundedness:.4f} "
        f"contra={contradiction_rate:.4f}"
    )


# 10. HUMAN-CALIBRATED SENSITIVITY RATES

print(
    "\n[6] EXTRACTING HUMAN-CALIBRATED SENSITIVITY RATES"
)


if (
    calibration[
        "primary_v3_results_replaced"
    ]
):

    raise RuntimeError(
        "Calibration file unexpectedly "
        "claims to replace primary V3 results."
    )


calibration_systems = (
    calibration[
        "systems"
    ]
)


calibrated_metrics = {}


for system in SYSTEM_ORDER:

    record = (
        calibration_systems[
            system
        ]
    )


    calibrated_rate = require_number(

        recursive_find(
            record,
            [
                "human_calibrated_hallucination_rate"
            ]
        ),

        f"{system} calibrated hallucination rate"
    )


    ci_record = (
        record[
            "bootstrap_95_ci"
        ]
    )


    ci_low = float(
        ci_record[
            "lower"
        ]
    )


    ci_high = float(
        ci_record[
            "upper"
        ]
    )


    calibrated_metrics[
        system
    ] = {

        "rate":
            calibrated_rate,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high
    }


    print(
        f"{DISPLAY_NAMES[system]:<24} "
        f"calibrated={calibrated_rate:.4f} "
        f"CI=[{ci_low:.4f}, {ci_high:.4f}]"
    )


# 11. RETRIEVAL METRICS

print(
    "\n[7] EXTRACTING RETRIEVAL METRICS"
)


retrieval_systems = (
    retrieval_summary[
        "system_summaries"
    ]
)


retrieval_metrics = {}


for system in [

    "basic_rag",
    "advanced_rag"

]:

    record = (
        retrieval_systems[
            system
        ]
    )


    source_hits = {

        k:
            float(
                record[
                    "source_hit_rates"
                ][
                    f"hit_at_{k}"
                ]
            )

        for k
        in [
            1,
            3,
            5,
            10
        ]
    }


    gold_recall = {

        k:
            float(
                record[
                    "gold_context_recall"
                ][
                    f"recall_at_{k}"
                ]
            )

        for k
        in [
            1,
            3,
            5,
            10
        ]
    }


    mrr = float(
        record[
            "mrr_at_10"
        ]
    )


    retrieval_metrics[
        system
    ] = {

        "source_hit":
            source_hits,

        "gold_context_recall":
            gold_recall,

        "mrr_at_10":
            mrr
    }


    print(
        f"\n{DISPLAY_NAMES[system]}"
    )


    print(
        f"  Hit@1/3/5/10 : "
        f"{source_hits[1]:.4f}, "
        f"{source_hits[3]:.4f}, "
        f"{source_hits[5]:.4f}, "
        f"{source_hits[10]:.4f}"
    )


    print(
        f"  MRR@10       : "
        f"{mrr:.4f}"
    )


    print(
        f"  Gold R@5     : "
        f"{gold_recall[5]:.4f}"
    )


# 12. BUILD SYSTEM SCORECARD

print(
    "\n[8] BUILDING INTEGRATED SYSTEM SCORECARD"
)


scorecard = {}


for system in SYSTEM_ORDER:

    row = {

        "system":
            system,

        "display_name":
            DISPLAY_NAMES[
                system
            ],

        "decision_accuracy":
            decision_metrics[
                system
            ][
                "accuracy"
            ],

        "decision_macro_f1":
            decision_metrics[
                system
            ][
                "macro_f1"
            ],

        "v3_hallucination_rate":
            hallucination_metrics[
                system
            ][
                "micro_hallucination_rate"
            ],

        "v3_groundedness":
            hallucination_metrics[
                system
            ][
                "groundedness"
            ],

        "v3_contradiction_rate":
            hallucination_metrics[
                system
            ][
                "contradiction_rate"
            ],

        "zero_claim_rate":
            hallucination_metrics[
                system
            ][
                "zero_claim_rate"
            ],

        "human_calibrated_hallucination_rate":
            calibrated_metrics[
                system
            ][
                "rate"
            ],

        "human_calibrated_ci_low":
            calibrated_metrics[
                system
            ][
                "ci_low"
            ],

        "human_calibrated_ci_high":
            calibrated_metrics[
                system
            ][
                "ci_high"
            ],

        "source_hit_at_1":
            None,

        "source_hit_at_3":
            None,

        "source_hit_at_5":
            None,

        "source_hit_at_10":
            None,

        "retrieval_mrr_at_10":
            None,

        "gold_context_recall_at_1":
            None,

        "gold_context_recall_at_3":
            None,

        "gold_context_recall_at_5":
            None,

        "gold_context_recall_at_10":
            None
    }


    if system in retrieval_metrics:

        retrieval = (
            retrieval_metrics[
                system
            ]
        )


        row[
            "source_hit_at_1"
        ] = (
            retrieval[
                "source_hit"
            ][
                1
            ]
        )


        row[
            "source_hit_at_3"
        ] = (
            retrieval[
                "source_hit"
            ][
                3
            ]
        )


        row[
            "source_hit_at_5"
        ] = (
            retrieval[
                "source_hit"
            ][
                5
            ]
        )


        row[
            "source_hit_at_10"
        ] = (
            retrieval[
                "source_hit"
            ][
                10
            ]
        )


        row[
            "retrieval_mrr_at_10"
        ] = (
            retrieval[
                "mrr_at_10"
            ]
        )


        row[
            "gold_context_recall_at_1"
        ] = (
            retrieval[
                "gold_context_recall"
            ][
                1
            ]
        )


        row[
            "gold_context_recall_at_3"
        ] = (
            retrieval[
                "gold_context_recall"
            ][
                3
            ]
        )


        row[
            "gold_context_recall_at_5"
        ] = (
            retrieval[
                "gold_context_recall"
            ][
                5
            ]
        )


        row[
            "gold_context_recall_at_10"
        ] = (
            retrieval[
                "gold_context_recall"
            ][
                10
            ]
        )


    scorecard[
        system
    ] = row


print(
    "PASS: Integrated system scorecard built."
)


# 13. DESCRIPTIVE DIFFERENCES

baseline_accuracy = (
    scorecard[
        "baseline_llm"
    ][
        "decision_accuracy"
    ]
)


baseline_hallucination = (
    scorecard[
        "baseline_llm"
    ][
        "v3_hallucination_rate"
    ]
)


descriptive_differences = {}


for system in [

    "basic_rag",
    "advanced_rag",
    "gold_context_control"

]:

    accuracy = (
        scorecard[
            system
        ][
            "decision_accuracy"
        ]
    )


    hallucination = (
        scorecard[
            system
        ][
            "v3_hallucination_rate"
        ]
    )


    descriptive_differences[
        system
    ] = {

        "accuracy_difference_vs_baseline":
            (
                accuracy
                -
                baseline_accuracy
            ),

        "hallucination_difference_vs_baseline":
            (
                hallucination
                -
                baseline_hallucination
            ),

        "hallucination_relative_reduction_vs_baseline":
            (
                (
                    baseline_hallucination
                    -
                    hallucination
                )
                /
                baseline_hallucination
            )
    }


# 14. COPY FROZEN INFERENTIAL RESULTS

# No new significance tests are performed.
# We preserve the exact inferential outputs generated in Sections 39, 42 and 45.


print(
    "\n[9] CONSOLIDATING EXISTING INFERENTIAL RESULTS"
)


statistical_results = {

    "decision":
        {

            "source":
                relative_path(
                    DECISION_STATS_PATH
                ),

            "comparisons":
                decision_stats[
                    "comparisons"
                ]
        },

    "hallucination":
        {

            "source":
                relative_path(
                    HALLUCINATION_STATS_PATH
                ),

            "observed_micro_rates":
                hallucination_stats[
                    "observed_micro_rates"
                ],

            "comparisons":
                hallucination_stats[
                    "comparisons"
                ]
        },

    "retrieval":
        {

            "source":
                relative_path(
                    RETRIEVAL_STATS_PATH
                ),

            "comparison":
                retrieval_stats[
                    "comparison"
                ],

            "source_hit_analysis":
                retrieval_stats[
                    "source_hit_analysis"
                ],

            "continuous_metric_analysis":
                retrieval_stats[
                    "continuous_metric_analysis"
                ]
        }
}


print(
    "PASS: Existing inferential results copied "
    "without recalculation."
)


# 15. PRINT FINAL PRIMARY SCORECARD

print(
    "\n[10] FINAL PRIMARY RESULTS SCORECARD"
)


print(
    f"{'System':<22}"
    f"{'Accuracy':>10}"
    f"{'Macro-F1':>11}"
    f"{'Halluc.':>10}"
    f"{'Grounded':>11}"
    f"{'Contra.':>10}"
    f"{'Calibr.':>10}"
)


print(
    "-" * 84
)


for system in SYSTEM_ORDER:

    row = (
        scorecard[
            system
        ]
    )


    print(
        f"{DISPLAY_NAMES[system]:<22}"
        f"{row['decision_accuracy']:>10.4f}"
        f"{row['decision_macro_f1']:>11.4f}"
        f"{row['v3_hallucination_rate']:>10.4f}"
        f"{row['v3_groundedness']:>11.4f}"
        f"{row['v3_contradiction_rate']:>10.4f}"
        f"{row['human_calibrated_hallucination_rate']:>10.4f}"
    )


print(
    "\nRetrieval systems:"
)


print(
    f"{'System':<22}"
    f"{'Hit@1':>9}"
    f"{'Hit@3':>9}"
    f"{'Hit@5':>9}"
    f"{'Hit@10':>9}"
    f"{'MRR@10':>10}"
    f"{'Gold R@5':>11}"
)


print(
    "-" * 79
)


for system in [

    "basic_rag",
    "advanced_rag"

]:

    row = (
        scorecard[
            system
        ]
    )


    print(
        f"{DISPLAY_NAMES[system]:<22}"
        f"{row['source_hit_at_1']:>9.4f}"
        f"{row['source_hit_at_3']:>9.4f}"
        f"{row['source_hit_at_5']:>9.4f}"
        f"{row['source_hit_at_10']:>9.4f}"
        f"{row['retrieval_mrr_at_10']:>10.4f}"
        f"{row['gold_context_recall_at_5']:>11.4f}"
    )


# 16. KEY DESCRIPTIVE EFFECTS

print(
    "\n[11] KEY DESCRIPTIVE EFFECTS VS BASELINE"
)


for system in [

    "basic_rag",
    "advanced_rag",
    "gold_context_control"

]:

    values = (
        descriptive_differences[
            system
        ]
    )


    print(
        f"\n{DISPLAY_NAMES[system]}"
    )


    print(
        f"  Accuracy difference     : "
        f"{values['accuracy_difference_vs_baseline'] * 100:+.2f} pp"
    )


    print(
        f"  Hallucination difference: "
        f"{values['hallucination_difference_vs_baseline'] * 100:+.2f} pp"
    )


    print(
        f"  Hallucination reduction : "
        f"{values['hallucination_relative_reduction_vs_baseline'] * 100:.2f}%"
    )


# 17. SAVE CSV SCORECARD

csv_fields = [

    "system",
    "display_name",

    "decision_accuracy",
    "decision_macro_f1",

    "v3_hallucination_rate",
    "v3_groundedness",
    "v3_contradiction_rate",
    "zero_claim_rate",

    "human_calibrated_hallucination_rate",
    "human_calibrated_ci_low",
    "human_calibrated_ci_high",

    "source_hit_at_1",
    "source_hit_at_3",
    "source_hit_at_5",
    "source_hit_at_10",

    "retrieval_mrr_at_10",

    "gold_context_recall_at_1",
    "gold_context_recall_at_3",
    "gold_context_recall_at_5",
    "gold_context_recall_at_10"
]


with OUTPUT_CSV_PATH.open(
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_fields
    )


    writer.writeheader()


    for system in SYSTEM_ORDER:

        writer.writerow(
            scorecard[
                system
            ]
        )


# 18. SAVE INTEGRATED JSON

output = {

    "split":
        "test",

    "analysis":
        "integrated_final_results",

    "integration_only":
        True,

    "new_metrics_added":
        False,

    "new_statistical_tests_performed":
        False,

    "new_llm_evaluator_used":
        False,

    "questions":
        500,

    "research_question":
        protocol_data[
            "research_question"
        ],

    "frozen_primary_metrics":
        primary_metrics,

    "systems":
        scorecard,

    "descriptive_effects_vs_baseline":
        descriptive_differences,

    "statistical_results":
        statistical_results,

    "human_calibration": {

        "role":
            "sensitivity analysis only",

        "primary_results_replaced":
            False,

        "limitation":
            calibration[
                "important_limitation"
            ]
    },

    "interpretation_guardrails": [

        (
            "Automatic V3 hallucination rates "
            "remain the primary hallucination results."
        ),

        (
            "Human-calibrated rates are sensitivity "
            "estimates and do not replace V3."
        ),

        (
            "Advanced RAG is not assumed superior "
            "to Basic RAG."
        ),

        (
            "Retrieval, decision and hallucination "
            "comparisons retain the paired "
            "500-question test design."
        ),

        (
            "No post-test tuning or system selection "
            "is performed in this integration step."
        )
    ],

    "source_files": {

        relative_path(
            path
        ):
            sha256_file(
                path
            )

        for path
        in input_paths
    }
}


save_json(
    output,
    OUTPUT_JSON_PATH
)


# 19. FINAL OUTPUT

print(
    "\n[12] FILES SAVED"
)


print(
    OUTPUT_JSON_PATH
)


print(
    OUTPUT_CSV_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 46 COMPLETE - "
    "INTEGRATED FINAL TEST RESULTS"
)


print(
    "=" * 78
)

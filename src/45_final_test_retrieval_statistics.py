import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


RETRIEVAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_retrieval_results.json"
)


RETRIEVAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_retrieval_summary.json"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_retrieval_statistics.json"
)


# 2. SETTINGS

BOOTSTRAP_SAMPLES = 10_000

RANDOM_SEED = 42


K_VALUES = [
    1,
    3,
    5,
    10
]


BASIC = "basic_rag"

ADVANCED = "advanced_rag"


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


# 4. EXACT MCNEMAR

def binomial_probability(
    n,
    k
):

    return (
        math.comb(
            n,
            k
        )
        *
        (
            0.5
            **
            n
        )
    )


def exact_mcnemar_p(
    a_only,
    b_only
):

    discordant = (
        a_only
        +
        b_only
    )


    if discordant == 0:

        return 1.0


    smaller = min(
        a_only,
        b_only
    )


    lower_tail = sum(

        binomial_probability(
            discordant,
            k
        )

        for k
        in range(
            smaller + 1
        )
    )


    return min(
        1.0,
        2.0
        *
        lower_tail
    )


# 5. HOLM CORRECTION

def holm_adjust(
    p_values
):

    number_tests = len(
        p_values
    )


    order = sorted(
        range(
            number_tests
        ),
        key=lambda index:
            p_values[
                index
            ]
    )


    adjusted = [
        None
    ] * number_tests


    running_max = 0.0


    for rank, original_index in enumerate(
        order
    ):

        multiplier = (
            number_tests
            -
            rank
        )


        candidate = min(
            1.0,
            multiplier
            *
            p_values[
                original_index
            ]
        )


        running_max = max(
            running_max,
            candidate
        )


        adjusted[
            original_index
        ] = running_max


    return adjusted


# 6. START

print("=" * 78)

print(
    "SECTION 45 - FINAL TEST PAIRED RETRIEVAL STATISTICS"
)

print("=" * 78)


# 7. LOAD DATA

print(
    "\n[1] LOADING FINAL RETRIEVAL EVALUATION"
)


detail_data = load_json(
    RETRIEVAL_RESULTS_PATH
)


summary_data = load_json(
    RETRIEVAL_SUMMARY_PATH
)


basic_records = (
    detail_data[
        "systems"
    ][
        BASIC
    ]
)


advanced_records = (
    detail_data[
        "systems"
    ][
        ADVANCED
    ]
)


print(
    f"Basic records    : "
    f"{len(basic_records)}"
)


print(
    f"Advanced records : "
    f"{len(advanced_records)}"
)


if len(
    basic_records
) != 500:

    raise ValueError(
        "Expected 500 Basic records."
    )


if len(
    advanced_records
) != 500:

    raise ValueError(
        "Expected 500 Advanced records."
    )


# 8. ALIGN PAIRED QUESTIONS

print(
    "\n[2] ALIGNING PAIRED QUESTIONS"
)


basic_by_id = {

    str(
        record[
            "question_id"
        ]
    ):
        record

    for record
    in basic_records
}


advanced_by_id = {

    str(
        record[
            "question_id"
        ]
    ):
        record

    for record
    in advanced_records
}


if (
    set(
        basic_by_id.keys()
    )
    !=
    set(
        advanced_by_id.keys()
    )
):

    raise ValueError(
        "Basic and Advanced question "
        "sets do not match."
    )


question_ids = sorted(
    basic_by_id.keys()
)


if len(
    question_ids
) != 500:

    raise ValueError(
        "Expected 500 paired questions."
    )


print(
    f"Paired questions : "
    f"{len(question_ids)}"
)


print(
    "PASS: Retrieval systems are "
    "aligned question-by-question."
)


# 9. BUILD ARRAYS

print(
    "\n[3] BUILDING PAIRED METRIC ARRAYS"
)


source_hits = {

    BASIC:
        {},

    ADVANCED:
        {}
}


gold_recall = {

    BASIC:
        {},

    ADVANCED:
        {}
}


for k in K_VALUES:

    source_hits[
        BASIC
    ][
        k
    ] = np.asarray(

        [
            basic_by_id[
                question_id
            ][
                "source_hits"
            ][
                f"hit_at_{k}"
            ]

            for question_id
            in question_ids
        ],

        dtype=np.int8
    )


    source_hits[
        ADVANCED
    ][
        k
    ] = np.asarray(

        [
            advanced_by_id[
                question_id
            ][
                "source_hits"
            ][
                f"hit_at_{k}"
            ]

            for question_id
            in question_ids
        ],

        dtype=np.int8
    )


    gold_recall[
        BASIC
    ][
        k
    ] = np.asarray(

        [
            basic_by_id[
                question_id
            ][
                "gold_context_recall"
            ][
                f"recall_at_{k}"
            ]

            for question_id
            in question_ids
        ],

        dtype=float
    )


    gold_recall[
        ADVANCED
    ][
        k
    ] = np.asarray(

        [
            advanced_by_id[
                question_id
            ][
                "gold_context_recall"
            ][
                f"recall_at_{k}"
            ]

            for question_id
            in question_ids
        ],

        dtype=float
    )


basic_rr = np.asarray(

    [
        basic_by_id[
            question_id
        ][
            "reciprocal_rank_at_10"
        ]

        for question_id
        in question_ids
    ],

    dtype=float
)


advanced_rr = np.asarray(

    [
        advanced_by_id[
            question_id
        ][
            "reciprocal_rank_at_10"
        ]

        for question_id
        in question_ids
    ],

    dtype=float
)


print(
    "PASS: Source-hit, Gold-recall, "
    "and reciprocal-rank arrays built."
)


# 10. VERIFY AGAINST SECTION 44 SUMMARY

print(
    "\n[4] VERIFYING SECTION 44 METRICS"
)


for system_name in [
    BASIC,
    ADVANCED
]:

    frozen = (
        summary_data[
            "system_summaries"
        ][
            system_name
        ]
    )


    for k in K_VALUES:

        calculated_hit = float(
            np.mean(
                source_hits[
                    system_name
                ][
                    k
                ]
            )
        )


        frozen_hit = (
            frozen[
                "source_hit_rates"
            ][
                f"hit_at_{k}"
            ]
        )


        if (
            abs(
                calculated_hit
                -
                frozen_hit
            )
            >
            1e-12
        ):

            raise RuntimeError(
                f"Hit@{k} mismatch "
                f"for {system_name}."
            )


        calculated_recall = float(
            np.mean(
                gold_recall[
                    system_name
                ][
                    k
                ]
            )
        )


        frozen_recall = (
            frozen[
                "gold_context_recall"
            ][
                f"recall_at_{k}"
            ]
        )


        if (
            abs(
                calculated_recall
                -
                frozen_recall
            )
            >
            1e-12
        ):

            raise RuntimeError(
                f"Recall@{k} mismatch "
                f"for {system_name}."
            )


    calculated_mrr = float(

        np.mean(

            basic_rr

            if system_name == BASIC

            else advanced_rr
        )
    )


    frozen_mrr = (
        frozen[
            "mrr_at_10"
        ]
    )


    if (
        abs(
            calculated_mrr
            -
            frozen_mrr
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"MRR mismatch "
            f"for {system_name}."
        )


    print(
        f"PASS: {system_name}"
    )


# 11. SOURCE-HIT MCNEMAR TESTS

print(
    "\n[5] SOURCE-HIT PAIRED MCNEMAR TESTS"
)


source_hit_results = []

source_hit_raw_p = []


for k in K_VALUES:

    basic = (
        source_hits[
            BASIC
        ][
            k
        ]
    )


    advanced = (
        source_hits[
            ADVANCED
        ][
            k
        ]
    )


    both_hit = int(
        np.sum(
            (basic == 1)
            &
            (advanced == 1)
        )
    )


    basic_only = int(
        np.sum(
            (basic == 1)
            &
            (advanced == 0)
        )
    )


    advanced_only = int(
        np.sum(
            (basic == 0)
            &
            (advanced == 1)
        )
    )


    neither = int(
        np.sum(
            (basic == 0)
            &
            (advanced == 0)
        )
    )


    p_value = exact_mcnemar_p(
        basic_only,
        advanced_only
    )


    difference = float(
        np.mean(
            advanced
        )
        -
        np.mean(
            basic
        )
    )


    source_hit_raw_p.append(
        p_value
    )


    source_hit_results.append(
        {

            "k":
                k,

            "basic_rate":
                float(
                    np.mean(
                        basic
                    )
                ),

            "advanced_rate":
                float(
                    np.mean(
                        advanced
                    )
                ),

            "advanced_minus_basic":
                difference,

            "both_hit":
                both_hit,

            "basic_only":
                basic_only,

            "advanced_only":
                advanced_only,

            "neither":
                neither,

            "mcnemar_p_raw":
                p_value,

            "mcnemar_p_holm":
                None
        }
    )


source_hit_holm = holm_adjust(
    source_hit_raw_p
)


for index, result in enumerate(
    source_hit_results
):

    result[
        "mcnemar_p_holm"
    ] = (
        source_hit_holm[
            index
        ]
    )


    print(
        f"\nHit@{result['k']}"
    )


    print(
        f"  Basic         : "
        f"{result['basic_rate']:.4f}"
    )


    print(
        f"  Advanced      : "
        f"{result['advanced_rate']:.4f}"
    )


    print(
        f"  Adv - Basic   : "
        f"{result['advanced_minus_basic'] * 100:+.2f} pp"
    )


    print(
        f"  Basic only    : "
        f"{result['basic_only']}"
    )


    print(
        f"  Advanced only : "
        f"{result['advanced_only']}"
    )


    print(
        f"  Raw p         : "
        f"{result['mcnemar_p_raw']:.8g}"
    )


    print(
        f"  Holm p        : "
        f"{result['mcnemar_p_holm']:.8g}"
    )


# 12. COMMON PAIRED BOOTSTRAP

print(
    "\n[6] PAIRED QUESTION BOOTSTRAP"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


bootstrap_indices = rng.integers(

    low=0,

    high=len(
        question_ids
    ),

    size=(
        BOOTSTRAP_SAMPLES,
        len(
            question_ids
        )
    ),

    dtype=np.int32
)


print(
    f"Bootstrap samples : "
    f"{BOOTSTRAP_SAMPLES}"
)


print(
    "Resampling unit   : question"
)


print(
    "Paired            : YES"
)


print(
    f"Random seed       : "
    f"{RANDOM_SEED}"
)


# 13. CONTINUOUS RETRIEVAL METRICS

continuous_metrics = []


for k in K_VALUES:

    continuous_metrics.append(
        {

            "name":
                f"gold_recall_at_{k}",

            "display":
                f"Gold Recall@{k}",

            "basic":
                gold_recall[
                    BASIC
                ][
                    k
                ],

            "advanced":
                gold_recall[
                    ADVANCED
                ][
                    k
                ]
        }
    )


continuous_metrics.append(
    {

        "name":
            "mrr_at_10",

        "display":
            "MRR@10",

        "basic":
            basic_rr,

        "advanced":
            advanced_rr
    }
)


continuous_results = []

continuous_raw_p = []


# 14. BOOTSTRAP + WILCOXON

print(
    "\n[7] GOLD-RECALL AND MRR PAIRED RESULTS"
)


for metric in continuous_metrics:

    basic = (
        metric[
            "basic"
        ]
    )


    advanced = (
        metric[
            "advanced"
        ]
    )


    differences = (
        advanced
        -
        basic
    )


    observed_difference = float(
        np.mean(
            differences
        )
    )


    bootstrap_difference = np.mean(

        differences[
            bootstrap_indices
        ],

        axis=1
    )


    ci_low = float(
        np.percentile(
            bootstrap_difference,
            2.5
        )
    )


    ci_high = float(
        np.percentile(
            bootstrap_difference,
            97.5
        )
    )


    advanced_better = int(
        np.sum(
            differences > 0
        )
    )


    basic_better = int(
        np.sum(
            differences < 0
        )
    )


    tied = int(
        np.sum(
            differences == 0
        )
    )


    if np.allclose(
        differences,
        0.0
    ):

        statistic = 0.0
        p_value = 1.0


    else:

        test = wilcoxon(

            advanced,

            basic,

            alternative="two-sided",

            zero_method="wilcox",

            method="auto"
        )


        statistic = float(
            test.statistic
        )


        p_value = float(
            test.pvalue
        )


    continuous_raw_p.append(
        p_value
    )


    continuous_results.append(
        {

            "metric":
                metric[
                    "name"
                ],

            "basic_mean":
                float(
                    np.mean(
                        basic
                    )
                ),

            "advanced_mean":
                float(
                    np.mean(
                        advanced
                    )
                ),

            "advanced_minus_basic":
                observed_difference,

            "paired_bootstrap_95_ci": [
                ci_low,
                ci_high
            ],

            "questions_advanced_better":
                advanced_better,

            "questions_basic_better":
                basic_better,

            "questions_tied":
                tied,

            "wilcoxon_statistic":
                statistic,

            "wilcoxon_p_raw":
                p_value,

            "wilcoxon_p_holm":
                None
        }
    )


continuous_holm = holm_adjust(
    continuous_raw_p
)


for index, result in enumerate(
    continuous_results
):

    result[
        "wilcoxon_p_holm"
    ] = (
        continuous_holm[
            index
        ]
    )


    metric_display = (
        continuous_metrics[
            index
        ][
            "display"
        ]
    )


    print(
        f"\n{metric_display}"
    )


    print(
        f"  Basic mean       : "
        f"{result['basic_mean']:.4f}"
    )


    print(
        f"  Advanced mean    : "
        f"{result['advanced_mean']:.4f}"
    )


    print(
        f"  Adv - Basic      : "
        f"{result['advanced_minus_basic']:+.4f} "
        f"({result['advanced_minus_basic'] * 100:+.2f} pp)"
    )


    print(
        f"  Bootstrap 95% CI : "
        f"["
        f"{result['paired_bootstrap_95_ci'][0] * 100:+.2f}, "
        f"{result['paired_bootstrap_95_ci'][1] * 100:+.2f}"
        f"] pp"
    )


    print(
        f"  Advanced better  : "
        f"{result['questions_advanced_better']}"
    )


    print(
        f"  Basic better     : "
        f"{result['questions_basic_better']}"
    )


    print(
        f"  Tied             : "
        f"{result['questions_tied']}"
    )


    print(
        f"  Wilcoxon raw p   : "
        f"{result['wilcoxon_p_raw']:.8g}"
    )


    print(
        f"  Wilcoxon Holm p  : "
        f"{result['wilcoxon_p_holm']:.8g}"
    )


# 15. COMPACT SCORECARD

print(
    "\n[8] RETRIEVAL INFERENCE SCORECARD"
)


print(
    f"{'Metric':<20}"
    f"{'Adv-Basic':>12}"
    f"{'CI low':>12}"
    f"{'CI high':>12}"
    f"{'Holm p':>14}"
)


print(
    "-" * 70
)


for result in continuous_results:

    print(
        f"{result['metric']:<20}"
        f"{result['advanced_minus_basic'] * 100:>+11.2f}%"
        f"{result['paired_bootstrap_95_ci'][0] * 100:>+11.2f}%"
        f"{result['paired_bootstrap_95_ci'][1] * 100:>+11.2f}%"
        f"{result['wilcoxon_p_holm']:>14.6g}"
    )


print(
    "\nSource Hit McNemar:"
)


for result in source_hit_results:

    print(
        f"  Hit@{result['k']:<2} "
        f"diff="
        f"{result['advanced_minus_basic'] * 100:+.2f} pp "
        f"Holm p="
        f"{result['mcnemar_p_holm']:.6g}"
    )


# 16. SAVE

output = {

    "split":
        "test",

    "analysis":
        "paired_final_retrieval_statistics",

    "questions":
        500,

    "comparison":
        "advanced_rag_minus_basic_rag",

    "source_hit_analysis": {

        "test":
            "exact McNemar",

        "multiple_testing":
            "Holm",

        "family_size":
            len(
                K_VALUES
            ),

        "results":
            source_hit_results
    },

    "continuous_metric_analysis": {

        "metrics": [
            "gold_recall_at_1",
            "gold_recall_at_3",
            "gold_recall_at_5",
            "gold_recall_at_10",
            "mrr_at_10"
        ],

        "effect":
            (
                "mean paired "
                "advanced-minus-basic difference"
            ),

        "bootstrap": {

            "samples":
                BOOTSTRAP_SAMPLES,

            "seed":
                RANDOM_SEED,

            "unit":
                "question",

            "paired":
                True,

            "confidence_level":
                0.95
        },

        "test":
            "Wilcoxon signed-rank",

        "multiple_testing":
            "Holm",

        "family_size":
            len(
                continuous_metrics
            ),

        "results":
            continuous_results
    }
}


save_json(
    output,
    OUTPUT_PATH
)


# 17. FINAL OUTPUT

print(
    "\n[9] FILE SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 45 COMPLETE - "
    "FINAL PAIRED RETRIEVAL STATISTICS"
)


print(
    "=" * 78
)


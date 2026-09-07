import json
import math
import random
from pathlib import Path

from scipy.stats import (
    binomtest,
    wilcoxon
)


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

DECISION_PATH = (
    GENERATION_RESULTS_DIR
    / "development_decision_evaluation.json"
)

HALLUCINATION_PATH = (
    GENERATION_RESULTS_DIR
    / "development_hallucination_v3.json"
)

OUTPUT_PATH = (
    GENERATION_RESULTS_DIR
    / "development_statistical_analysis.json"
)


# 2. SETTINGS

SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


PAIRINGS = [

    (
        "baseline_llm",
        "basic_rag"
    ),

    (
        "baseline_llm",
        "advanced_rag"
    ),

    (
        "basic_rag",
        "advanced_rag"
    ),

    (
        "advanced_rag",
        "gold_context_control"
    ),

    (
        "basic_rag",
        "gold_context_control"
    )
]


BOOTSTRAP_ITERATIONS = 10000

RANDOM_SEED = 42

ALPHA = 0.05


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

        return json.load(file)


def save_json(data, path):

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


def mean(values):

    if not values:

        return None

    return sum(values) / len(values)


# 4. PERCENTILE

def percentile(
    sorted_values,
    probability
):

    if not sorted_values:

        return None


    position = (
        probability
        *
        (
            len(sorted_values)
            -
            1
        )
    )


    lower_index = int(
        math.floor(
            position
        )
    )


    upper_index = int(
        math.ceil(
            position
        )
    )


    if (
        lower_index
        ==
        upper_index
    ):

        return sorted_values[
            lower_index
        ]


    fraction = (
        position
        -
        lower_index
    )


    lower_value = (
        sorted_values[
            lower_index
        ]
    )


    upper_value = (
        sorted_values[
            upper_index
        ]
    )


    return (

        lower_value
        +
        fraction
        *
        (
            upper_value
            -
            lower_value
        )
    )


# 5. PAIRED BOOTSTRAP CI

def paired_bootstrap_ci(
    values_a,
    values_b,
    iterations,
    seed
):

    if (
        len(values_a)
        !=
        len(values_b)
    ):

        raise ValueError(
            "Paired samples must "
            "have equal length."
        )


    sample_size = len(
        values_a
    )


    if sample_size == 0:

        return {
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None
        }


    observed_differences = [

        a - b

        for a, b
        in zip(
            values_a,
            values_b
        )
    ]


    observed_mean_difference = (
        mean(
            observed_differences
        )
    )


    rng = random.Random(
        seed
    )


    bootstrap_means = []


    for _ in range(
        iterations
    ):

        sampled_differences = [

            observed_differences[
                rng.randrange(
                    sample_size
                )
            ]

            for _ in range(
                sample_size
            )
        ]


        bootstrap_means.append(
            mean(
                sampled_differences
            )
        )


    bootstrap_means.sort()


    ci_low = percentile(
        bootstrap_means,
        0.025
    )


    ci_high = percentile(
        bootstrap_means,
        0.975
    )


    return {

        "mean_difference":
            observed_mean_difference,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high
    }


# 6. LOAD DATA

print("=" * 78)

print(
    "SECTION 23 - DEVELOPMENT STATISTICAL ANALYSIS"
)

print("=" * 78)


decision_data = load_json(
    DECISION_PATH
)

hallucination_data = load_json(
    HALLUCINATION_PATH
)


print(
    "\n[1] DATA LOADED"
)


print(
    f"Decision questions      : "
    f"{decision_data['questions']}"
)


print(
    f"Hallucination questions : "
    f"{len(hallucination_data['judgments'])}"
)


# 7. INDEX DECISION CORRECTNESS

decision_correctness = {

    system_name: {}

    for system_name
    in SYSTEM_ORDER
}


for record in decision_data[
    "question_details"
]:

    question_id = (
        record[
            "question_id"
        ]
    )


    for system_name in SYSTEM_ORDER:

        decision_correctness[
            system_name
        ][
            question_id
        ] = (

            record[
                "systems"
            ][
                system_name
            ][
                "correct"
            ]
        )


# 8. INDEX HALLUCINATION RESULTS

hallucination_rates = {

    system_name: {}

    for system_name
    in SYSTEM_ORDER
}


zero_claim_status = {

    system_name: {}

    for system_name
    in SYSTEM_ORDER
}


for judgment in hallucination_data[
    "judgments"
]:

    question_id = (
        judgment[
            "question_id"
        ]
    )


    for system_name in SYSTEM_ORDER:

        result = (

            judgment[
                "systems"
            ][
                system_name
            ]
        )


        hallucination_rates[
            system_name
        ][
            question_id
        ] = (

            result[
                "hallucination_rate"
            ]
        )


        zero_claim_status[
            system_name
        ][
            question_id
        ] = (

            result[
                "claim_counts"
            ][
                "total"
            ]
            ==
            0
        )


question_ids = sorted(

    hallucination_rates[
        "baseline_llm"
    ].keys()
)


if len(question_ids) != 500:

    raise ValueError(
        "Expected 500 paired questions."
    )


# 9. MCNEMAR TEST
# McNemar examines discordant pairs:
# b = A correct, B wrong
# c = A wrong, B correct
# Exact two-sided binomial test is used.


def exact_mcnemar(
    system_a,
    system_b
):

    both_correct = 0

    both_wrong = 0

    a_only_correct = 0

    b_only_correct = 0


    for question_id in question_ids:

        a_correct = (

            decision_correctness[
                system_a
            ][
                question_id
            ]
        )


        b_correct = (

            decision_correctness[
                system_b
            ][
                question_id
            ]
        )


        if (
            a_correct
            and
            b_correct
        ):

            both_correct += 1


        elif (
            not a_correct
            and
            not b_correct
        ):

            both_wrong += 1


        elif (
            a_correct
            and
            not b_correct
        ):

            a_only_correct += 1


        else:

            b_only_correct += 1


    discordant = (

        a_only_correct
        +
        b_only_correct
    )


    if discordant > 0:

        p_value = (

            binomtest(

                min(
                    a_only_correct,
                    b_only_correct
                ),

                n=discordant,

                p=0.5,

                alternative="two-sided"

            ).pvalue
        )


    else:

        p_value = 1.0


    accuracy_a = mean([

        1.0

        if decision_correctness[
            system_a
        ][
            question_id
        ]

        else 0.0

        for question_id
        in question_ids
    ])


    accuracy_b = mean([

        1.0

        if decision_correctness[
            system_b
        ][
            question_id
        ]

        else 0.0

        for question_id
        in question_ids
    ])


    return {

        "system_a":
            system_a,

        "system_b":
            system_b,

        "accuracy_a":
            accuracy_a,

        "accuracy_b":
            accuracy_b,

        "accuracy_difference_a_minus_b":
            (
                accuracy_a
                -
                accuracy_b
            ),

        "both_correct":
            both_correct,

        "both_wrong":
            both_wrong,

        "a_only_correct":
            a_only_correct,

        "b_only_correct":
            b_only_correct,

        "discordant_pairs":
            discordant,

        "mcnemar_exact_p":
            p_value
    }


# 10. DECISION STATISTICS

print(
    "\n[2] PAIRED DECISION ACCURACY - MCNEMAR"
)


decision_statistics = {}


for system_a, system_b in PAIRINGS:

    result = exact_mcnemar(
        system_a,
        system_b
    )


    key = (
        f"{system_a}_vs_{system_b}"
    )


    decision_statistics[
        key
    ] = result


    print(
        "\n"
        + "-" * 72
    )


    print(
        f"{system_a} VS {system_b}"
    )


    print(
        f"Accuracy A             : "
        f"{result['accuracy_a']:.4f}"
    )


    print(
        f"Accuracy B             : "
        f"{result['accuracy_b']:.4f}"
    )


    print(
        f"Difference A - B       : "
        f"{result['accuracy_difference_a_minus_b']:+.4f}"
    )


    print(
        f"A only correct         : "
        f"{result['a_only_correct']}"
    )


    print(
        f"B only correct         : "
        f"{result['b_only_correct']}"
    )


    print(
        f"Exact McNemar p-value  : "
        f"{result['mcnemar_exact_p']:.6g}"
    )



# 11. HALLUCINATION PAIRED ANALYSIS


print(
    "\n[3] PAIRED HALLUCINATION ANALYSIS"
)


hallucination_statistics = {}


for pair_number, (
    system_a,
    system_b
) in enumerate(
    PAIRINGS,
    start=1
):

    paired_a = []

    paired_b = []

    excluded_due_zero_claim = 0


    for question_id in question_ids:

        rate_a = (

            hallucination_rates[
                system_a
            ][
                question_id
            ]
        )


        rate_b = (

            hallucination_rates[
                system_b
            ][
                question_id
            ]
        )


        # Hallucination is undefined when there
        # are zero scored factual claims.

        if (
            rate_a is None
            or
            rate_b is None
        ):

            excluded_due_zero_claim += 1

            continue


        paired_a.append(
            rate_a
        )

        paired_b.append(
            rate_b
        )


    differences = [

        a - b

        for a, b in zip(
            paired_a,
            paired_b
        )
    ]


    # Wilcoxon signed-rank test

    nonzero_differences = [

        difference

        for difference
        in differences

        if difference != 0
    ]


    if nonzero_differences:

        wilcoxon_result = wilcoxon(

            paired_a,
            paired_b,

            alternative="two-sided",

            zero_method="wilcox",

            method="auto"
        )


        wilcoxon_statistic = float(
            wilcoxon_result.statistic
        )


        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )


    else:

        wilcoxon_statistic = 0.0

        wilcoxon_p = 1.0


    # Paired bootstrap confidence interval

    bootstrap = paired_bootstrap_ci(

        paired_a,
        paired_b,

        iterations=(
            BOOTSTRAP_ITERATIONS
        ),

        seed=(
            RANDOM_SEED
            +
            pair_number
        )
    )


    mean_a = mean(
        paired_a
    )


    mean_b = mean(
        paired_b
    )


    median_difference = None


    if differences:

        sorted_differences = sorted(
            differences
        )


        median_difference = percentile(
            sorted_differences,
            0.5
        )


    key = (
        f"{system_a}_vs_{system_b}"
    )


    result = {

        "system_a":
            system_a,

        "system_b":
            system_b,

        "paired_answers":
            len(
                paired_a
            ),

        "excluded_zero_claim_pairs":
            excluded_due_zero_claim,

        "mean_hallucination_a":
            mean_a,

        "mean_hallucination_b":
            mean_b,

        "mean_difference_a_minus_b":
            bootstrap[
                "mean_difference"
            ],

        "bootstrap_95_ci_low":
            bootstrap[
                "ci_low"
            ],

        "bootstrap_95_ci_high":
            bootstrap[
                "ci_high"
            ],

        "median_difference":
            median_difference,

        "wilcoxon_statistic":
            wilcoxon_statistic,

        "wilcoxon_p":
            wilcoxon_p
    }


    hallucination_statistics[
        key
    ] = result


    print(
        "\n"
        + "-" * 72
    )


    print(
        f"{system_a} VS {system_b}"
    )


    print(
        f"Paired answers          : "
        f"{result['paired_answers']}"
    )


    print(
        f"Zero-claim exclusions   : "
        f"{result['excluded_zero_claim_pairs']}"
    )


    print(
        f"Mean hallucination A    : "
        f"{result['mean_hallucination_a']:.4f}"
    )


    print(
        f"Mean hallucination B    : "
        f"{result['mean_hallucination_b']:.4f}"
    )


    print(
        f"Mean difference A - B  : "
        f"{result['mean_difference_a_minus_b']:+.4f}"
    )


    print(
        "Bootstrap 95% CI        : "
        f"["
        f"{result['bootstrap_95_ci_low']:+.4f}, "
        f"{result['bootstrap_95_ci_high']:+.4f}"
        f"]"
    )


    print(
        f"Wilcoxon p-value        : "
        f"{result['wilcoxon_p']:.6g}"
    )


# 12. ZERO-CLAIM PAIRED TESTS

# We separately test whether one system produces zero-claim/abstention answers more frequently than another.

print(
    "\n[4] ZERO-CLAIM / ABSTENTION COMPARISON"
)


zero_claim_statistics = {}


for system_a, system_b in PAIRINGS:

    a_only_zero = 0

    b_only_zero = 0

    both_zero = 0

    neither_zero = 0


    for question_id in question_ids:

        a_zero = (

            zero_claim_status[
                system_a
            ][
                question_id
            ]
        )


        b_zero = (

            zero_claim_status[
                system_b
            ][
                question_id
            ]
        )


        if (
            a_zero
            and
            b_zero
        ):

            both_zero += 1


        elif (
            a_zero
            and
            not b_zero
        ):

            a_only_zero += 1


        elif (
            not a_zero
            and
            b_zero
        ):

            b_only_zero += 1


        else:

            neither_zero += 1


    discordant = (

        a_only_zero
        +
        b_only_zero
    )


    if discordant > 0:

        p_value = (

            binomtest(

                min(
                    a_only_zero,
                    b_only_zero
                ),

                n=discordant,

                p=0.5,

                alternative="two-sided"

            ).pvalue
        )


    else:

        p_value = 1.0


    key = (
        f"{system_a}_vs_{system_b}"
    )


    result = {

        "system_a":
            system_a,

        "system_b":
            system_b,

        "a_only_zero":
            a_only_zero,

        "b_only_zero":
            b_only_zero,

        "both_zero":
            both_zero,

        "neither_zero":
            neither_zero,

        "exact_mcnemar_p":
            p_value
    }


    zero_claim_statistics[
        key
    ] = result


    print(
        "\n"
        + "-"
        * 72
    )


    print(
        f"{system_a} VS {system_b}"
    )


    print(
        f"A only zero-claim      : "
        f"{a_only_zero}"
    )


    print(
        f"B only zero-claim      : "
        f"{b_only_zero}"
    )


    print(
        f"Both zero-claim        : "
        f"{both_zero}"
    )


    print(
        f"Exact McNemar p-value  : "
        f"{p_value:.6g}"
    )


# 13. MULTIPLE-COMPARISON

print(
    "\n[5] SIGNIFICANCE INTERPRETATION"
)


print(
    """
Primary comparisons for the dissertation:

1. Baseline LLM vs Basic RAG
2. Baseline LLM vs Advanced RAG
3. Basic RAG vs Advanced RAG

The Gold-context system is an oracle/control
and is interpreted primarily as a reference.

Because multiple inferential comparisons are
performed, raw p-values should not be treated
as the only evidence.

Effect sizes and 95% confidence intervals
will be reported alongside significance tests.
"""
)


# 14. SAVE

output = {

    "split":
        "development",

    "questions":
        500,

    "bootstrap_iterations":
        BOOTSTRAP_ITERATIONS,

    "random_seed":
        RANDOM_SEED,

    "alpha":
        ALPHA,

    "decision_statistics":
        decision_statistics,

    "hallucination_statistics":
        hallucination_statistics,

    "zero_claim_statistics":
        zero_claim_statistics
}


save_json(
    output,
    OUTPUT_PATH
)


print(
    "\n[6] RESULTS SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 23 COMPLETE"
)

print(
    "=" * 78
)
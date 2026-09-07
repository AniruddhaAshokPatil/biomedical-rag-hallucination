import hashlib
import json
import math

from pathlib import Path

import numpy as np


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


FINAL_FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "final_test_artifact_manifest.json"
)


DECISION_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_decision_results.json"
)


DECISION_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_decision_summary.json"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_decision_statistics.json"
)


# 2. SETTINGS

BOOTSTRAP_SAMPLES = 10_000

RANDOM_SEED = 42


LABELS = [
    "yes",
    "no",
    "maybe"
]


LABEL_TO_INT = {
    "yes": 0,
    "no": 1,
    "maybe": 2
}


SYSTEM_DISPLAY_NAMES = {

    "baseline_llm":
        "Baseline LLM",

    "basic_rag":
        "Basic RAG",

    "advanced_rag":
        "Advanced RAG",

    "gold_context_control":
        "Gold-context control"
}


# 3. PLANNED COMPARISONS

# Difference direction is always:
#   system_b - system_a


COMPARISONS = [

    {
        "name":
            "basic_vs_baseline",

        "system_a":
            "baseline_llm",

        "system_b":
            "basic_rag"
    },

    {
        "name":
            "advanced_vs_baseline",

        "system_a":
            "baseline_llm",

        "system_b":
            "advanced_rag"
    },

    {
        "name":
            "advanced_vs_basic",

        "system_a":
            "basic_rag",

        "system_b":
            "advanced_rag"
    },

    {
        "name":
            "gold_vs_advanced",

        "system_a":
            "advanced_rag",

        "system_b":
            "gold_context_control"
    },

    {
        "name":
            "gold_vs_basic",

        "system_a":
            "basic_rag",

        "system_b":
            "gold_context_control"
    }
]


# 4. HELPERS

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


# 5. EXACT BINOMIAL / MCNEMAR

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
    a_only_correct,
    b_only_correct
):

    discordant = (
        a_only_correct
        +
        b_only_correct
    )


    if discordant == 0:

        return 1.0


    smaller = min(
        a_only_correct,
        b_only_correct
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


    p_value = min(
        1.0,
        2.0
        *
        lower_tail
    )


    return p_value


# 6. HOLM CORRECTION

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


# 7. MACRO-F1

def macro_f1(
    gold,
    prediction
):

    class_f1_scores = []


    for label in range(
        len(
            LABELS
        )
    ):

        true_positive = np.sum(
            (
                gold == label
            )
            &
            (
                prediction == label
            )
        )


        false_positive = np.sum(
            (
                gold != label
            )
            &
            (
                prediction == label
            )
        )


        false_negative = np.sum(
            (
                gold == label
            )
            &
            (
                prediction != label
            )
        )


        precision_denominator = (
            true_positive
            +
            false_positive
        )


        recall_denominator = (
            true_positive
            +
            false_negative
        )


        precision = (

            true_positive
            /
            precision_denominator

            if precision_denominator
            >
            0

            else 0.0
        )


        recall = (

            true_positive
            /
            recall_denominator

            if recall_denominator
            >
            0

            else 0.0
        )


        if (
            precision
            +
            recall
        ) == 0:

            f1 = 0.0

        else:

            f1 = (
                2.0
                *
                precision
                *
                recall
                /
                (
                    precision
                    +
                    recall
                )
            )


        class_f1_scores.append(
            f1
        )


    return float(
        np.mean(
            class_f1_scores
        )
    )


# 8. VECTORIZED BOOTSTRAP MACRO-F1

def bootstrap_macro_f1(
    sampled_gold,
    sampled_prediction
):

    number_bootstraps = (
        sampled_gold.shape[
            0
        ]
    )


    f1_sum = np.zeros(
        number_bootstraps,
        dtype=float
    )


    for label in range(
        len(
            LABELS
        )
    ):

        true_positive = np.sum(

            (
                sampled_gold
                ==
                label
            )
            &
            (
                sampled_prediction
                ==
                label
            ),

            axis=1
        )


        false_positive = np.sum(

            (
                sampled_gold
                !=
                label
            )
            &
            (
                sampled_prediction
                ==
                label
            ),

            axis=1
        )


        false_negative = np.sum(

            (
                sampled_gold
                ==
                label
            )
            &
            (
                sampled_prediction
                !=
                label
            ),

            axis=1
        )


        precision_denominator = (
            true_positive
            +
            false_positive
        )


        recall_denominator = (
            true_positive
            +
            false_negative
        )


        precision = np.divide(

            true_positive,

            precision_denominator,

            out=np.zeros_like(
                true_positive,
                dtype=float
            ),

            where=(
                precision_denominator
                !=
                0
            )
        )


        recall = np.divide(

            true_positive,

            recall_denominator,

            out=np.zeros_like(
                true_positive,
                dtype=float
            ),

            where=(
                recall_denominator
                !=
                0
            )
        )


        f1_denominator = (
            precision
            +
            recall
        )


        f1 = np.divide(

            2.0
            *
            precision
            *
            recall,

            f1_denominator,

            out=np.zeros_like(
                precision,
                dtype=float
            ),

            where=(
                f1_denominator
                !=
                0
            )
        )


        f1_sum += f1


    return (
        f1_sum
        /
        len(
            LABELS
        )
    )


# 9. START

print("=" * 78)

print(
    "SECTION 39 - FINAL TEST PAIRED DECISION STATISTICS"
)

print("=" * 78)


# 10. VERIFY FROZEN FINAL ANSWERS HAVE NOT CHANGED

print(
    "\n[1] VERIFYING FINAL-TEST FREEZE"
)


freeze_manifest = load_json(
    FINAL_FREEZE_MANIFEST_PATH
)


if (
    freeze_manifest[
        "status"
    ]
    !=
    "FROZEN_BEFORE_TEST_EVALUATION"
):

    raise RuntimeError(
        "Final test was not frozen "
        "before evaluation."
    )


changed_files = []

missing_files = []


for relative_path, expected_hash in (
    freeze_manifest[
        "sha256"
    ].items()
):

    path = (
        PROJECT_ROOT
        /
        relative_path
    )


    if not path.exists():

        missing_files.append(
            relative_path
        )

        continue


    actual_hash = sha256_file(
        path
    )


    if (
        actual_hash
        !=
        expected_hash
    ):

        changed_files.append(
            relative_path
        )


print(
    f"Frozen files checked : "
    f"{len(freeze_manifest['sha256'])}"
)


print(
    f"Changed files        : "
    f"{len(changed_files)}"
)


print(
    f"Missing files        : "
    f"{len(missing_files)}"
)


if (
    changed_files
    or
    missing_files
):

    raise RuntimeError(
        "Frozen test artifacts changed "
        "after generation."
    )


print(
    "PASS: Frozen test answers remain unchanged."
)


# 11. LOAD SECTION 38 RESULTS

print(
    "\n[2] LOADING FINAL DECISION RESULTS"
)


paired_results = load_json(
    DECISION_RESULTS_PATH
)


decision_summary = load_json(
    DECISION_SUMMARY_PATH
)


if len(
    paired_results
) != 500:

    raise ValueError(
        "Expected 500 paired results."
    )


print(
    f"Paired questions : "
    f"{len(paired_results)}"
)



# 12. CREATE NUMPY ARRAYS


gold = np.array(
    [

        LABEL_TO_INT[
            record[
                "gold_decision"
            ]
        ]

        for record
        in paired_results
    ],
    dtype=np.int8
)


predictions = {}


correctness = {}


for system in SYSTEM_DISPLAY_NAMES:

    predictions[
        system
    ] = np.array(
        [

            LABEL_TO_INT[
                record[
                    system
                ][
                    "predicted_decision"
                ]
            ]

            for record
            in paired_results
        ],
        dtype=np.int8
    )


    correctness[
        system
    ] = (

        predictions[
            system
        ]

        ==

        gold
    )


# 13. GENERATE ONE COMMON PAIRED BOOTSTRAP SAMPLE

# Same resampled question indices are used for all systems.
# This preserves the paired design.


print(
    "\n[3] GENERATING PAIRED BOOTSTRAP SAMPLES"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


bootstrap_indices = rng.integers(

    low=0,

    high=len(
        paired_results
    ),

    size=(
        BOOTSTRAP_SAMPLES,
        len(
            paired_results
        )
    ),

    dtype=np.int32
)


sampled_gold = gold[
    bootstrap_indices
]


print(
    f"Bootstrap samples : "
    f"{BOOTSTRAP_SAMPLES}"
)


print(
    f"Random seed       : "
    f"{RANDOM_SEED}"
)


# 14. PRECOMPUTE BOOTSTRAP METRICS PER SYSTEM

bootstrap_accuracy = {}

bootstrap_macro_f1_values = {}


for system in SYSTEM_DISPLAY_NAMES:

    sampled_prediction = (
        predictions[
            system
        ][
            bootstrap_indices
        ]
    )


    bootstrap_accuracy[
        system
    ] = np.mean(

        sampled_prediction
        ==
        sampled_gold,

        axis=1
    )


    bootstrap_macro_f1_values[
        system
    ] = bootstrap_macro_f1(
        sampled_gold,
        sampled_prediction
    )


# 15. RUN PLANNED COMPARISONS

print(
    "\n[4] PAIRED COMPARISONS"
)


comparison_results = []


raw_mcnemar_p_values = []


for comparison in COMPARISONS:

    name = comparison[
        "name"
    ]


    system_a = comparison[
        "system_a"
    ]


    system_b = comparison[
        "system_b"
    ]


    correct_a = (
        correctness[
            system_a
        ]
    )


    correct_b = (
        correctness[
            system_b
        ]
    )


    both_correct = int(
        np.sum(
            correct_a
            &
            correct_b
        )
    )


    a_only_correct = int(
        np.sum(
            correct_a
            &
            ~correct_b
        )
    )


    b_only_correct = int(
        np.sum(
            ~correct_a
            &
            correct_b
        )
    )


    both_wrong = int(
        np.sum(
            ~correct_a
            &
            ~correct_b
        )
    )


    mcnemar_p = exact_mcnemar_p(
        a_only_correct,
        b_only_correct
    )


    raw_mcnemar_p_values.append(
        mcnemar_p
    )


    # Observed accuracy difference

    accuracy_a = float(
        np.mean(
            correct_a
        )
    )


    accuracy_b = float(
        np.mean(
            correct_b
        )
    )


    accuracy_difference = (
        accuracy_b
        -
        accuracy_a
    )


    # Paired bootstrap accuracy difference

    bootstrap_accuracy_difference = (

        bootstrap_accuracy[
            system_b
        ]

        -

        bootstrap_accuracy[
            system_a
        ]
    )


    accuracy_ci_low = float(
        np.percentile(
            bootstrap_accuracy_difference,
            2.5
        )
    )


    accuracy_ci_high = float(
        np.percentile(
            bootstrap_accuracy_difference,
            97.5
        )
    )


    # Observed macro-F1 difference

    macro_f1_a = macro_f1(
        gold,
        predictions[
            system_a
        ]
    )


    macro_f1_b = macro_f1(
        gold,
        predictions[
            system_b
        ]
    )


    macro_f1_difference = (
        macro_f1_b
        -
        macro_f1_a
    )


    # Paired bootstrap macro-F1 difference
   
    bootstrap_macro_difference = (

        bootstrap_macro_f1_values[
            system_b
        ]

        -

        bootstrap_macro_f1_values[
            system_a
        ]
    )


    macro_ci_low = float(
        np.percentile(
            bootstrap_macro_difference,
            2.5
        )
    )


    macro_ci_high = float(
        np.percentile(
            bootstrap_macro_difference,
            97.5
        )
    )


    # Paired win ratio

    if a_only_correct == 0:

        paired_odds_ratio = None

    else:

        paired_odds_ratio = (
            b_only_correct
            /
            a_only_correct
        )


    comparison_results.append(
        {

            "name":
                name,

            "system_a":
                system_a,

            "system_b":
                system_b,

            "difference_direction":
                "system_b_minus_system_a",

            "paired_correctness": {

                "both_correct":
                    both_correct,

                "system_a_only_correct":
                    a_only_correct,

                "system_b_only_correct":
                    b_only_correct,

                "both_wrong":
                    both_wrong
            },

            "accuracy": {

                "system_a":
                    accuracy_a,

                "system_b":
                    accuracy_b,

                "difference":
                    accuracy_difference,

                "bootstrap_95_ci": [
                    accuracy_ci_low,
                    accuracy_ci_high
                ]
            },

            "macro_f1": {

                "system_a":
                    macro_f1_a,

                "system_b":
                    macro_f1_b,

                "difference":
                    macro_f1_difference,

                "bootstrap_95_ci": [
                    macro_ci_low,
                    macro_ci_high
                ]
            },

            "mcnemar": {

                "discordant_pairs":
                    (
                        a_only_correct
                        +
                        b_only_correct
                    ),

                "exact_p_raw":
                    mcnemar_p,

                "exact_p_holm":
                    None
            },

            "paired_odds_ratio_b_over_a":
                paired_odds_ratio
        }
    )


# 16. HOLM CORRECTION

holm_p_values = holm_adjust(
    raw_mcnemar_p_values
)


for result, adjusted_p in zip(
    comparison_results,
    holm_p_values
):

    result[
        "mcnemar"
    ][
        "exact_p_holm"
    ] = adjusted_p


# 17. PRINT RESULTS

for result in comparison_results:

    system_a = result[
        "system_a"
    ]


    system_b = result[
        "system_b"
    ]


    paired = result[
        "paired_correctness"
    ]


    accuracy_result = result[
        "accuracy"
    ]


    macro_result = result[
        "macro_f1"
    ]


    mcnemar = result[
        "mcnemar"
    ]


    print(
        "\n"
        + "=" * 70
    )


    print(
        f"{SYSTEM_DISPLAY_NAMES[system_b]} "
        f"vs "
        f"{SYSTEM_DISPLAY_NAMES[system_a]}"
    )


    print(
        "=" * 70
    )


    print(
        "Paired correctness"
    )


    print(
        f"  Both correct         : "
        f"{paired['both_correct']}"
    )


    print(
        f"  {SYSTEM_DISPLAY_NAMES[system_a]} only : "
        f"{paired['system_a_only_correct']}"
    )


    print(
        f"  {SYSTEM_DISPLAY_NAMES[system_b]} only : "
        f"{paired['system_b_only_correct']}"
    )


    print(
        f"  Both wrong           : "
        f"{paired['both_wrong']}"
    )


    print(
        "\nAccuracy"
    )


    print(
        f"  Difference           : "
        f"{accuracy_result['difference'] * 100:+.2f} pp"
    )


    print(
        f"  Paired bootstrap CI  : "
        f"["
        f"{accuracy_result['bootstrap_95_ci'][0] * 100:+.2f}, "
        f"{accuracy_result['bootstrap_95_ci'][1] * 100:+.2f}"
        f"] pp"
    )


    print(
        "\nMacro-F1"
    )


    print(
        f"  Difference           : "
        f"{macro_result['difference']:+.4f}"
    )


    print(
        f"  Paired bootstrap CI  : "
        f"["
        f"{macro_result['bootstrap_95_ci'][0]:+.4f}, "
        f"{macro_result['bootstrap_95_ci'][1]:+.4f}"
        f"]"
    )


    print(
        "\nExact McNemar"
    )


    print(
        f"  Raw p                : "
        f"{mcnemar['exact_p_raw']:.8g}"
    )


    print(
        f"  Holm-adjusted p      : "
        f"{mcnemar['exact_p_holm']:.8g}"
    )


# 18. SIGNIFICANCE SUMMARY

print(
    "\n[5] HOLM-CORRECTED SIGNIFICANCE SUMMARY"
)


print(
    f"{'Comparison':<32}"
    f"{'Acc diff':>12}"
    f"{'McNemar p':>16}"
    f"{'Holm p':>16}"
    f"{'Sig .05':>10}"
)


print(
    "-" * 86
)


for result in comparison_results:

    name = result[
        "name"
    ]


    accuracy_difference = (
        result[
            "accuracy"
        ][
            "difference"
        ]
    )


    raw_p = (
        result[
            "mcnemar"
        ][
            "exact_p_raw"
        ]
    )


    holm_p = (
        result[
            "mcnemar"
        ][
            "exact_p_holm"
        ]
    )


    significant = (
        holm_p
        <
        0.05
    )


    print(
        f"{name:<32}"
        f"{accuracy_difference * 100:>+11.2f}%"
        f"{raw_p:>16.6g}"
        f"{holm_p:>16.6g}"
        f"{str(significant):>10}"
    )


# 19. SAVE STATISTICAL RESULTS

output = {

    "split":
        "test",

    "analysis":
        "confirmatory_paired_decision_statistics",

    "questions":
        500,

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

    "mcnemar": {

        "method":
            "exact_binomial_two_sided",

        "multiple_testing":
            "Holm",

        "family_size":
            len(
                COMPARISONS
            )
    },

    "comparisons":
        comparison_results
}


save_json(
    output,
    OUTPUT_PATH
)


# 20. FINAL OUTPUT

print(
    "\n[6] FILE SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 39 COMPLETE - "
    "FINAL PAIRED DECISION STATISTICS"
)


print(
    "=" * 78
)


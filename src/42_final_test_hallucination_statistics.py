import hashlib
import json
import math

from pathlib import Path

import numpy as np

from scipy.stats import wilcoxon


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_hallucination_judgment_manifest.json"
)


RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_hallucination_v3_results.json"
)


SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_hallucination_v3_summary.json"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_hallucination_statistics.json"
)


# 2. SETTINGS

BOOTSTRAP_SAMPLES = 10_000

RANDOM_SEED = 42


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


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

# All effect differences are:

# system_b - system_a
# Therefore:
# Negative hallucination difference = lower hallucination for system_b.


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


# 5. EXACT MCNEMAR TEST

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


# 7. START

print("=" * 78)

print(
    "SECTION 42 - FINAL TEST PAIRED HALLUCINATION STATISTICS"
)

print("=" * 78)


# 8. VERIFY FROZEN HALLUCINATION JUDGMENTS

print(
    "\n[1] VERIFYING FROZEN HALLUCINATION JUDGMENTS"
)


freeze_manifest = load_json(
    FREEZE_MANIFEST_PATH
)


if (
    freeze_manifest[
        "status"
    ]
    !=
    "FROZEN_BEFORE_HALLUCINATION_STATISTICS"
):

    raise RuntimeError(
        "Hallucination judgments were not "
        "properly frozen before statistics."
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
    f"Frozen artifacts checked : "
    f"{len(freeze_manifest['sha256'])}"
)


print(
    f"Changed artifacts        : "
    f"{len(changed_files)}"
)


print(
    f"Missing artifacts        : "
    f"{len(missing_files)}"
)


if (
    changed_files
    or
    missing_files
):

    raise RuntimeError(
        "Frozen hallucination artifacts "
        "have changed."
    )


print(
    "PASS: Frozen V3 judgments "
    "remain unchanged."
)


# 9. LOAD JUDGMENTS

print(
    "\n[2] LOADING FINAL HALLUCINATION RESULTS"
)


results_data = load_json(
    RESULTS_PATH
)


summary_data = load_json(
    SUMMARY_PATH
)


judgments = (
    results_data[
        "judgments"
    ]
)


if len(
    judgments
) != 500:

    raise ValueError(
        "Expected exactly 500 judgments."
    )


print(
    f"Paired questions : "
    f"{len(judgments)}"
)


# 10. BUILD PER-QUESTION ARRAYS

print(
    "\n[3] BUILDING QUESTION-LEVEL CLAIM ARRAYS"
)


total_claims = {}

hallucinated_claims = {}

answer_rates = {}

zero_claim = {}


for system_name in SYSTEM_ORDER:

    system_totals = []

    system_hallucinated = []


    for judgment in judgments:

        result = (
            judgment[
                "systems"
            ][
                system_name
            ]
        )


        counts = (
            result[
                "claim_counts"
            ]
        )


        total = (
            counts[
                "total"
            ]
        )


        hallucinated = (
            counts[
                "unsupported"
            ]
            +
            counts[
                "contradicted"
            ]
        )


        if (
            hallucinated
            >
            total
        ):

            raise ValueError(
                "Hallucinated claim count "
                "cannot exceed total claims."
            )


        system_totals.append(
            total
        )


        system_hallucinated.append(
            hallucinated
        )


    total_claims[
        system_name
    ] = np.asarray(
        system_totals,
        dtype=np.int32
    )


    hallucinated_claims[
        system_name
    ] = np.asarray(
        system_hallucinated,
        dtype=np.int32
    )


    rates = np.full(
        len(
            judgments
        ),
        np.nan,
        dtype=float
    )


    nonzero_mask = (
        total_claims[
            system_name
        ]
        >
        0
    )


    rates[
        nonzero_mask
    ] = (

        hallucinated_claims[
            system_name
        ][
            nonzero_mask
        ]

        /

        total_claims[
            system_name
        ][
            nonzero_mask
        ]
    )


    answer_rates[
        system_name
    ] = rates


    zero_claim[
        system_name
    ] = (
        ~nonzero_mask
    )


    print(
        f"{system_name:<22}"
        f" claims={int(np.sum(total_claims[system_name])):<5}"
        f" zero_answers={int(np.sum(zero_claim[system_name]))}"
    )


# 11. VERIFY RECOMPUTED MICRO RATES

print(
    "\n[4] VERIFYING SECTION 40 MICRO RATES"
)


observed_micro_rates = {}


for system_name in SYSTEM_ORDER:

    numerator = int(
        np.sum(
            hallucinated_claims[
                system_name
            ]
        )
    )


    denominator = int(
        np.sum(
            total_claims[
                system_name
            ]
        )
    )


    rate = (
        numerator
        /
        denominator
    )


    observed_micro_rates[
        system_name
    ] = rate


    frozen_rate = (
        summary_data[
            "system_summaries"
        ][
            system_name
        ][
            "micro_claim_hallucination_rate"
        ]
    )


    difference = abs(
        rate
        -
        frozen_rate
    )


    print(
        f"{system_name:<22}"
        f" recomputed={rate:.6f}"
        f" frozen={frozen_rate:.6f}"
    )


    if difference > 1e-12:

        raise RuntimeError(
            f"Micro hallucination rate "
            f"mismatch for {system_name}."
        )


print(
    "PASS: Claim arrays reproduce "
    "the frozen Section 40 rates."
)



# 12. QUESTION-CLUSTER BOOTSTRAP

# The QUESTION is the resampling unit.

# All claims belonging to a sampled question remain together.

# This avoids treating claims from the same answer/question as statistically independent.



print(
    "\n[5] QUESTION-CLUSTER BOOTSTRAP"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


bootstrap_indices = rng.integers(

    low=0,

    high=len(
        judgments
    ),

    size=(
        BOOTSTRAP_SAMPLES,
        len(
            judgments
        )
    ),

    dtype=np.int32
)


bootstrap_micro_rates = {}


for system_name in SYSTEM_ORDER:

    sampled_hallucinated = np.sum(

        hallucinated_claims[
            system_name
        ][
            bootstrap_indices
        ],

        axis=1
    )


    sampled_total = np.sum(

        total_claims[
            system_name
        ][
            bootstrap_indices
        ],

        axis=1
    )


    bootstrap_micro_rates[
        system_name
    ] = np.divide(

        sampled_hallucinated,

        sampled_total,

        out=np.full(
            BOOTSTRAP_SAMPLES,
            np.nan,
            dtype=float
        ),

        where=(
            sampled_total
            >
            0
        )
    )


print(
    f"Bootstrap samples : "
    f"{BOOTSTRAP_SAMPLES}"
)


print(
    f"Resampling unit   : question"
)


print(
    f"Random seed       : "
    f"{RANDOM_SEED}"
)


# 13. RUN PAIRED COMPARISONS

print(
    "\n[6] PAIRED HALLUCINATION COMPARISONS"
)


comparison_results = []

wilcoxon_raw_p_values = []

zero_claim_raw_p_values = []


for comparison in COMPARISONS:

    name = (
        comparison[
            "name"
        ]
    )


    system_a = (
        comparison[
            "system_a"
        ]
    )


    system_b = (
        comparison[
            "system_b"
        ]
    )


    # PRIMARY MICRO CLAIM RATE

    micro_a = (
        observed_micro_rates[
            system_a
        ]
    )


    micro_b = (
        observed_micro_rates[
            system_b
        ]
    )


    micro_difference = (
        micro_b
        -
        micro_a
    )


    bootstrap_difference = (

        bootstrap_micro_rates[
            system_b
        ]

        -

        bootstrap_micro_rates[
            system_a
        ]
    )


    micro_ci_low = float(
        np.nanpercentile(
            bootstrap_difference,
            2.5
        )
    )


    micro_ci_high = float(
        np.nanpercentile(
            bootstrap_difference,
            97.5
        )
    )


    if micro_a > 0:

        relative_reduction = (
            (
                micro_a
                -
                micro_b
            )
            /
            micro_a
        )

    else:

        relative_reduction = None


    # PAIRED ANSWER LEVEL ANALYSIS
    # Only questions with scored claims in BOTH
    # systems can enter this comparison.


    rate_a = (
        answer_rates[
            system_a
        ]
    )


    rate_b = (
        answer_rates[
            system_b
        ]
    )


    valid_pair_mask = (
        ~np.isnan(
            rate_a
        )
        &
        ~np.isnan(
            rate_b
        )
    )


    paired_a = (
        rate_a[
            valid_pair_mask
        ]
    )


    paired_b = (
        rate_b[
            valid_pair_mask
        ]
    )


    paired_differences = (
        paired_b
        -
        paired_a
    )


    paired_n = len(
        paired_differences
    )


    mean_answer_difference = float(
        np.mean(
            paired_differences
        )
    )


    median_answer_difference = float(
        np.median(
            paired_differences
        )
    )


    # Bootstrap answer-level mean difference

    pair_rng = np.random.default_rng(
        RANDOM_SEED
        +
        COMPARISONS.index(
            comparison
        )
        +
        1000
    )


    answer_bootstrap_indices = (
        pair_rng.integers(

            low=0,

            high=paired_n,

            size=(
                BOOTSTRAP_SAMPLES,
                paired_n
            ),

            dtype=np.int32
        )
    )


    answer_bootstrap_differences = np.mean(

        paired_differences[
            answer_bootstrap_indices
        ],

        axis=1
    )


    answer_ci_low = float(
        np.percentile(
            answer_bootstrap_differences,
            2.5
        )
    )


    answer_ci_high = float(
        np.percentile(
            answer_bootstrap_differences,
            97.5
        )
    )


    # Wilcoxon signed rank

    if np.allclose(
        paired_differences,
        0.0
    ):

        wilcoxon_statistic = 0.0

        wilcoxon_p = 1.0


    else:

        wilcoxon_result = wilcoxon(

            paired_b,

            paired_a,

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


    wilcoxon_raw_p_values.append(
        wilcoxon_p
    )


    # ZERO-CLAIM MCNEMAR

    zero_a = (
        zero_claim[
            system_a
        ]
    )


    zero_b = (
        zero_claim[
            system_b
        ]
    )


    both_zero = int(
        np.sum(
            zero_a
            &
            zero_b
        )
    )


    a_zero_only = int(
        np.sum(
            zero_a
            &
            ~zero_b
        )
    )


    b_zero_only = int(
        np.sum(
            ~zero_a
            &
            zero_b
        )
    )


    neither_zero = int(
        np.sum(
            ~zero_a
            &
            ~zero_b
        )
    )


    zero_claim_p = (
        exact_mcnemar_p(
            a_zero_only,
            b_zero_only
        )
    )


    zero_claim_raw_p_values.append(
        zero_claim_p
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

            "primary_micro_claim_hallucination": {

                "system_a":
                    micro_a,

                "system_b":
                    micro_b,

                "absolute_difference":
                    micro_difference,

                "question_cluster_bootstrap_95_ci": [
                    micro_ci_low,
                    micro_ci_high
                ],

                "relative_reduction_b_vs_a":
                    relative_reduction
            },

            "paired_answer_level": {

                "n_pairs_with_scored_claims":
                    paired_n,

                "mean_difference":
                    mean_answer_difference,

                "median_difference":
                    median_answer_difference,

                "bootstrap_mean_difference_95_ci": [
                    answer_ci_low,
                    answer_ci_high
                ],

                "wilcoxon_statistic":
                    wilcoxon_statistic,

                "wilcoxon_p_raw":
                    wilcoxon_p,

                "wilcoxon_p_holm":
                    None
            },

            "zero_claim_analysis": {

                "both_zero":
                    both_zero,

                "system_a_zero_only":
                    a_zero_only,

                "system_b_zero_only":
                    b_zero_only,

                "neither_zero":
                    neither_zero,

                "mcnemar_p_raw":
                    zero_claim_p,

                "mcnemar_p_holm":
                    None
            }
        }
    )


# 14. HOLM CORRECTIONS

# Separate multiplicity families:
# 1. five Wilcoxon hallucination comparisons
# 2. five zero-claim McNemar comparisons


wilcoxon_holm = holm_adjust(
    wilcoxon_raw_p_values
)


zero_claim_holm = holm_adjust(
    zero_claim_raw_p_values
)


for index, result in enumerate(
    comparison_results
):

    result[
        "paired_answer_level"
    ][
        "wilcoxon_p_holm"
    ] = (
        wilcoxon_holm[
            index
        ]
    )


    result[
        "zero_claim_analysis"
    ][
        "mcnemar_p_holm"
    ] = (
        zero_claim_holm[
            index
        ]
    )


# 15. PRINT PRIMARY MICRO RESULTS

print(
    "\n[7] PRIMARY MICRO-CLAIM RESULTS"
)


for result in comparison_results:

    system_a = (
        result[
            "system_a"
        ]
    )


    system_b = (
        result[
            "system_b"
        ]
    )


    primary = (
        result[
            "primary_micro_claim_hallucination"
        ]
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        f"{SYSTEM_DISPLAY_NAMES[system_b]} "
        f"vs "
        f"{SYSTEM_DISPLAY_NAMES[system_a]}"
    )


    print(
        "=" * 72
    )


    print(
        f"{SYSTEM_DISPLAY_NAMES[system_a]} rate : "
        f"{primary['system_a']:.4f}"
    )


    print(
        f"{SYSTEM_DISPLAY_NAMES[system_b]} rate : "
        f"{primary['system_b']:.4f}"
    )


    print(
        f"Difference (B - A) : "
        f"{primary['absolute_difference']:+.4f}"
    )


    print(
        f"Difference in pp    : "
        f"{primary['absolute_difference'] * 100:+.2f}"
    )


    print(
        "Cluster bootstrap CI: "
        f"["
        f"{primary['question_cluster_bootstrap_95_ci'][0] * 100:+.2f}, "
        f"{primary['question_cluster_bootstrap_95_ci'][1] * 100:+.2f}"
        f"] pp"
    )


    relative_reduction = (
        primary[
            "relative_reduction_b_vs_a"
        ]
    )


    if relative_reduction is not None:

        print(
            f"Relative reduction   : "
            f"{relative_reduction * 100:+.2f}%"
        )


# 16. PAIRED ANSWER-LEVEL RESULTS

print(
    "\n[8] PAIRED ANSWER-LEVEL RESULTS"
)


for result in comparison_results:

    system_a = (
        result[
            "system_a"
        ]
    )


    system_b = (
        result[
            "system_b"
        ]
    )


    paired = (
        result[
            "paired_answer_level"
        ]
    )


    print(
        "\n"
        f"{SYSTEM_DISPLAY_NAMES[system_b]} "
        f"vs "
        f"{SYSTEM_DISPLAY_NAMES[system_a]}"
    )


    print(
        f"  Paired N             : "
        f"{paired['n_pairs_with_scored_claims']}"
    )


    print(
        f"  Mean difference      : "
        f"{paired['mean_difference']:+.4f}"
    )


    print(
        f"  Bootstrap 95% CI     : "
        f"["
        f"{paired['bootstrap_mean_difference_95_ci'][0]:+.4f}, "
        f"{paired['bootstrap_mean_difference_95_ci'][1]:+.4f}"
        f"]"
    )


    print(
        f"  Wilcoxon raw p       : "
        f"{paired['wilcoxon_p_raw']:.8g}"
    )


    print(
        f"  Wilcoxon Holm p      : "
        f"{paired['wilcoxon_p_holm']:.8g}"
    )


# 17. ZERO-CLAIM ANALYSIS

print(
    "\n[9] ZERO-CLAIM PAIRED ANALYSIS"
)


for result in comparison_results:

    system_a = (
        result[
            "system_a"
        ]
    )


    system_b = (
        result[
            "system_b"
        ]
    )


    zero = (
        result[
            "zero_claim_analysis"
        ]
    )


    print(
        "\n"
        f"{SYSTEM_DISPLAY_NAMES[system_b]} "
        f"vs "
        f"{SYSTEM_DISPLAY_NAMES[system_a]}"
    )


    print(
        f"  Both zero            : "
        f"{zero['both_zero']}"
    )


    print(
        f"  {SYSTEM_DISPLAY_NAMES[system_a]} zero only : "
        f"{zero['system_a_zero_only']}"
    )


    print(
        f"  {SYSTEM_DISPLAY_NAMES[system_b]} zero only : "
        f"{zero['system_b_zero_only']}"
    )


    print(
        f"  Neither zero         : "
        f"{zero['neither_zero']}"
    )


    print(
        f"  McNemar raw p        : "
        f"{zero['mcnemar_p_raw']:.8g}"
    )


    print(
        f"  McNemar Holm p       : "
        f"{zero['mcnemar_p_holm']:.8g}"
    )


# 18. COMPACT SCORECARD

print(
    "\n[10] HALLUCINATION INFERENCE SCORECARD"
)


print(
    f"{'Comparison':<30}"
    f"{'Micro diff':>12}"
    f"{'95% CI low':>13}"
    f"{'95% CI high':>14}"
    f"{'Wilcox Holm':>14}"
)


print(
    "-" * 83
)


for result in comparison_results:

    primary = (
        result[
            "primary_micro_claim_hallucination"
        ]
    )


    paired = (
        result[
            "paired_answer_level"
        ]
    )


    print(
        f"{result['name']:<30}"
        f"{primary['absolute_difference'] * 100:>+11.2f}%"
        f"{primary['question_cluster_bootstrap_95_ci'][0] * 100:>+12.2f}%"
        f"{primary['question_cluster_bootstrap_95_ci'][1] * 100:>+13.2f}%"
        f"{paired['wilcoxon_p_holm']:>14.6g}"
    )


# 19. SAVE RESULTS

output = {

    "split":
        "test",

    "analysis":
        "confirmatory_paired_hallucination_statistics",

    "questions":
        500,

    "primary_metric":
        (
            "micro claim hallucination rate"
        ),

    "primary_metric_definition":
        (
            "(unsupported + contradicted) "
            "/ scored factual claims"
        ),

    "bootstrap": {

        "samples":
            BOOTSTRAP_SAMPLES,

        "seed":
            RANDOM_SEED,

        "resampling_unit":
            "question",

        "paired":
            True,

        "confidence_level":
            0.95
    },

    "answer_level_test": {

        "method":
            "Wilcoxon signed-rank",

        "multiple_testing":
            "Holm",

        "family_size":
            len(
                COMPARISONS
            )
    },

    "zero_claim_test": {

        "method":
            "exact McNemar",

        "multiple_testing":
            "Holm",

        "family_size":
            len(
                COMPARISONS
            )
    },

    "observed_micro_rates":
        observed_micro_rates,

    "comparisons":
        comparison_results
}


save_json(
    output,
    OUTPUT_PATH
)


# 20. FINAL OUTPUT

print(
    "\n[11] FILE SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 42 COMPLETE - "
    "FINAL PAIRED HALLUCINATION STATISTICS"
)


print(
    "=" * 78
)
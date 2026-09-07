import csv
import hashlib
import json
from collections import Counter, defaultdict
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


DECISION_RESULTS_PATH = (
    EVALUATION_DIR
    / "final_test_decision_results.json"
)


HALLUCINATION_RESULTS_PATH = (
    EVALUATION_DIR
    / "final_test_hallucination_v3_results.json"
)


RETRIEVAL_RESULTS_PATH = (
    EVALUATION_DIR
    / "final_test_retrieval_results.json"
)


INTEGRATED_RESULTS_PATH = (
    EVALUATION_DIR
    / "integrated_final_results.json"
)


OUTPUT_JSON_PATH = (
    EVALUATION_DIR
    / "exploratory_error_analysis.json"
)


OUTPUT_CSV_PATH = (
    EVALUATION_DIR
    / "exploratory_error_analysis_per_question.csv"
)


# 2. SYSTEMS

SYSTEM_ORDER = [

    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


RAG_SYSTEMS = [

    "basic_rag",
    "advanced_rag"
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


def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:

        return None


    return (
        numerator
        /
        denominator
    )


def percentage(
    numerator,
    denominator
):

    value = safe_divide(
        numerator,
        denominator
    )


    if value is None:

        return None


    return (
        value
        *
        100.0
    )


# 4. START

print("=" * 78)

print(
    "SECTION 47 - EXPLORATORY FINAL TEST ERROR ANALYSIS"
)

print("=" * 78)


print(
    """
IMPORTANT:

This is an EXPLORATORY analysis.

It uses only already-frozen outputs from the
confirmatory experiment.

It does NOT:
- create new LLM judgments
- change any hallucination labels
- modify retrieval outputs
- tune Basic or Advanced RAG
- introduce a new confirmatory endpoint
- perform new significance testing

The purpose is to understand where residual
errors occurred.
"""
)


# 5. LOAD FROZEN RESULTS

print(
    "[1] LOADING FROZEN FINAL TEST RESULTS"
)


decision_results = load_json(
    DECISION_RESULTS_PATH
)


hallucination_data = load_json(
    HALLUCINATION_RESULTS_PATH
)


retrieval_data = load_json(
    RETRIEVAL_RESULTS_PATH
)


integrated_results = load_json(
    INTEGRATED_RESULTS_PATH
)


input_paths = [

    DECISION_RESULTS_PATH,
    HALLUCINATION_RESULTS_PATH,
    RETRIEVAL_RESULTS_PATH,
    INTEGRATED_RESULTS_PATH
]


for path in input_paths:

    print(
        "PASS: "
        f"{relative_path(path)}"
    )


# 6. VALIDATE BASIC SHAPES

print(
    "\n[2] VALIDATING INPUT SHAPES"
)


if len(
    decision_results
) != 500:

    raise ValueError(
        "Expected 500 decision records."
    )


hallucination_judgments = (
    hallucination_data[
        "judgments"
    ]
)


if len(
    hallucination_judgments
) != 500:

    raise ValueError(
        "Expected 500 hallucination judgments."
    )


if (
    retrieval_data[
        "questions"
    ]
    != 500
):

    raise ValueError(
        "Expected 500 retrieval questions."
    )


if (
    integrated_results[
        "integration_only"
    ]
    is not True
):

    raise RuntimeError(
        "Integrated result artifact "
        "does not have expected status."
    )


print(
    "PASS: All exploratory inputs "
    "contain 500 final-test questions."
)


# 7. INDEX DECISION RESULTS

print(
    "\n[3] INDEXING DECISION RESULTS"
)


decision_by_id = {}


for record in decision_results:

    question_id = str(
        record[
            "question_id"
        ]
    )


    if question_id in decision_by_id:

        raise ValueError(
            f"Duplicate decision question: "
            f"{question_id}"
        )


    decision_by_id[
        question_id
    ] = record


print(
    f"Decision questions indexed : "
    f"{len(decision_by_id)}"
)


# 8. INDEX HALLUCINATION RESULTS

print(
    "\n[4] INDEXING FROZEN V3 JUDGMENTS"
)


hallucination_by_id = {}


for record in hallucination_judgments:

    question_id = str(
        record[
            "question_id"
        ]
    )


    if question_id in hallucination_by_id:

        raise ValueError(
            f"Duplicate hallucination question: "
            f"{question_id}"
        )


    systems = (
        record[
            "systems"
        ]
    )


    for system in SYSTEM_ORDER:

        if system not in systems:

            raise KeyError(
                f"{system} missing for "
                f"question {question_id}."
            )


    hallucination_by_id[
        question_id
    ] = record


print(
    f"V3 questions indexed       : "
    f"{len(hallucination_by_id)}"
)


# 9. INDEX RETRIEVAL RESULTS

print(
    "\n[5] INDEXING RETRIEVAL RESULTS"
)


retrieval_by_system = {}


for system in RAG_SYSTEMS:

    records = (
        retrieval_data[
            "systems"
        ][
            system
        ]
    )


    indexed = {}


    for record in records:

        question_id = str(
            record[
                "question_id"
            ]
        )


        if question_id in indexed:

            raise ValueError(
                f"Duplicate retrieval question "
                f"for {system}: {question_id}"
            )


        indexed[
            question_id
        ] = record


    if len(
        indexed
    ) != 500:

        raise ValueError(
            f"{system} retrieval index "
            "does not contain 500 questions."
        )


    retrieval_by_system[
        system
    ] = indexed


    print(
        f"{DISPLAY_NAMES[system]:<22}: "
        f"{len(indexed)}"
    )


# 10. VERIFY QUESTION ALIGNMENT

print(
    "\n[6] VERIFYING QUESTION ALIGNMENT"
)


question_ids = sorted(
    decision_by_id.keys()
)


reference_set = set(
    question_ids
)


if (
    set(
        hallucination_by_id.keys()
    )
    !=
    reference_set
):

    raise RuntimeError(
        "Hallucination question IDs "
        "do not match decisions."
    )


for system in RAG_SYSTEMS:

    if (
        set(
            retrieval_by_system[
                system
            ].keys()
        )
        !=
        reference_set
    ):

        raise RuntimeError(
            f"{system} retrieval IDs "
            "do not match decisions."
        )


print(
    "PASS: All analyses align on the "
    "same 500 question IDs."
)


# 11. EXTRACT CLAIM LEVEL V3 COUNTS

def extract_claim_metrics(
    system_record
):

    claims = (
        system_record.get(
            "claims",
            []
        )
        or
        []
    )


    excluded = (
        system_record.get(
            "excluded_statements",
            []
        )
        or
        []
    )


    supported = 0
    unsupported = 0
    contradicted = 0


    for claim in claims:

        label = str(
            claim.get(
                "label",
                ""
            )
        ).strip().lower()


        if label == "supported":

            supported += 1


        elif label == "unsupported":

            unsupported += 1


        elif label == "contradicted":

            contradicted += 1


        else:

            raise ValueError(
                "Unexpected V3 claim label: "
                f"{label}"
            )


    total_claims = len(
        claims
    )


    hallucinated_claims = (
        unsupported
        +
        contradicted
    )


    if total_claims > 0:

        hallucination_rate = (
            hallucinated_claims
            /
            total_claims
        )


    else:

        hallucination_rate = None


    return {

        "total_claims":
            total_claims,

        "supported_claims":
            supported,

        "unsupported_claims":
            unsupported,

        "contradicted_claims":
            contradicted,

        "hallucinated_claims":
            hallucinated_claims,

        "has_hallucinated_claim":
            hallucinated_claims > 0,

        "zero_claim_answer":
            total_claims == 0,

        "hallucination_rate":
            hallucination_rate,

        "excluded_statement_count":
            len(
                excluded
            )
    }



# 12. BUILD QUESTION LEVEL EXPLORATORY DATASET


print(
    "\n[7] BUILDING QUESTION-LEVEL ERROR DATASET"
)


per_question = []


for question_id in question_ids:

    decision_record = (
        decision_by_id[
            question_id
        ]
    )


    hallucination_record = (
        hallucination_by_id[
            question_id
        ]
    )


    row = {

        "question_id":
            question_id,

        "gold_decision":
            decision_record[
                "gold_decision"
            ]
    }


    # Decision + hallucination fields for all systems

    for system in SYSTEM_ORDER:

        decision_system = (
            decision_record[
                system
            ]
        )


        hall_system = (
            hallucination_record[
                "systems"
            ][
                system
            ]
        )


        claim_metrics = extract_claim_metrics(
            hall_system
        )


        row[
            f"{system}_predicted_decision"
        ] = (
            decision_system[
                "predicted_decision"
            ]
        )


        row[
            f"{system}_decision_correct"
        ] = bool(
            decision_system[
                "correct"
            ]
        )


        row[
            f"{system}_total_claims"
        ] = (
            claim_metrics[
                "total_claims"
            ]
        )


        row[
            f"{system}_supported_claims"
        ] = (
            claim_metrics[
                "supported_claims"
            ]
        )


        row[
            f"{system}_unsupported_claims"
        ] = (
            claim_metrics[
                "unsupported_claims"
            ]
        )


        row[
            f"{system}_contradicted_claims"
        ] = (
            claim_metrics[
                "contradicted_claims"
            ]
        )


        row[
            f"{system}_hallucinated_claims"
        ] = (
            claim_metrics[
                "hallucinated_claims"
            ]
        )


        row[
            f"{system}_has_hallucinated_claim"
        ] = (
            claim_metrics[
                "has_hallucinated_claim"
            ]
        )


        row[
            f"{system}_zero_claim_answer"
        ] = (
            claim_metrics[
                "zero_claim_answer"
            ]
        )


        row[
            f"{system}_answer_hallucination_rate"
        ] = (
            claim_metrics[
                "hallucination_rate"
            ]
        )


        row[
            f"{system}_excluded_statement_count"
        ] = (
            claim_metrics[
                "excluded_statement_count"
            ]
        )


    # Retrieval fields for Basic / Advanced

    for system in RAG_SYSTEMS:

        retrieval_record = (
            retrieval_by_system[
                system
            ][
                question_id
            ]
        )


        row[
            f"{system}_source_hit_at_1"
        ] = bool(
            retrieval_record[
                "source_hits"
            ][
                "hit_at_1"
            ]
        )


        row[
            f"{system}_source_hit_at_3"
        ] = bool(
            retrieval_record[
                "source_hits"
            ][
                "hit_at_3"
            ]
        )


        row[
            f"{system}_source_hit_at_5"
        ] = bool(
            retrieval_record[
                "source_hits"
            ][
                "hit_at_5"
            ]
        )


        row[
            f"{system}_source_hit_at_10"
        ] = bool(
            retrieval_record[
                "source_hits"
            ][
                "hit_at_10"
            ]
        )


        row[
            f"{system}_gold_recall_at_5"
        ] = float(
            retrieval_record[
                "gold_context_recall"
            ][
                "recall_at_5"
            ]
        )


        row[
            f"{system}_first_correct_source_rank"
        ] = (
            retrieval_record[
                "first_correct_source_rank_at_10"
            ]
        )


    # Rule based exploratory failure categories

    for system in RAG_SYSTEMS:

        decision_correct = (
            row[
                f"{system}_decision_correct"
            ]
        )


        hit_5 = (
            row[
                f"{system}_source_hit_at_5"
            ]
        )


        has_hallucination = (
            row[
                f"{system}_has_hallucinated_claim"
            ]
        )


        zero_claim = (
            row[
                f"{system}_zero_claim_answer"
            ]
        )


        if not hit_5:

            retrieval_status = (
                "source_miss_at_5"
            )


        else:

            retrieval_status = (
                "source_hit_at_5"
            )


        row[
            f"{system}_retrieval_status"
        ] = retrieval_status


        # Coarse error mechanism:
        
        # Retrieval failure: source not retrieved in generation top 5
        
        # Downstream error despite retrieval: source retrieved, but decision wrong
        # Hallucination despite retrieval: source retrieved, but V3 finds >=1
        #   unsupported/contradicted claim
        # Clean success: source hit, correct decision, no hallucinated factual claims
        # Zero claim cases are retained separately.

        if not hit_5:

            primary_error_category = (
                "retrieval_failure"
            )


        elif not decision_correct:

            primary_error_category = (
                "decision_error_despite_source_retrieval"
            )


        elif has_hallucination:

            primary_error_category = (
                "hallucination_despite_correct_decision_and_retrieval"
            )


        elif zero_claim:

            primary_error_category = (
                "correct_decision_zero_scored_claims"
            )


        else:

            primary_error_category = (
                "clean_success"
            )


        row[
            f"{system}_primary_error_category"
        ] = primary_error_category


    # Basic vs Advanced relationship

    basic_correct = (
        row[
            "basic_rag_decision_correct"
        ]
    )


    advanced_correct = (
        row[
            "advanced_rag_decision_correct"
        ]
    )


    if (
        basic_correct
        and
        advanced_correct
    ):

        decision_pair_category = (
            "both_correct"
        )


    elif (
        basic_correct
        and
        not advanced_correct
    ):

        decision_pair_category = (
            "basic_only_correct"
        )


    elif (
        advanced_correct
        and
        not basic_correct
    ):

        decision_pair_category = (
            "advanced_only_correct"
        )


    else:

        decision_pair_category = (
            "both_wrong"
        )


    row[
        "basic_advanced_decision_pair"
    ] = decision_pair_category


    basic_hit = (
        row[
            "basic_rag_source_hit_at_5"
        ]
    )


    advanced_hit = (
        row[
            "advanced_rag_source_hit_at_5"
        ]
    )


    if (
        basic_hit
        and
        advanced_hit
    ):

        retrieval_pair_category = (
            "both_hit"
        )


    elif (
        basic_hit
        and
        not advanced_hit
    ):

        retrieval_pair_category = (
            "basic_only_hit"
        )


    elif (
        advanced_hit
        and
        not basic_hit
    ):

        retrieval_pair_category = (
            "advanced_only_hit"
        )


    else:

        retrieval_pair_category = (
            "neither_hit"
        )


    row[
        "basic_advanced_retrieval_pair_at_5"
    ] = retrieval_pair_category


    basic_hall = (
        row[
            "basic_rag_has_hallucinated_claim"
        ]
    )


    advanced_hall = (
        row[
            "advanced_rag_has_hallucinated_claim"
        ]
    )


    if (
        basic_hall
        and
        advanced_hall
    ):

        hallucination_pair_category = (
            "both_have_hallucinated_claim"
        )


    elif (
        basic_hall
        and
        not advanced_hall
    ):

        hallucination_pair_category = (
            "basic_only_has_hallucinated_claim"
        )


    elif (
        advanced_hall
        and
        not basic_hall
    ):

        hallucination_pair_category = (
            "advanced_only_has_hallucinated_claim"
        )


    else:

        hallucination_pair_category = (
            "neither_has_hallucinated_claim"
        )


    row[
        "basic_advanced_hallucination_pair"
    ] = hallucination_pair_category


    per_question.append(
        row
    )


print(
    f"Question-level rows built : "
    f"{len(per_question)}"
)


# 13. VERIFY AGGREGATE DECISION COUNTS

print(
    "\n[8] VERIFYING AGAINST CONFIRMATORY RESULTS"
)


expected_correct = {

    "baseline_llm":
        161,

    "basic_rag":
        311,

    "advanced_rag":
        303,

    "gold_context_control":
        361
}


for system in SYSTEM_ORDER:

    calculated = sum(

        1

        for row
        in per_question

        if row[
            f"{system}_decision_correct"
        ]
    )


    if (
        calculated
        !=
        expected_correct[
            system
        ]
    ):

        raise RuntimeError(
            f"Decision count mismatch "
            f"for {system}: "
            f"{calculated}"
        )


    print(
        f"PASS: {DISPLAY_NAMES[system]:<22} "
        f"correct={calculated}"
    )


# 14. VERIFY CLAIM COUNTS

expected_claim_totals = {

    "baseline_llm":
        {
            "total":
                997,
            "hallucinated":
                609
        },

    "basic_rag":
        {
            "total":
                1484,
            "hallucinated":
                126
        },

    "advanced_rag":
        {
            "total":
                1414,
            "hallucinated":
                111
        },

    "gold_context_control":
        {
            "total":
                1581,
            "hallucinated":
                68
        }
}


for system in SYSTEM_ORDER:

    total_claims = sum(

        row[
            f"{system}_total_claims"
        ]

        for row
        in per_question
    )


    hallucinated_claims = sum(

        row[
            f"{system}_hallucinated_claims"
        ]

        for row
        in per_question
    )


    if (
        total_claims
        !=
        expected_claim_totals[
            system
        ][
            "total"
        ]
    ):

        raise RuntimeError(
            f"Total claim mismatch "
            f"for {system}."
        )


    if (
        hallucinated_claims
        !=
        expected_claim_totals[
            system
        ][
            "hallucinated"
        ]
    ):

        raise RuntimeError(
            f"Hallucinated claim mismatch "
            f"for {system}."
        )


    print(
        f"PASS: {DISPLAY_NAMES[system]:<22} "
        f"claims={total_claims} "
        f"hallucinated={hallucinated_claims}"
    )


# 15. RAG ERROR MECHANISM SUMMARY

print(
    "\n[9] RAG ERROR-MECHANISM SUMMARY"
)


rag_error_summary = {}


for system in RAG_SYSTEMS:

    category_counts = Counter(

        row[
            f"{system}_primary_error_category"
        ]

        for row
        in per_question
    )


    retrieval_misses = sum(

        1

        for row
        in per_question

        if not row[
            f"{system}_source_hit_at_5"
        ]
    )


    decision_errors = sum(

        1

        for row
        in per_question

        if not row[
            f"{system}_decision_correct"
        ]
    )


    decision_errors_with_hit = sum(

        1

        for row
        in per_question

        if (
            row[
                f"{system}_source_hit_at_5"
            ]
            and
            not row[
                f"{system}_decision_correct"
            ]
        )
    )


    decision_errors_with_miss = sum(

        1

        for row
        in per_question

        if (
            not row[
                f"{system}_source_hit_at_5"
            ]
            and
            not row[
                f"{system}_decision_correct"
            ]
        )
    )


    hallucination_answers = sum(

        1

        for row
        in per_question

        if row[
            f"{system}_has_hallucinated_claim"
        ]
    )


    hallucination_with_hit = sum(

        1

        for row
        in per_question

        if (
            row[
                f"{system}_source_hit_at_5"
            ]
            and
            row[
                f"{system}_has_hallucinated_claim"
            ]
        )
    )


    hallucination_with_miss = sum(

        1

        for row
        in per_question

        if (
            not row[
                f"{system}_source_hit_at_5"
            ]
            and
            row[
                f"{system}_has_hallucinated_claim"
            ]
        )
    )


    zero_claim_answers = sum(

        1

        for row
        in per_question

        if row[
            f"{system}_zero_claim_answer"
        ]
    )


    mean_gold_recall_correct = safe_divide(

        sum(
            row[
                f"{system}_gold_recall_at_5"
            ]

            for row
            in per_question

            if row[
                f"{system}_decision_correct"
            ]
        ),

        sum(
            1

            for row
            in per_question

            if row[
                f"{system}_decision_correct"
            ]
        )
    )


    mean_gold_recall_wrong = safe_divide(

        sum(
            row[
                f"{system}_gold_recall_at_5"
            ]

            for row
            in per_question

            if not row[
                f"{system}_decision_correct"
            ]
        ),

        sum(
            1

            for row
            in per_question

            if not row[
                f"{system}_decision_correct"
            ]
        )
    )


    rag_error_summary[
        system
    ] = {

        "questions":
            500,

        "retrieval_misses_at_5":
            retrieval_misses,

        "decision_errors":
            decision_errors,

        "decision_errors_with_source_hit_at_5":
            decision_errors_with_hit,

        "decision_errors_with_source_miss_at_5":
            decision_errors_with_miss,

        "proportion_of_decision_errors_despite_source_hit":
            safe_divide(
                decision_errors_with_hit,
                decision_errors
            ),

        "answers_with_at_least_one_hallucinated_claim":
            hallucination_answers,

        "hallucination_answers_with_source_hit_at_5":
            hallucination_with_hit,

        "hallucination_answers_with_source_miss_at_5":
            hallucination_with_miss,

        "proportion_of_hallucination_answers_despite_source_hit":
            safe_divide(
                hallucination_with_hit,
                hallucination_answers
            ),

        "zero_claim_answers":
            zero_claim_answers,

        "mean_gold_context_recall_at_5_when_decision_correct":
            mean_gold_recall_correct,

        "mean_gold_context_recall_at_5_when_decision_wrong":
            mean_gold_recall_wrong,

        "rule_based_primary_categories":
            dict(
                category_counts
            )
    }


    print(
        "\n"
        + DISPLAY_NAMES[
            system
        ]
    )


    print(
        f"  Retrieval misses @5          : "
        f"{retrieval_misses}/500"
    )


    print(
        f"  Decision errors              : "
        f"{decision_errors}/500"
    )


    print(
        f"  Decision errors despite hit  : "
        f"{decision_errors_with_hit}"
    )


    print(
        f"  Decision errors after miss   : "
        f"{decision_errors_with_miss}"
    )


    print(
        f"  Hallucination-answer cases   : "
        f"{hallucination_answers}"
    )


    print(
        f"  Hallucination despite hit    : "
        f"{hallucination_with_hit}"
    )


    print(
        f"  Hallucination after miss     : "
        f"{hallucination_with_miss}"
    )


    print(
        f"  Mean Gold R@5 | correct      : "
        f"{mean_gold_recall_correct:.4f}"
    )


    print(
        f"  Mean Gold R@5 | wrong        : "
        f"{mean_gold_recall_wrong:.4f}"
    )


    print(
        "  Rule-based categories:"
    )


    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"    {category:<48} "
            f"{count}"
        )


# 16. BASIC VS ADVANCED DISAGREEMENTS

print(
    "\n[10] BASIC VS ADVANCED DISAGREEMENTS"
)


decision_pair_counts = Counter(

    row[
        "basic_advanced_decision_pair"
    ]

    for row
    in per_question
)


retrieval_pair_counts = Counter(

    row[
        "basic_advanced_retrieval_pair_at_5"
    ]

    for row
    in per_question
)


hallucination_pair_counts = Counter(

    row[
        "basic_advanced_hallucination_pair"
    ]

    for row
    in per_question
)


print(
    "\nDecision correctness:"
)


for category, count in sorted(
    decision_pair_counts.items()
):

    print(
        f"  {category:<28} "
        f"{count}"
    )


print(
    "\nSource Hit@5:"
)


for category, count in sorted(
    retrieval_pair_counts.items()
):

    print(
        f"  {category:<28} "
        f"{count}"
    )


print(
    "\nAt least one V3 hallucinated claim:"
)


for category, count in sorted(
    hallucination_pair_counts.items()
):

    print(
        f"  {category:<38} "
        f"{count}"
    )


# 17. BASIC/ADVANCED DECISION DISAGREEMENT WITH RETRIEVAL

decision_disagreement_details = Counter()


for row in per_question:

    decision_pair = (
        row[
            "basic_advanced_decision_pair"
        ]
    )


    if decision_pair not in {

        "basic_only_correct",
        "advanced_only_correct"

    }:

        continue


    retrieval_pair = (
        row[
            "basic_advanced_retrieval_pair_at_5"
        ]
    )


    decision_disagreement_details[
        (
            decision_pair,
            retrieval_pair
        )
    ] += 1


print(
    "\nDecision disagreements cross-tabbed "
    "with Source Hit@5:"
)


for (
    decision_category,
    retrieval_category
), count in sorted(
    decision_disagreement_details.items()
):

    print(
        f"  {decision_category:<24} "
        f"{retrieval_category:<20} "
        f"{count}"
    )


# 18. ERROR DISTRIBUTION BY GOLD LABEL

print(
    "\n[11] DECISION ERRORS BY GOLD LABEL"
)


errors_by_gold_label = {}


for system in SYSTEM_ORDER:

    counts = Counter()


    for row in per_question:

        if not row[
            f"{system}_decision_correct"
        ]:

            counts[
                row[
                    "gold_decision"
                ]
            ] += 1


    errors_by_gold_label[
        system
    ] = dict(
        counts
    )


    print(
        f"\n{DISPLAY_NAMES[system]}"
    )


    for label in [

        "yes",
        "no",
        "maybe"

    ]:

        print(
            f"  {label:<5}: "
            f"{counts.get(label, 0)}"
        )


# 19. PREDICTION CONFUSION AMONG ERRORS

print(
    "\n[12] MOST COMMON WRONG DECISION TRANSITIONS"
)


wrong_transitions = {}


for system in SYSTEM_ORDER:

    transitions = Counter()


    for row in per_question:

        if row[
            f"{system}_decision_correct"
        ]:

            continue


        gold = (
            row[
                "gold_decision"
            ]
        )


        predicted = (
            row[
                f"{system}_predicted_decision"
            ]
        )


        transitions[
            f"{gold}->{predicted}"
        ] += 1


    wrong_transitions[
        system
    ] = dict(
        transitions
    )


    print(
        f"\n{DISPLAY_NAMES[system]}"
    )


    for transition, count in (
        transitions.most_common()
    ):

        print(
            f"  {transition:<12} "
            f"{count}"
        )


# 20. V3 HALLUCINATION TYPE BREAKDOWN

print(
    "\n[13] V3 HALLUCINATION TYPE BREAKDOWN"
)


hallucination_type_summary = {}


for system in SYSTEM_ORDER:

    unsupported = sum(

        row[
            f"{system}_unsupported_claims"
        ]

        for row
        in per_question
    )


    contradicted = sum(

        row[
            f"{system}_contradicted_claims"
        ]

        for row
        in per_question
    )


    total_hallucinated = (
        unsupported
        +
        contradicted
    )


    hallucination_type_summary[
        system
    ] = {

        "unsupported_claims":
            unsupported,

        "contradicted_claims":
            contradicted,

        "total_hallucinated_claims":
            total_hallucinated,

        "unsupported_share":
            safe_divide(
                unsupported,
                total_hallucinated
            ),

        "contradicted_share":
            safe_divide(
                contradicted,
                total_hallucinated
            )
    }


    print(
        f"\n{DISPLAY_NAMES[system]}"
    )


    print(
        f"  Unsupported  : "
        f"{unsupported}"
    )


    print(
        f"  Contradicted : "
        f"{contradicted}"
    )


    print(
        f"  Total        : "
        f"{total_hallucinated}"
    )


    print(
        f"  Unsupported share : "
        f"{percentage(unsupported, total_hallucinated):.2f}%"
    )


    print(
        f"  Contradicted share: "
        f"{percentage(contradicted, total_hallucinated):.2f}%"
    )


# 21. GOLD CONTEXT RESIDUAL ERROR ANALYSIS

print(
    "\n[14] GOLD-CONTEXT RESIDUAL ERRORS"
)


gold_decision_errors = [

    row

    for row
    in per_question

    if not row[
        "gold_context_control_decision_correct"
    ]
]


gold_hallucination_answers = [

    row

    for row
    in per_question

    if row[
        "gold_context_control_has_hallucinated_claim"
    ]
]


gold_wrong_with_hallucination = sum(

    1

    for row
    in gold_decision_errors

    if row[
        "gold_context_control_has_hallucinated_claim"
    ]
)


gold_wrong_without_hallucination = (

    len(
        gold_decision_errors
    )
    -
    gold_wrong_with_hallucination
)


gold_residual_summary = {

    "decision_errors":
        len(
            gold_decision_errors
        ),

    "answers_with_hallucinated_claim":
        len(
            gold_hallucination_answers
        ),

    "decision_errors_with_hallucinated_claim":
        gold_wrong_with_hallucination,

    "decision_errors_without_hallucinated_claim":
        gold_wrong_without_hallucination,

    "interpretation":
        (
            "Gold-context decision errors occurring "
            "without V3 hallucinated claims indicate "
            "residual downstream errors that cannot "
            "be explained solely by evidence retrieval "
            "or unsupported factual generation."
        )
}


print(
    f"Gold decision errors               : "
    f"{len(gold_decision_errors)}"
)


print(
    f"Gold answers with hallucinated claim: "
    f"{len(gold_hallucination_answers)}"
)


print(
    f"Wrong + hallucinated               : "
    f"{gold_wrong_with_hallucination}"
)


print(
    f"Wrong + no hallucinated claims     : "
    f"{gold_wrong_without_hallucination}"
)


# 22. QUESTION IDs FOR MANUAL QUALITATIVE INSPECTION

# These lists are NOT a new metric.
# They simply create reproducible candidate sets for later qualitative discussion.



manual_inspection_sets = {

    "basic_retrieval_failures":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if not row[
                "basic_rag_source_hit_at_5"
            ]
        ],

    "advanced_retrieval_failures":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if not row[
                "advanced_rag_source_hit_at_5"
            ]
        ],

    "basic_wrong_despite_source_hit":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if (
                row[
                    "basic_rag_source_hit_at_5"
                ]
                and
                not row[
                    "basic_rag_decision_correct"
                ]
            )
        ],

    "advanced_wrong_despite_source_hit":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if (
                row[
                    "advanced_rag_source_hit_at_5"
                ]
                and
                not row[
                    "advanced_rag_decision_correct"
                ]
            )
        ],

    "basic_only_correct":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if (
                row[
                    "basic_advanced_decision_pair"
                ]
                ==
                "basic_only_correct"
            )
        ],

    "advanced_only_correct":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if (
                row[
                    "basic_advanced_decision_pair"
                ]
                ==
                "advanced_only_correct"
            )
        ],

    "gold_wrong_without_hallucinated_claim":
        [

            row[
                "question_id"
            ]

            for row
            in per_question

            if (
                not row[
                    "gold_context_control_decision_correct"
                ]
                and
                not row[
                    "gold_context_control_has_hallucinated_claim"
                ]
            )
        ]
}


# 23. SAVE QUESTION LEVEL CSV

print(
    "\n[15] SAVING QUESTION-LEVEL EXPLORATORY DATA"
)


if not per_question:

    raise RuntimeError(
        "No question-level rows generated."
    )


csv_fields = list(
    per_question[
        0
    ].keys()
)


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


    for row in per_question:

        writer.writerow(
            row
        )


print(
    OUTPUT_CSV_PATH
)


# 24. SAVE SUMMARY JSON

output = {

    "split":
        "test",

    "analysis":
        "exploratory_error_analysis",

    "exploratory_only":
        True,

    "confirmatory_endpoint":
        False,

    "new_llm_judgments":
        False,

    "new_statistical_tests":
        False,

    "questions":
        500,

    "definitions": {

        "retrieval_failure":
            (
                "No correct-source passage appears "
                "in the frozen generation top-5."
            ),

        "decision_error_despite_source_retrieval":
            (
                "The correct source appears in the "
                "top-5, but the generated yes/no/maybe "
                "decision is incorrect."
            ),

        "hallucination_despite_correct_decision_and_retrieval":
            (
                "The correct source appears in the "
                "top-5 and the decision is correct, "
                "but frozen V3 identifies at least one "
                "unsupported or contradicted factual claim."
            ),

        "clean_success":
            (
                "Correct-source top-5 retrieval, correct "
                "decision, and no frozen V3 unsupported "
                "or contradicted factual claims."
            )
    },

    "rag_error_summary":
        rag_error_summary,

    "basic_vs_advanced": {

        "decision_correctness":
            dict(
                decision_pair_counts
            ),

        "source_hit_at_5":
            dict(
                retrieval_pair_counts
            ),

        "hallucination_presence":
            dict(
                hallucination_pair_counts
            ),

        "decision_disagreements_by_retrieval_pattern":
            [

                {
                    "decision_category":
                        decision_category,

                    "retrieval_category":
                        retrieval_category,

                    "count":
                        count
                }

                for (
                    decision_category,
                    retrieval_category
                ), count
                in sorted(
                    decision_disagreement_details.items()
                )
            ]
    },

    "decision_errors_by_gold_label":
        errors_by_gold_label,

    "wrong_decision_transitions":
        wrong_transitions,

    "hallucination_type_breakdown":
        hallucination_type_summary,

    "gold_context_residual_errors":
        gold_residual_summary,

    "manual_inspection_candidate_sets":
        manual_inspection_sets,

    "input_hashes": {

        relative_path(
            path
        ):
            sha256_file(
                path
            )

        for path
        in input_paths
    },

    "interpretation_guardrails": [

        (
            "This analysis is exploratory and "
            "must not be presented as a new "
            "pre-specified confirmatory endpoint."
        ),

        (
            "Rule-based categories describe observable "
            "patterns in frozen outputs; they are not "
            "causal diagnoses."
        ),

        (
            "A source hit does not guarantee that all "
            "answer-bearing gold evidence was retrieved."
        ),

        (
            "A decision error despite source retrieval "
            "may reflect incomplete evidence retrieval, "
            "generation/interpretation error, uncertainty "
            "handling, or benchmark ambiguity."
        ),

        (
            "V3 hallucination labels retain the human-"
            "validation limitations documented before "
            "the final test."
        )
    ]
}


save_json(
    output,
    OUTPUT_JSON_PATH
)


# 25. FINAL EXPLORATORY SCORECARD

print(
    "\n[16] EXPLORATORY ERROR SCORECARD"
)


print(
    f"{'Measure':<44}"
    f"{'Basic':>12}"
    f"{'Advanced':>12}"
)


print(
    "-" * 68
)


basic_summary = (
    rag_error_summary[
        "basic_rag"
    ]
)


advanced_summary = (
    rag_error_summary[
        "advanced_rag"
    ]
)


rows_to_print = [

    (
        "Retrieval misses @5",
        "retrieval_misses_at_5"
    ),

    (
        "Decision errors",
        "decision_errors"
    ),

    (
        "Decision errors despite source hit",
        "decision_errors_with_source_hit_at_5"
    ),

    (
        "Decision errors after source miss",
        "decision_errors_with_source_miss_at_5"
    ),

    (
        "Answers with hallucinated claim",
        "answers_with_at_least_one_hallucinated_claim"
    ),

    (
        "Hallucination answers despite source hit",
        "hallucination_answers_with_source_hit_at_5"
    ),

    (
        "Hallucination answers after source miss",
        "hallucination_answers_with_source_miss_at_5"
    ),

    (
        "Zero-claim answers",
        "zero_claim_answers"
    )
]


for label, key in rows_to_print:

    print(
        f"{label:<44}"
        f"{basic_summary[key]:>12}"
        f"{advanced_summary[key]:>12}"
    )


print(
    "\nMean Gold-context Recall@5:"
)


print(
    f"  Basic | correct decision : "
    f"{basic_summary['mean_gold_context_recall_at_5_when_decision_correct']:.4f}"
)


print(
    f"  Basic | wrong decision   : "
    f"{basic_summary['mean_gold_context_recall_at_5_when_decision_wrong']:.4f}"
)


print(
    f"  Adv.  | correct decision : "
    f"{advanced_summary['mean_gold_context_recall_at_5_when_decision_correct']:.4f}"
)


print(
    f"  Adv.  | wrong decision   : "
    f"{advanced_summary['mean_gold_context_recall_at_5_when_decision_wrong']:.4f}"
)


# 26. FILES SAVED

print(
    "\n[17] FILES SAVED"
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
    "SECTION 47 COMPLETE - "
    "EXPLORATORY ERROR ANALYSIS"
)


print(
    "=" * 78
)


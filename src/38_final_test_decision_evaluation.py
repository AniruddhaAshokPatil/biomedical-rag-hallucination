import hashlib
import json

from collections import Counter
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


FINAL_ARTIFACT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "final_test_artifact_manifest.json"
)


GENERATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "final_test_generation_results.json"
)


GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_ground_truth.json"
)


TEST_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_questions.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DECISION_RESULTS_PATH = (
    OUTPUT_DIR
    / "final_test_decision_results.json"
)


DECISION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_test_decision_summary.json"
)


# 2. CONSTANTS

LABELS = [
    "yes",
    "no",
    "maybe"
]


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


# 4. METRIC HELPERS

def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:

        return 0.0


    return (
        numerator
        /
        denominator
    )


def calculate_class_metrics(
    gold_labels,
    predicted_labels,
    target_label
):

    true_positive = sum(

        1

        for gold, prediction
        in zip(
            gold_labels,
            predicted_labels
        )

        if (
            gold == target_label
            and
            prediction == target_label
        )
    )


    false_positive = sum(

        1

        for gold, prediction
        in zip(
            gold_labels,
            predicted_labels
        )

        if (
            gold != target_label
            and
            prediction == target_label
        )
    )


    false_negative = sum(

        1

        for gold, prediction
        in zip(
            gold_labels,
            predicted_labels
        )

        if (
            gold == target_label
            and
            prediction != target_label
        )
    )


    support = sum(

        1

        for gold
        in gold_labels

        if gold == target_label
    )


    precision = safe_divide(
        true_positive,
        (
            true_positive
            +
            false_positive
        )
    )


    recall = safe_divide(
        true_positive,
        (
            true_positive
            +
            false_negative
        )
    )


    if (
        precision
        +
        recall
    ) == 0:

        f1 = 0.0

    else:

        f1 = (
            2
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


    return {

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "support":
            support,

        "true_positive":
            true_positive,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative
    }


def calculate_confusion_matrix(
    gold_labels,
    predicted_labels
):

    matrix = {

        gold_label: {

            predicted_label:
                0

            for predicted_label
            in LABELS
        }

        for gold_label
        in LABELS
    }


    for gold, prediction in zip(
        gold_labels,
        predicted_labels
    ):

        matrix[
            gold
        ][
            prediction
        ] += 1


    return matrix


def evaluate_predictions(
    gold_labels,
    predicted_labels
):

    if (
        len(
            gold_labels
        )
        !=
        len(
            predicted_labels
        )
    ):

        raise ValueError(
            "Gold and prediction lengths differ."
        )


    number_questions = len(
        gold_labels
    )


    correct = sum(

        1

        for gold, prediction
        in zip(
            gold_labels,
            predicted_labels
        )

        if gold == prediction
    )


    accuracy = safe_divide(
        correct,
        number_questions
    )


    class_metrics = {}


    for label in LABELS:

        class_metrics[
            label
        ] = calculate_class_metrics(
            gold_labels,
            predicted_labels,
            label
        )


    macro_f1 = sum(

        class_metrics[
            label
        ][
            "f1"
        ]

        for label
        in LABELS

    ) / len(
        LABELS
    )


    macro_precision = sum(

        class_metrics[
            label
        ][
            "precision"
        ]

        for label
        in LABELS

    ) / len(
        LABELS
    )


    macro_recall = sum(

        class_metrics[
            label
        ][
            "recall"
        ]

        for label
        in LABELS

    ) / len(
        LABELS
    )


    confusion_matrix = (
        calculate_confusion_matrix(
            gold_labels,
            predicted_labels
        )
    )


    return {

        "n":
            number_questions,

        "correct":
            correct,

        "incorrect":
            (
                number_questions
                -
                correct
            ),

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "per_class":
            class_metrics,

        "confusion_matrix":
            confusion_matrix
    }


# 5. START

print("=" * 78)

print(
    "SECTION 38 - FINAL TEST DECISION EVALUATION"
)

print("=" * 78)


# 6. VERIFY FROZEN FINAL-TEST ARTIFACTS

print(
    "\n[1] VERIFYING FROZEN FINAL-TEST ARTIFACTS"
)


artifact_manifest = load_json(
    FINAL_ARTIFACT_MANIFEST_PATH
)


if (
    artifact_manifest[
        "status"
    ]
    !=
    "FROZEN_BEFORE_TEST_EVALUATION"
):

    raise RuntimeError(
        "Final-test artifacts were not "
        "properly frozen before evaluation."
    )


changed_files = []

missing_files = []


for relative_path, expected_hash in (
    artifact_manifest[
        "sha256"
    ].items()
):

    path = (
        PROJECT_ROOT
        / relative_path
    )


    if not path.exists():

        missing_files.append(
            relative_path
        )

        continue


    actual_hash = sha256_file(
        path
    )


    if actual_hash != expected_hash:

        changed_files.append(
            relative_path
        )


print(
    f"Frozen files checked : "
    f"{len(artifact_manifest['sha256'])}"
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

    if changed_files:

        print(
            "\nChanged:"
        )


        for filename in changed_files:

            print(
                filename
            )


    if missing_files:

        print(
            "\nMissing:"
        )


        for filename in missing_files:

            print(
                filename
            )


    raise RuntimeError(
        "Frozen final-test artifact "
        "integrity check failed."
    )


print(
    "PASS: Frozen generation artifacts "
    "are unchanged."
)


# 7. LOAD FINAL GENERATED ANSWERS

print(
    "\n[2] LOADING FROZEN GENERATED ANSWERS"
)


generation_data = load_json(
    GENERATION_RESULTS_PATH
)


generation_results = (
    generation_data[
        "results"
    ]
)


print(
    f"Generated answers : "
    f"{len(generation_results)}"
)


if len(
    generation_results
) != 2000:

    raise ValueError(
        "Expected exactly 2,000 "
        "generated answers."
    )


# 8. OPEN FINAL TEST GOLD DECISIONS

print(
    "\n[3] OPENING FINAL TEST GOLD DECISIONS"
)


test_questions = load_json(
    TEST_QUESTIONS_PATH
)


ground_truth = load_json(
    GROUND_TRUTH_PATH
)


test_question_ids = [

    str(
        record[
            "question_id"
        ]
    )

    for record
    in test_questions
]


if len(
    test_question_ids
) != 500:

    raise ValueError(
        "Expected exactly 500 "
        "test question IDs."
    )


gold_decisions = {}


for question_id in test_question_ids:

    if question_id not in ground_truth:

        raise KeyError(
            f"Missing ground truth "
            f"for {question_id}."
        )


    gold_decision = (
        ground_truth[
            question_id
        ][
            "gold_decision"
        ]
    )


    if gold_decision not in LABELS:

        raise ValueError(
            f"Unexpected gold decision "
            f"for {question_id}: "
            f"{gold_decision}"
        )


    gold_decisions[
        question_id
    ] = gold_decision


print(
    "FINAL TEST GOLD LABELS ARE NOW OPEN."
)


print(
    f"Gold decisions loaded : "
    f"{len(gold_decisions)}"
)


gold_distribution = Counter(
    gold_decisions.values()
)


print(
    "Gold distribution:"
)


for label in LABELS:

    print(
        f"  {label:<5}: "
        f"{gold_distribution[label]}"
    )


# 9. INDEX GENERATED ANSWERS

prediction_lookup = {}


for result in generation_results:

    question_id = str(
        result[
            "question_id"
        ]
    )


    system = result[
        "system"
    ]


    decision = result[
        "decision"
    ]


    if system not in SYSTEM_ORDER:

        raise ValueError(
            f"Unexpected system: "
            f"{system}"
        )


    if decision not in LABELS:

        raise ValueError(
            f"Unexpected decision "
            f"{decision} for "
            f"{question_id} / "
            f"{system}"
        )


    key = (
        question_id,
        system
    )


    if key in prediction_lookup:

        raise ValueError(
            f"Duplicate prediction "
            f"for {key}."
        )


    prediction_lookup[
        key
    ] = result


if len(
    prediction_lookup
) != 2000:

    raise ValueError(
        "Expected 2,000 unique "
        "question-system predictions."
    )


print(
    "\n[4] PREDICTION ALIGNMENT"
)


print(
    "Unique frozen predictions : "
    f"{len(prediction_lookup)}"
)


# 10. BUILD PAIRED PER-QUESTION RESULTS

paired_results = []


for question_id in test_question_ids:

    gold_decision = (
        gold_decisions[
            question_id
        ]
    )


    record = {

        "question_id":
            question_id,

        "gold_decision":
            gold_decision
    }


    for system in SYSTEM_ORDER:

        key = (
            question_id,
            system
        )


        if key not in prediction_lookup:

            raise KeyError(
                f"Missing prediction "
                f"for {key}."
            )


        prediction = (
            prediction_lookup[
                key
            ][
                "decision"
            ]
        )


        record[
            system
        ] = {

            "predicted_decision":
                prediction,

            "correct":
                (
                    prediction
                    ==
                    gold_decision
                )
        }


    paired_results.append(
        record
    )


if len(
    paired_results
) != 500:

    raise ValueError(
        "Expected 500 paired "
        "question records."
    )


print(
    "PASS: All 500 questions have "
    "four paired predictions."
)


# 11. CALCULATE DECISION METRICS

print(
    "\n[5] FINAL TEST DECISION RESULTS"
)


system_metrics = {}


for system in SYSTEM_ORDER:

    gold_labels = [

        record[
            "gold_decision"
        ]

        for record
        in paired_results
    ]


    predicted_labels = [

        record[
            system
        ][
            "predicted_decision"
        ]

        for record
        in paired_results
    ]


    metrics = evaluate_predictions(
        gold_labels,
        predicted_labels
    )


    system_metrics[
        system
    ] = metrics


    print(
        "\n"
        + SYSTEM_DISPLAY_NAMES[
            system
        ]
    )


    print(
        "-" * 45
    )


    print(
        f"Correct    : "
        f"{metrics['correct']}/500"
    )


    print(
        f"Accuracy   : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )


    print(
        f"Macro-F1   : "
        f"{metrics['macro_f1']:.4f}"
    )


# 12. PER-CLASS PERFORMANCE

print(
    "\n[6] PER-CLASS PRECISION / RECALL / F1"
)


for system in SYSTEM_ORDER:

    metrics = (
        system_metrics[
            system
        ]
    )


    print(
        "\n"
        + SYSTEM_DISPLAY_NAMES[
            system
        ]
    )


    print(
        f"{'Class':<8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )


    for label in LABELS:

        class_result = (
            metrics[
                "per_class"
            ][
                label
            ]
        )


        print(
            f"{label:<8}"
            f"{class_result['precision']:>12.4f}"
            f"{class_result['recall']:>12.4f}"
            f"{class_result['f1']:>12.4f}"
            f"{class_result['support']:>10}"
        )


# 13. CONFUSION MATRICES

print(
    "\n[7] CONFUSION MATRICES"
)


print(
    "Rows = GOLD"
)


print(
    "Columns = PREDICTED"
)


for system in SYSTEM_ORDER:

    matrix = (
        system_metrics[
            system
        ][
            "confusion_matrix"
        ]
    )


    print(
        "\n"
        + SYSTEM_DISPLAY_NAMES[
            system
        ]
    )


    print(
        f"{'':<10}"
        f"{'yes':>8}"
        f"{'no':>8}"
        f"{'maybe':>8}"
    )


    for gold_label in LABELS:

        print(
            f"{gold_label:<10}"
            f"{matrix[gold_label]['yes']:>8}"
            f"{matrix[gold_label]['no']:>8}"
            f"{matrix[gold_label]['maybe']:>8}"
        )


# 14. SIMPLE DESCRIPTIVE DIFFERENCES

# These are descriptive percentage-point differences only.
# No p-values are calculated in this section.


print(
    "\n[8] DESCRIPTIVE ACCURACY DIFFERENCES"
)


baseline_accuracy = (
    system_metrics[
        "baseline_llm"
    ][
        "accuracy"
    ]
)


basic_accuracy = (
    system_metrics[
        "basic_rag"
    ][
        "accuracy"
    ]
)


advanced_accuracy = (
    system_metrics[
        "advanced_rag"
    ][
        "accuracy"
    ]
)


gold_accuracy = (
    system_metrics[
        "gold_context_control"
    ][
        "accuracy"
    ]
)


differences = {

    "basic_minus_baseline":
        (
            basic_accuracy
            -
            baseline_accuracy
        ),

    "advanced_minus_baseline":
        (
            advanced_accuracy
            -
            baseline_accuracy
        ),

    "advanced_minus_basic":
        (
            advanced_accuracy
            -
            basic_accuracy
        ),

    "gold_minus_advanced":
        (
            gold_accuracy
            -
            advanced_accuracy
        ),

    "gold_minus_basic":
        (
            gold_accuracy
            -
            basic_accuracy
        )
}


print(
    f"Basic - Baseline    : "
    f"{differences['basic_minus_baseline'] * 100:+.2f} pp"
)


print(
    f"Advanced - Baseline : "
    f"{differences['advanced_minus_baseline'] * 100:+.2f} pp"
)


print(
    f"Advanced - Basic    : "
    f"{differences['advanced_minus_basic'] * 100:+.2f} pp"
)


print(
    f"Gold - Advanced     : "
    f"{differences['gold_minus_advanced'] * 100:+.2f} pp"
)


# 15. SAVE RESULTS

save_json(
    paired_results,
    DECISION_RESULTS_PATH
)


summary = {

    "split":
        "test",

    "evaluation_stage":
        "confirmatory",

    "questions":
        500,

    "labels":
        LABELS,

    "gold_distribution": {

        label:
            gold_distribution[
                label
            ]

        for label
        in LABELS
    },

    "systems":
        system_metrics,

    "descriptive_accuracy_differences":
        differences,

    "statistical_tests_performed":
        False,

    "hallucination_evaluation_performed":
        False
}


save_json(
    summary,
    DECISION_SUMMARY_PATH
)


# 16. FINAL SUMMARY TABLE

print(
    "\n[9] FINAL DECISION SCORECARD"
)


print(
    f"{'System':<24}"
    f"{'Correct':>10}"
    f"{'Accuracy':>12}"
    f"{'Macro-F1':>12}"
)


print(
    "-" * 58
)


for system in SYSTEM_ORDER:

    metrics = (
        system_metrics[
            system
        ]
    )


    print(
        f"{SYSTEM_DISPLAY_NAMES[system]:<24}"
        f"{metrics['correct']:>10}"
        f"{metrics['accuracy']:>12.4f}"
        f"{metrics['macro_f1']:>12.4f}"
    )


# 17. FILES SAVED

print(
    "\n[10] FILES SAVED"
)


print(
    DECISION_RESULTS_PATH
)


print(
    DECISION_SUMMARY_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 38 COMPLETE - "
    "FINAL TEST DECISION METRICS CALCULATED"
)


print(
    "=" * 78
)


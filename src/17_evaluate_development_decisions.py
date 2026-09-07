import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


GENERATION_PATH = (
    GENERATION_RESULTS_DIR
    / "development_generations.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR
    / "evaluation_ground_truth.json"
)

OUTPUT_PATH = (
    GENERATION_RESULTS_DIR
    / "development_decision_evaluation.json"
)

# 2. SETTINGS

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


# 4. CLASSIFICATION METRICS

def calculate_metrics(
    gold_labels,
    predictions
):

    total = len(
        gold_labels
    )


    correct = sum(

        1

        for gold, prediction
        in zip(
            gold_labels,
            predictions
        )

        if gold == prediction
    )


    accuracy = (
        correct
        /
        total
    )


    class_metrics = {}

    class_f1_values = []


    for label in LABELS:

        true_positive = sum(

            1

            for gold, prediction
            in zip(
                gold_labels,
                predictions
            )

            if (
                gold == label
                and
                prediction == label
            )
        )


        false_positive = sum(

            1

            for gold, prediction
            in zip(
                gold_labels,
                predictions
            )

            if (
                gold != label
                and
                prediction == label
            )
        )


        false_negative = sum(

            1

            for gold, prediction
            in zip(
                gold_labels,
                predictions
            )

            if (
                gold == label
                and
                prediction != label
            )
        )


        support = sum(

            1

            for gold in gold_labels

            if gold == label
        )


        precision = (

            true_positive
            /
            (
                true_positive
                +
                false_positive
            )

            if (
                true_positive
                +
                false_positive
            ) > 0

            else 0.0
        )


        recall = (

            true_positive
            /
            (
                true_positive
                +
                false_negative
            )

            if (
                true_positive
                +
                false_negative
            ) > 0

            else 0.0
        )


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

            if (
                precision
                +
                recall
            ) > 0

            else 0.0
        )


        class_metrics[
            label
        ] = {

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "support":
                support
        }


        class_f1_values.append(
            f1
        )


    macro_f1 = (

        sum(
            class_f1_values
        )

        /
        len(
            class_f1_values
        )
    )


    return {

        "accuracy":
            accuracy,

        "correct":
            correct,

        "total":
            total,

        "macro_f1":
            macro_f1,

        "class_metrics":
            class_metrics
    }


# 5. CONFUSION MATRIX

def calculate_confusion_matrix(
    gold_labels,
    predictions
):

    matrix = {

        gold_label: {

            predicted_label: 0

            for predicted_label
            in LABELS
        }

        for gold_label
        in LABELS
    }


    for gold, prediction in zip(
        gold_labels,
        predictions
    ):

        matrix[
            gold
        ][
            prediction
        ] += 1


    return matrix


# 6. LOAD DATA

print("=" * 75)

print(
    "SECTION 17 - DEVELOPMENT DECISION EVALUATION"
)

print("=" * 75)


generation_file = load_json(
    GENERATION_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)


results = generation_file[
    "results"
]


print(
    "\n[1] DATA LOADED"
)


print(
    f"Generated responses : "
    f"{len(results)}"
)


# 7. VALIDATE DEVELOPMENT SPLIT

question_ids = sorted(

    set(

        result[
            "question_id"
        ]

        for result
        in results
    )
)


print(
    f"Unique questions    : "
    f"{len(question_ids)}"
)


wrong_split = []


for question_id in question_ids:

    if (
        ground_truth[
            question_id
        ]["split"]
        != "development"
    ):

        wrong_split.append(
            question_id
        )


print(
    f"Non-development IDs : "
    f"{len(wrong_split)}"
)


if wrong_split:

    raise ValueError(
        "Test-set leakage detected."
    )


if len(question_ids) != 500:

    raise ValueError(
        "Expected 500 development questions."
    )


print(
    "PASS: Evaluation uses development "
    "questions only."
)


# 8. INDEX PREDICTIONS

predictions_by_system = {

    system_name: {}

    for system_name
    in SYSTEM_ORDER
}


for result in results:

    system_name = result[
        "system"
    ]

    question_id = result[
        "question_id"
    ]


    predictions_by_system[
        system_name
    ][
        question_id
    ] = result[
        "decision"
    ]


# 9. GOLD DISTRIBUTION

gold_distribution = {

    label: 0

    for label
    in LABELS
}


for question_id in question_ids:

    gold_label = (

        ground_truth[
            question_id
        ][
            "gold_decision"
        ]
    )


    gold_distribution[
        gold_label
    ] += 1


print(
    "\n[2] GOLD DECISION DISTRIBUTION"
)


for label in LABELS:

    print(
        f"{label:<6}: "
        f"{gold_distribution[label]}"
    )


# 10. EVALUATE EACH SYSTEM

system_evaluations = {}

question_level_correctness = {}


print(
    "\n[3] SYSTEM PERFORMANCE"
)


for system_name in SYSTEM_ORDER:

    gold_labels = []

    predictions = []


    question_level_correctness[
        system_name
    ] = {}


    for question_id in question_ids:

        gold = (

            ground_truth[
                question_id
            ][
                "gold_decision"
            ]
        )


        prediction = (

            predictions_by_system[
                system_name
            ][
                question_id
            ]
        )


        gold_labels.append(
            gold
        )

        predictions.append(
            prediction
        )


        question_level_correctness[
            system_name
        ][
            question_id
        ] = (
            gold == prediction
        )


    metrics = calculate_metrics(
        gold_labels,
        predictions
    )


    matrix = calculate_confusion_matrix(
        gold_labels,
        predictions
    )


    system_evaluations[
        system_name
    ] = {

        **metrics,

        "confusion_matrix":
            matrix
    }


    print(
        f"\n{system_name}"
    )


    print(
        f"  Correct  : "
        f"{metrics['correct']}/"
        f"{metrics['total']}"
    )


    print(
        f"  Accuracy : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )


    print(
        f"  Macro-F1 : "
        f"{metrics['macro_f1']:.4f}"
    )


# 11. MAIN COMPARISON TABLE

print(
    "\n[4] MAIN COMPARISON TABLE"
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


for system_name in SYSTEM_ORDER:

    metrics = (
        system_evaluations[
            system_name
        ]
    )


    correct_string = (
        f"{metrics['correct']}/500"
    )


    print(
        f"{system_name:<24}"
        f"{correct_string:>10}"
        f"{metrics['accuracy']:>12.4f}"
        f"{metrics['macro_f1']:>12.4f}"
    )


# 12. CLASS-LEVEL RESULTS

print(
    "\n[5] CLASS-LEVEL PERFORMANCE"
)


for system_name in SYSTEM_ORDER:

    print(
        "\n"
        + system_name
    )


    metrics = (

        system_evaluations[
            system_name
        ][
            "class_metrics"
        ]
    )


    print(
        f"{'Label':<8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )


    for label in LABELS:

        values = metrics[
            label
        ]


        print(
            f"{label:<8}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1']:>12.4f}"
            f"{values['support']:>10}"
        )


# 13. CONFUSION MATRICES

print(
    "\n[6] CONFUSION MATRICES"
)

print(
    "Rows = GOLD label"
)

print(
    "Columns = PREDICTED label"
)


for system_name in SYSTEM_ORDER:

    matrix = (

        system_evaluations[
            system_name
        ][
            "confusion_matrix"
        ]
    )


    print(
        f"\n{system_name}"
    )


    print(
        f"{'Gold':<8}"
        f"{'yes':>8}"
        f"{'no':>8}"
        f"{'maybe':>8}"
    )


    for gold_label in LABELS:

        print(
            f"{gold_label:<8}"

            f"{matrix[gold_label]['yes']:>8}"

            f"{matrix[gold_label]['no']:>8}"

            f"{matrix[gold_label]['maybe']:>8}"
        )


# 14. PAIRED SYSTEM COMPARISONS

# Since every system answered the SAM 500 questions, comparisons are paired.
# Example:
# advanced only correct = questions where
# Advanced is correct and Basic is wrong.


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
    )
]


paired_results = {}


print(
    "\n[7] PAIRED CORRECTNESS COMPARISON"
)


for system_a, system_b in PAIRINGS:

    both_correct = 0

    both_wrong = 0

    a_only_correct = 0

    b_only_correct = 0


    for question_id in question_ids:

        a_correct = (

            question_level_correctness[
                system_a
            ][
                question_id
            ]
        )


        b_correct = (

            question_level_correctness[
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


    key = (
        f"{system_a}"
        f"_vs_"
        f"{system_b}"
    )


    paired_results[
        key
    ] = {

        "system_a":
            system_a,

        "system_b":
            system_b,

        "both_correct":
            both_correct,

        "both_wrong":
            both_wrong,

        "system_a_only_correct":
            a_only_correct,

        "system_b_only_correct":
            b_only_correct
    }


    print(
        "\n"
        + "-"
        * 70
    )


    print(
        f"{system_a} "
        f"VS "
        f"{system_b}"
    )


    print(
        f"Both correct           : "
        f"{both_correct}"
    )


    print(
        f"Both wrong             : "
        f"{both_wrong}"
    )


    print(
        f"{system_a} only correct : "
        f"{a_only_correct}"
    )


    print(
        f"{system_b} only correct : "
        f"{b_only_correct}"
    )


# 15. SAVE PER-QUESTION DATA

question_details = []


for question_id in question_ids:

    record = {

        "question_id":
            question_id,

        "gold_decision":
            ground_truth[
                question_id
            ][
                "gold_decision"
            ],

        "systems": {}
    }


    for system_name in SYSTEM_ORDER:

        prediction = (

            predictions_by_system[
                system_name
            ][
                question_id
            ]
        )


        record[
            "systems"
        ][
            system_name
        ] = {

            "prediction":
                prediction,

            "correct":
                (
                    prediction
                    ==
                    record[
                        "gold_decision"
                    ]
                )
        }


    question_details.append(
        record
    )


# 16. SAVE FINAL OUTPUT

output = {

    "split":
        "development",

    "questions":
        len(question_ids),

    "gold_distribution":
        gold_distribution,

    "system_evaluations":
        system_evaluations,

    "paired_comparisons":
        paired_results,

    "question_details":
        question_details
}


save_json(
    output,
    OUTPUT_PATH
)


print(
    "\n[8] RESULTS SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 17 COMPLETE"
)

print(
    "=" * 75
)
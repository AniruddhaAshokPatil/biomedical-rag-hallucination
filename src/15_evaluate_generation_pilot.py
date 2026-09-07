import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


PILOT_PATH = (
    RESULTS_DIR
    / "generation_pilot_25.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR
    / "evaluation_ground_truth.json"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "generation_pilot_25_evaluation.json"
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

    f1_values = []


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
                sum(
                    1
                    for gold
                    in gold_labels
                    if gold == label
                )
        }


        f1_values.append(
            f1
        )


    macro_f1 = (

        sum(
            f1_values
        )
        /
        len(
            f1_values
        )
    )


    return {
        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "class_metrics":
            class_metrics
    }


# 5. CONFUSION MATRIX

def confusion_matrix(
    gold_labels,
    predictions
):

    matrix = {

        gold: {
            prediction: 0
            for prediction
            in LABELS
        }

        for gold
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
print("SECTION 15 - EVALUATE GENERATION PILOT")
print("=" * 75)


pilot = load_json(
    PILOT_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)


pilot_results = (
    pilot[
        "results"
    ]
)


print("\n[1] DATA LOADED")

print(
    f"Pilot outputs: "
    f"{len(pilot_results)}"
)


# 7. PILOT QUESTION SET

question_ids = sorted(

    set(

        result[
            "question_id"
        ]

        for result
        in pilot_results
    )
)


print(
    f"Unique pilot questions: "
    f"{len(question_ids)}"
)


# 8. PILOT GOLD DISTRIBUTION

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
    "\n[2] PILOT GOLD DISTRIBUTION"
)


for label in LABELS:

    print(
        f"{label:<6}: "
        f"{gold_distribution[label]}"
    )


# 9. EVALUATE EACH SYSTEM

all_evaluations = {}


print(
    "\n[3] SYSTEM PERFORMANCE"
)


for system_name in SYSTEM_ORDER:

    system_results = {

        result[
            "question_id"
        ]: result

        for result
        in pilot_results

        if (
            result[
                "system"
            ]
            ==
            system_name
        )
    }


    gold_labels = []

    predictions = []


    for question_id in question_ids:

        gold_labels.append(

            ground_truth[
                question_id
            ][
                "gold_decision"
            ]
        )


        predictions.append(

            system_results[
                question_id
            ][
                "decision"
            ]
        )


    metrics = calculate_metrics(
        gold_labels,
        predictions
    )


    matrix = confusion_matrix(
        gold_labels,
        predictions
    )


    all_evaluations[
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
        f"  Accuracy : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        f"  Macro-F1 : "
        f"{metrics['macro_f1']:.4f}"
    )


# 10. COMPARISON TABLE

print(
    "\n[4] COMPARISON TABLE"
)


print(
    f"{'System':<24}"
    f"{'Accuracy':>12}"
    f"{'Macro-F1':>12}"
)

print(
    "-" * 48
)


for system_name in SYSTEM_ORDER:

    metrics = (
        all_evaluations[
            system_name
        ]
    )


    print(
        f"{system_name:<24}"
        f"{metrics['accuracy']:>12.4f}"
        f"{metrics['macro_f1']:>12.4f}"
    )


# 11. CONFUSION MATRICES

print(
    "\n[5] CONFUSION MATRICES"
)

print(
    "Rows = GOLD label"
)

print(
    "Columns = PREDICTED label"
)


for system_name in SYSTEM_ORDER:

    matrix = (

        all_evaluations[
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


# 12. CLASS-LEVEL PERFORMANCE

print(
    "\n[6] CLASS-LEVEL F1"
)


for system_name in SYSTEM_ORDER:

    print(
        f"\n{system_name}"
    )


    class_metrics = (

        all_evaluations[
            system_name
        ][
            "class_metrics"
        ]
    )


    for label in LABELS:

        values = (
            class_metrics[
                label
            ]
        )


        print(
            f"  {label:<6}"
            f" F1={values['f1']:.4f}"
            f" support={values['support']}"
        )


# 13. SAVE

output = {

    "pilot_questions":
        len(question_ids),

    "gold_distribution":
        gold_distribution,

    "systems":
        all_evaluations
}


save_json(
    output,
    OUTPUT_PATH
)


print(
    "\n[7] RESULTS SAVED"
)

print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 15 COMPLETE"
)

print(
    "=" * 75
)
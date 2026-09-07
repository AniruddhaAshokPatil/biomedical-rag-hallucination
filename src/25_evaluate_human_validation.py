import csv
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
    / "human_validation"
)

ANNOTATION_PATH = (
    VALIDATION_DIR
    / "human_validation_claims.csv"
)

KEY_PATH = (
    VALIDATION_DIR
    / "human_validation_key.json"
)

OUTPUT_PATH = (
    VALIDATION_DIR
    / "human_validation_results.json"
)


# 2. LABELS

LABELS = [
    "supported",
    "unsupported",
    "contradicted"
]


BINARY_LABELS = [
    "grounded",
    "hallucinated"
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


def clean_label(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def to_binary(label):

    if label == "supported":

        return "grounded"

    return "hallucinated"


# 4. CSV LOADER WITH EXCEL ENCODING SUPPORT

def load_annotation_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


    encodings = [
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]


    last_error = None


    for encoding in encodings:

        try:

            with path.open(
                "r",
                encoding=encoding,
                newline=""
            ) as file:

                reader = csv.DictReader(
                    file
                )

                rows = list(
                    reader
                )

            print(
                f"CSV encoding used : "
                f"{encoding}"
            )

            return rows


        except UnicodeDecodeError as error:

            last_error = error


    raise last_error


# 5. CONFUSION MATRIX

def confusion_matrix(
    human_labels,
    automatic_labels,
    labels
):

    matrix = {

        human_label: {

            automatic_label: 0

            for automatic_label
            in labels
        }

        for human_label
        in labels
    }


    for human, automatic in zip(
        human_labels,
        automatic_labels
    ):

        matrix[
            human
        ][
            automatic
        ] += 1


    return matrix


# 6. CLASSIFICATION METRICS

def classification_metrics(
    human_labels,
    automatic_labels,
    labels
):

    total = len(
        human_labels
    )


    correct = sum(

        1

        for human, automatic
        in zip(
            human_labels,
            automatic_labels
        )

        if human == automatic
    )


    accuracy = (
        correct
        /
        total
    )


    class_metrics = {}

    f1_values = []


    for label in labels:

        tp = sum(

            1

            for human, automatic
            in zip(
                human_labels,
                automatic_labels
            )

            if (
                human == label
                and
                automatic == label
            )
        )


        fp = sum(

            1

            for human, automatic
            in zip(
                human_labels,
                automatic_labels
            )

            if (
                human != label
                and
                automatic == label
            )
        )


        fn = sum(

            1

            for human, automatic
            in zip(
                human_labels,
                automatic_labels
            )

            if (
                human == label
                and
                automatic != label
            )
        )


        support = sum(

            1

            for human
            in human_labels

            if human == label
        )


        precision = (

            tp
            /
            (
                tp
                +
                fp
            )

            if (
                tp
                +
                fp
            ) > 0

            else 0.0
        )


        recall = (

            tp
            /
            (
                tp
                +
                fn
            )

            if (
                tp
                +
                fn
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

        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "class_metrics":
            class_metrics
    }


# 7. COHEN'S KAPPA

def cohens_kappa(
    human_labels,
    automatic_labels,
    labels
):

    total = len(
        human_labels
    )


    observed_agreement = (

        sum(

            1

            for human, automatic
            in zip(
                human_labels,
                automatic_labels
            )

            if human == automatic
        )

        /
        total
    )


    expected_agreement = 0.0


    for label in labels:

        human_count = sum(

            1

            for value
            in human_labels

            if value == label
        )


        automatic_count = sum(

            1

            for value
            in automatic_labels

            if value == label
        )


        expected_agreement += (

            (
                human_count
                /
                total
            )

            *
            (
                automatic_count
                /
                total
            )
        )


    if expected_agreement == 1:

        kappa = 1.0

    else:

        kappa = (

            (
                observed_agreement
                -
                expected_agreement
            )

            /
            (
                1
                -
                expected_agreement
            )
        )


    return {

        "observed_agreement":
            observed_agreement,

        "expected_agreement":
            expected_agreement,

        "kappa":
            kappa
    }


# 8. LOAD COMPLETED HUMAN ANNOTATIONS

print("=" * 78)

print(
    "SECTION 25 - HUMAN VALIDATION OF V3 JUDGE"
)

print("=" * 78)


annotation_rows = (
    load_annotation_csv(
        ANNOTATION_PATH
    )
)


key_data = load_json(
    KEY_PATH
)


automatic_key = (
    key_data[
        "key"
    ]
)


print(
    "\n[1] DATA LOADED"
)


print(
    f"Human annotation rows : "
    f"{len(annotation_rows)}"
)


print(
    f"Hidden key records     : "
    f"{len(automatic_key)}"
)



# 9. VALIDATE HUMAN FILE


print(
    "\n[2] HUMAN ANNOTATION VALIDATION"
)


if len(annotation_rows) != 120:

    raise ValueError(
        "Expected exactly 120 "
        "human annotation rows."
    )


validation_ids = [

    row[
        "validation_id"
    ].strip()

    for row
    in annotation_rows
]


if len(set(validation_ids)) != 120:

    raise ValueError(
        "Duplicate validation IDs found."
    )


if (
    set(validation_ids)
    !=
    set(automatic_key.keys())
):

    raise ValueError(
        "Human annotation IDs do not "
        "match the hidden key."
    )


human_labels = []

automatic_labels = []

combined_records = []


invalid_human_labels = []


for row in annotation_rows:

    validation_id = (
        row[
            "validation_id"
        ].strip()
    )


    human_label = clean_label(
        row[
            "human_label"
        ]
    )


    if human_label not in LABELS:

        invalid_human_labels.append(
            (
                validation_id,
                human_label
            )
        )

        continue


    automatic_label = clean_label(

        automatic_key[
            validation_id
        ][
            "automatic_label"
        ]
    )


    human_labels.append(
        human_label
    )

    automatic_labels.append(
        automatic_label
    )


    combined_records.append(
        {

            "validation_id":
                validation_id,

            "human_label":
                human_label,

            "automatic_label":
                automatic_label,

            "agreement":
                (
                    human_label
                    ==
                    automatic_label
                ),

            "human_notes":
                row.get(
                    "human_notes",
                    ""
                ),

            "question_id":

                automatic_key[
                    validation_id
                ][
                    "question_id"
                ],

            "system":

                automatic_key[
                    validation_id
                ][
                    "system"
                ]
        }
    )


print(
    f"Invalid human labels : "
    f"{len(invalid_human_labels)}"
)


if invalid_human_labels:

    print(
        invalid_human_labels
    )

    raise ValueError(
        "Fix invalid human labels "
        "before continuing."
    )


print(
    "PASS: All 120 human labels "
    "are valid."
)



# 10. HUMAN LABEL DISTRIBUTION


print(
    "\n[3] LABEL DISTRIBUTIONS"
)


human_distribution = {

    label: human_labels.count(
        label
    )

    for label in LABELS
}


automatic_distribution = {

    label: automatic_labels.count(
        label
    )

    for label in LABELS
}


print(
    "\nHuman labels:"
)


for label in LABELS:

    print(
        f"  {label:<13}: "
        f"{human_distribution[label]}"
    )


print(
    "\nAutomatic labels:"
)


for label in LABELS:

    print(
        f"  {label:<13}: "
        f"{automatic_distribution[label]}"
    )



# 11. THREE-CLASS EVALUATION


three_class_metrics = (
    classification_metrics(
        human_labels,
        automatic_labels,
        LABELS
    )
)


three_class_matrix = (
    confusion_matrix(
        human_labels,
        automatic_labels,
        LABELS
    )
)


three_class_kappa = (
    cohens_kappa(
        human_labels,
        automatic_labels,
        LABELS
    )
)


print(
    "\n[4] THREE-CLASS AGREEMENT"
)


print(
    f"Exact agreement : "
    f"{three_class_metrics['correct']}/"
    f"{three_class_metrics['total']}"
)


print(
    f"Accuracy        : "
    f"{three_class_metrics['accuracy']:.4f} "
    f"({three_class_metrics['accuracy'] * 100:.2f}%)"
)


print(
    f"Macro-F1        : "
    f"{three_class_metrics['macro_f1']:.4f}"
)


print(
    f"Cohen's kappa   : "
    f"{three_class_kappa['kappa']:.4f}"
)


# 12. CLASS-LEVEL PERFORMANCE

print(
    "\n[5] THREE-CLASS PERFORMANCE"
)


print(
    f"{'Label':<15}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>10}"
)


for label in LABELS:

    values = (

        three_class_metrics[
            "class_metrics"
        ][
            label
        ]
    )


    print(
        f"{label:<15}"
        f"{values['precision']:>12.4f}"
        f"{values['recall']:>12.4f}"
        f"{values['f1']:>12.4f}"
        f"{values['support']:>10}"
    )


# 13. THREE CLASS CONFUSION MATRIX

print(
    "\n[6] THREE-CLASS CONFUSION MATRIX"
)

print(
    "Rows = HUMAN"
)

print(
    "Columns = AUTOMATIC V3"
)


print(
    f"{'Human':<15}"
    f"{'supported':>14}"
    f"{'unsupported':>14}"
    f"{'contradicted':>14}"
)


for human_label in LABELS:

    print(
        f"{human_label:<15}"

        f"{three_class_matrix[human_label]['supported']:>14}"

        f"{three_class_matrix[human_label]['unsupported']:>14}"

        f"{three_class_matrix[human_label]['contradicted']:>14}"
    )



# 14. BINARY HALLUCINATION VALIDATION

# Dissertation hallucination metric:
# supported -> grounded
# unsupported -> hallucinated
# contradicted -> hallucinated
# This is therefore the most directly relevant validation of the main hallucination metric.


human_binary = [

    to_binary(
        label
    )

    for label
    in human_labels
]


automatic_binary = [

    to_binary(
        label
    )

    for label
    in automatic_labels
]


binary_metrics = (
    classification_metrics(
        human_binary,
        automatic_binary,
        BINARY_LABELS
    )
)


binary_matrix = (
    confusion_matrix(
        human_binary,
        automatic_binary,
        BINARY_LABELS
    )
)


binary_kappa = (
    cohens_kappa(
        human_binary,
        automatic_binary,
        BINARY_LABELS
    )
)


print(
    "\n[7] BINARY HALLUCINATION AGREEMENT"
)


print(
    f"Exact agreement : "
    f"{binary_metrics['correct']}/"
    f"{binary_metrics['total']}"
)


print(
    f"Accuracy        : "
    f"{binary_metrics['accuracy']:.4f} "
    f"({binary_metrics['accuracy'] * 100:.2f}%)"
)


print(
    f"Macro-F1        : "
    f"{binary_metrics['macro_f1']:.4f}"
)


print(
    f"Cohen's kappa   : "
    f"{binary_kappa['kappa']:.4f}"
)


print(
    "\nBinary class performance:"
)


print(
    f"{'Label':<15}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>10}"
)


for label in BINARY_LABELS:

    values = (

        binary_metrics[
            "class_metrics"
        ][
            label
        ]
    )


    print(
        f"{label:<15}"
        f"{values['precision']:>12.4f}"
        f"{values['recall']:>12.4f}"
        f"{values['f1']:>12.4f}"
        f"{values['support']:>10}"
    )


# 15. BINARY CONFUSION MATRIX

print(
    "\n[8] BINARY CONFUSION MATRIX"
)

print(
    "Rows = HUMAN"
)

print(
    "Columns = AUTOMATIC V3"
)


print(
    f"{'Human':<15}"
    f"{'grounded':>14}"
    f"{'hallucinated':>14}"
)


for human_label in BINARY_LABELS:

    print(
        f"{human_label:<15}"

        f"{binary_matrix[human_label]['grounded']:>14}"

        f"{binary_matrix[human_label]['hallucinated']:>14}"
    )


# 16. AUTOMATIC-LABEL STRATUM AGREEMENT

# Section 24 sampled exactly:
# 40 automatic supported
# 40 automatic unsupported
# 40 automatic contradicted
# Therefore we also report agreement separately inside each automatic judge category.


print(
    "\n[9] AGREEMENT BY AUTOMATIC LABEL"
)


stratum_agreement = {}


for automatic_label in LABELS:

    matching_records = [

        record

        for record
        in combined_records

        if (
            record[
                "automatic_label"
            ]
            ==
            automatic_label
        )
    ]


    agreeing = sum(

        1

        for record
        in matching_records

        if record[
            "agreement"
        ]
    )


    rate = (

        agreeing
        /
        len(
            matching_records
        )
    )


    stratum_agreement[
        automatic_label
    ] = {

        "sampled":
            len(
                matching_records
            ),

        "human_agreed":
            agreeing,

        "agreement_rate":
            rate
    }


    print(
        f"{automatic_label:<13}: "
        f"{agreeing}/"
        f"{len(matching_records)} "
        f"({rate * 100:.2f}%)"
    )


# 17. DISAGREEMENTS

disagreements = [

    record

    for record
    in combined_records

    if not record[
        "agreement"
    ]
]


print(
    "\n[10] DISAGREEMENT SUMMARY"
)


print(
    f"Total disagreements : "
    f"{len(disagreements)}"
)


transition_counts = {}


for record in disagreements:

    key = (

        f"human_{record['human_label']}"
        f"__auto_{record['automatic_label']}"
    )


    transition_counts[
        key
    ] = (
        transition_counts.get(
            key,
            0
        )
        +
        1
    )


for key, count in sorted(
    transition_counts.items()
):

    print(
        f"{key:<45}: "
        f"{count}"
    )


# 18. SAVE RESULTS

output = {

    "sample_size":
        len(
            human_labels
        ),

    "sampling_note":
        (
            "Human validation sample was "
            "stratified by automatic V3 label: "
            "40 supported, 40 unsupported, "
            "40 contradicted. Overall sample "
            "accuracy therefore describes the "
            "balanced validation sample rather "
            "than the natural full-corpus "
            "claim distribution."
        ),

    "human_distribution":
        human_distribution,

    "automatic_distribution":
        automatic_distribution,

    "three_class": {

        "metrics":
            three_class_metrics,

        "cohens_kappa":
            three_class_kappa,

        "confusion_matrix":
            three_class_matrix
    },

    "binary_hallucination": {

        "definition": {
            "grounded":
                "supported",

            "hallucinated":
                (
                    "unsupported or "
                    "contradicted"
                )
        },

        "metrics":
            binary_metrics,

        "cohens_kappa":
            binary_kappa,

        "confusion_matrix":
            binary_matrix
    },

    "agreement_by_automatic_label":
        stratum_agreement,

    "disagreement_count":
        len(
            disagreements
        ),

    "disagreement_transitions":
        transition_counts,

    "records":
        combined_records
}


save_json(
    output,
    OUTPUT_PATH
)


print(
    "\n[11] RESULTS SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 25 COMPLETE"
)

print(
    "=" * 78
)
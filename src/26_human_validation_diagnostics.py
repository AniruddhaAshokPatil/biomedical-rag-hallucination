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
    / "human_validation_diagnostics.json"
)

DISAGREEMENT_PATH = (
    VALIDATION_DIR
    / "human_validation_disagreements.csv"
)


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


# 2. HELPERS

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


def clean(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def binary(label):

    if label == "supported":
        return "grounded"

    return "hallucinated"


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


# 3. LOAD CSV WITH EXCEL ENCODING SUPPORT

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

                rows = list(
                    csv.DictReader(
                        file
                    )
                )


            print(
                f"CSV encoding used : "
                f"{encoding}"
            )


            return rows


        except UnicodeDecodeError as error:

            last_error = error


    raise last_error


# 4. LOAD DATA

print("=" * 78)

print(
    "SECTION 26 - HUMAN VALIDATION DIAGNOSTICS"
)

print("=" * 78)


rows = load_annotation_csv(
    ANNOTATION_PATH
)


key_data = load_json(
    KEY_PATH
)


key = key_data[
    "key"
]


print(
    "\n[1] DATA LOADED"
)


print(
    f"Human claims : "
    f"{len(rows)}"
)


print(
    f"Hidden key   : "
    f"{len(key)}"
)


if len(rows) != 120:

    raise ValueError(
        "Expected exactly 120 "
        "human validation rows."
    )


# 5. COMBINE HUMAN + AUTOMATIC

records = []

valid_labels = {
    "supported",
    "unsupported",
    "contradicted"
}


for row in rows:

    validation_id = (
        row[
            "validation_id"
        ].strip()
    )


    if validation_id not in key:

        raise ValueError(
            f"{validation_id} not found "
            f"in hidden key."
        )


    human_label = clean(
        row[
            "human_label"
        ]
    )


    automatic_label = clean(

        key[
            validation_id
        ][
            "automatic_label"
        ]
    )


    if human_label not in valid_labels:

        raise ValueError(
            f"Invalid human label "
            f"for {validation_id}: "
            f"{human_label}"
        )


    if automatic_label not in valid_labels:

        raise ValueError(
            f"Invalid automatic label "
            f"for {validation_id}: "
            f"{automatic_label}"
        )


    record = {

        "validation_id":
            validation_id,

        "question":
            row.get(
                "question",
                ""
            ),

        "generated_answer":
            row.get(
                "generated_answer",
                ""
            ),

        "claim":
            row.get(
                "claim_to_evaluate",
                ""
            ),

        "gold_evidence":
            row.get(
                "gold_evidence",
                ""
            ),

        "human_label":
            human_label,

        "automatic_label":
            automatic_label,

        "human_binary":
            binary(
                human_label
            ),

        "automatic_binary":
            binary(
                automatic_label
            ),

        "system":

            key[
                validation_id
            ][
                "system"
            ],

        "question_id":

            key[
                validation_id
            ][
                "question_id"
            ],

        "automatic_rationale":

            key[
                validation_id
            ].get(
                "automatic_rationale",
                ""
            ),

        "human_notes":
            row.get(
                "human_notes",
                ""
            )
    }


    records.append(
        record
    )


print(
    "PASS: Human and automatic "
    "records combined successfully."
)


# 6. OVERALL BINARY ERROR PROFILE

print(
    "\n[2] OVERALL BINARY ERROR PROFILE"
)


true_grounded = sum(

    1

    for record in records

    if (
        record[
            "human_binary"
        ]
        ==
        "grounded"
    )
)


true_hallucinated = sum(

    1

    for record in records

    if (
        record[
            "human_binary"
        ]
        ==
        "hallucinated"
    )
)


true_positive = sum(

    1

    for record in records

    if (
        record[
            "human_binary"
        ]
        ==
        "hallucinated"

        and

        record[
            "automatic_binary"
        ]
        ==
        "hallucinated"
    )
)


false_positive = sum(

    1

    for record in records

    if (
        record[
            "human_binary"
        ]
        ==
        "grounded"

        and

        record[
            "automatic_binary"
        ]
        ==
        "hallucinated"
    )
)


true_negative = sum(

    1

    for record in records

    if (
        record[
            "human_binary"
        ]
        ==
        "grounded"

        and

        record[
            "automatic_binary"
        ]
        ==
        "grounded"
    )
)


false_negative = sum(

    1

    for record in records

    if (
        record[
            "human_binary"
        ]
        ==
        "hallucinated"

        and

        record[
            "automatic_binary"
        ]
        ==
        "grounded"
    )
)


sensitivity = safe_divide(
    true_positive,
    true_hallucinated
)


specificity = safe_divide(
    true_negative,
    true_grounded
)


false_positive_rate = safe_divide(
    false_positive,
    true_grounded
)


false_negative_rate = safe_divide(
    false_negative,
    true_hallucinated
)


print(
    f"Human grounded       : "
    f"{true_grounded}"
)

print(
    f"Human hallucinated   : "
    f"{true_hallucinated}"
)

print(
    f"True positives       : "
    f"{true_positive}"
)

print(
    f"False positives      : "
    f"{false_positive}"
)

print(
    f"True negatives       : "
    f"{true_negative}"
)

print(
    f"False negatives      : "
    f"{false_negative}"
)


print(
    f"Hallucination recall : "
    f"{sensitivity:.4f}"
)

print(
    f"Grounded specificity : "
    f"{specificity:.4f}"
)

print(
    f"False-positive rate  : "
    f"{false_positive_rate:.4f}"
)

print(
    f"False-negative rate  : "
    f"{false_negative_rate:.4f}"
)


# 7. HUMAN CONFIRMATION BY AUTOMATIC LABEL

print(
    "\n[3] HUMAN CONFIRMATION BY AUTOMATIC LABEL"
)


label_confirmation = {}


for automatic_label in [
    "supported",
    "unsupported",
    "contradicted"
]:

    matching = [

        record

        for record in records

        if (
            record[
                "automatic_label"
            ]
            ==
            automatic_label
        )
    ]


    human_supported = sum(

        1

        for record in matching

        if (
            record[
                "human_label"
            ]
            ==
            "supported"
        )
    )


    human_unsupported = sum(

        1

        for record in matching

        if (
            record[
                "human_label"
            ]
            ==
            "unsupported"
        )
    )


    human_contradicted = sum(

        1

        for record in matching

        if (
            record[
                "human_label"
            ]
            ==
            "contradicted"
        )
    )


    human_hallucinated = (

        human_unsupported
        +
        human_contradicted
    )


    binary_confirmation = (
        safe_divide(
            human_hallucinated,
            len(
                matching
            )
        )
    )


    label_confirmation[
        automatic_label
    ] = {

        "sample":
            len(
                matching
            ),

        "human_supported":
            human_supported,

        "human_unsupported":
            human_unsupported,

        "human_contradicted":
            human_contradicted,

        "human_hallucinated":
            human_hallucinated,

        "binary_hallucination_confirmation":
            binary_confirmation
    }


    print(
        f"\nAutomatic {automatic_label}:"
    )


    print(
        f"  Sample              : "
        f"{len(matching)}"
    )


    print(
        f"  Human supported     : "
        f"{human_supported}"
    )


    print(
        f"  Human unsupported   : "
        f"{human_unsupported}"
    )


    print(
        f"  Human contradicted  : "
        f"{human_contradicted}"
    )


    if automatic_label != "supported":

        print(
            f"  Confirmed hallucinated: "
            f"{human_hallucinated}/"
            f"{len(matching)} "
            f"({binary_confirmation * 100:.2f}%)"
        )


# 8. BINARY AGREEMENT BY SYSTEM

print(
    "\n[4] BINARY AGREEMENT BY SYSTEM"
)


print(
    """
NOTE:
The validation sample was NOT stratified
by system.

These system-level results are diagnostic,
not final performance estimates.
"""
)


system_results = {}


for system_name in SYSTEM_ORDER:

    system_records = [

        record

        for record in records

        if (
            record[
                "system"
            ]
            ==
            system_name
        )
    ]


    if not system_records:

        continue


    correct = sum(

        1

        for record in system_records

        if (
            record[
                "human_binary"
            ]
            ==
            record[
                "automatic_binary"
            ]
        )
    )


    false_positives = sum(

        1

        for record in system_records

        if (
            record[
                "human_binary"
            ]
            ==
            "grounded"

            and

            record[
                "automatic_binary"
            ]
            ==
            "hallucinated"
        )
    )


    false_negatives = sum(

        1

        for record in system_records

        if (
            record[
                "human_binary"
            ]
            ==
            "hallucinated"

            and

            record[
                "automatic_binary"
            ]
            ==
            "grounded"
        )
    )


    human_grounded = sum(

        1

        for record in system_records

        if (
            record[
                "human_binary"
            ]
            ==
            "grounded"
        )
    )


    human_hallucinated = sum(

        1

        for record in system_records

        if (
            record[
                "human_binary"
            ]
            ==
            "hallucinated"
        )
    )


    automatic_hallucinated = sum(

        1

        for record in system_records

        if (
            record[
                "automatic_binary"
            ]
            ==
            "hallucinated"
        )
    )


    agreement = safe_divide(
        correct,
        len(
            system_records
        )
    )


    system_false_positive_rate = (
        safe_divide(
            false_positives,
            human_grounded
        )
    )


    system_false_negative_rate = (
        safe_divide(
            false_negatives,
            human_hallucinated
        )
    )


    system_results[
        system_name
    ] = {

        "claims":
            len(
                system_records
            ),

        "human_grounded":
            human_grounded,

        "human_hallucinated":
            human_hallucinated,

        "automatic_hallucinated":
            automatic_hallucinated,

        "binary_agreement":
            agreement,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,

        "false_positive_rate":
            system_false_positive_rate,

        "false_negative_rate":
            system_false_negative_rate
    }


    print(
        f"\n{system_name}"
    )


    print(
        f"  Claims              : "
        f"{len(system_records)}"
    )


    print(
        f"  Binary agreement    : "
        f"{agreement:.4f}"
    )


    print(
        f"  Human grounded      : "
        f"{human_grounded}"
    )


    print(
        f"  Human hallucinated  : "
        f"{human_hallucinated}"
    )


    print(
        f"  Auto hallucinated   : "
        f"{automatic_hallucinated}"
    )


    print(
        f"  False positives     : "
        f"{false_positives}"
    )


    print(
        f"  False negatives     : "
        f"{false_negatives}"
    )


    if system_false_positive_rate is not None:

        print(
            f"  False-positive rate : "
            f"{system_false_positive_rate:.4f}"
        )


    if system_false_negative_rate is not None:

        print(
            f"  False-negative rate : "
            f"{system_false_negative_rate:.4f}"
        )


# 9. THREE CLASS DISAGREEMENT TYPES

print(
    "\n[5] THREE-CLASS DISAGREEMENT TYPES"
)


disagreements = [

    record

    for record in records

    if (
        record[
            "human_label"
        ]
        !=
        record[
            "automatic_label"
        ]
    )
]


transition_counts = {}


for record in disagreements:

    transition = (

        record[
            "automatic_label"
        ]
        +
        " -> "
        +
        record[
            "human_label"
        ]
    )


    transition_counts[
        transition
    ] = (

        transition_counts.get(
            transition,
            0
        )
        +
        1
    )


for transition, count in sorted(
    transition_counts.items(),
    key=lambda item: (
        -item[1],
        item[0]
    )
):

    print(
        f"{transition:<35}: "
        f"{count}"
    )



# 10. BINARY DISAGREEMENTS


binary_disagreements = [

    record

    for record in records

    if (
        record[
            "human_binary"
        ]
        !=
        record[
            "automatic_binary"
        ]
    )
]


print(
    "\n[6] BINARY DISAGREEMENTS"
)


print(
    f"Binary disagreements : "
    f"{len(binary_disagreements)}"
)


print(
    f"False hallucination calls "
    f"(human grounded / auto hallucinated): "
    f"{false_positive}"
)


print(
    f"Missed hallucinations "
    f"(human hallucinated / auto grounded): "
    f"{false_negative}"
)



# 11. SAVE DISAGREEMENT REVIEW CSV


with DISAGREEMENT_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    fieldnames = [

        "validation_id",
        "system",
        "question",
        "generated_answer",
        "claim",
        "gold_evidence",
        "human_label",
        "automatic_label",
        "human_binary",
        "automatic_binary",
        "automatic_rationale",
        "human_notes"
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )


    writer.writeheader()


    for record in disagreements:

        writer.writerow({

            field:
                record.get(
                    field,
                    ""
                )

            for field
            in fieldnames
        })



# 12. SAVE JSON


output = {

    "sample_size":
        len(
            records
        ),

    "overall_binary": {

        "human_grounded":
            true_grounded,

        "human_hallucinated":
            true_hallucinated,

        "true_positive":
            true_positive,

        "false_positive":
            false_positive,

        "true_negative":
            true_negative,

        "false_negative":
            false_negative,

        "hallucination_recall":
            sensitivity,

        "grounded_specificity":
            specificity,

        "false_positive_rate":
            false_positive_rate,

        "false_negative_rate":
            false_negative_rate
    },

    "automatic_label_confirmation":
        label_confirmation,

    "system_diagnostics":
        system_results,

    "three_class_disagreement_types":
        transition_counts,

    "three_class_disagreements":
        len(
            disagreements
        ),

    "binary_disagreements":
        len(
            binary_disagreements
        )
}


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False
    )



# 13. FINISH


print(
    "\n[7] FILES SAVED"
)


print(
    OUTPUT_PATH
)


print(
    DISAGREEMENT_PATH
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 26 COMPLETE"
)

print(
    "=" * 78
)
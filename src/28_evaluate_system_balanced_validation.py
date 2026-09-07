import csv
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
    / "human_validation_system_balanced"
)

ANNOTATION_PATH = (
    VALIDATION_DIR
    / "system_balanced_validation_claims.csv"
)

KEY_PATH = (
    VALIDATION_DIR
    / "system_balanced_validation_key.json"
)

OUTPUT_PATH = (
    VALIDATION_DIR
    / "system_balanced_validation_results.json"
)


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


VALID_LABELS = {
    "grounded",
    "hallucinated"
}


# 2. HELPERS

def clean(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
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


def load_csv(path):

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
                f"CSV encoding used : {encoding}"
            )

            return rows


        except UnicodeDecodeError as error:

            last_error = error


    raise last_error


def confusion_counts(records):

    tp = sum(
        1
        for r in records
        if (
            r["human_label"] == "hallucinated"
            and
            r["automatic_label"] == "hallucinated"
        )
    )

    fp = sum(
        1
        for r in records
        if (
            r["human_label"] == "grounded"
            and
            r["automatic_label"] == "hallucinated"
        )
    )

    tn = sum(
        1
        for r in records
        if (
            r["human_label"] == "grounded"
            and
            r["automatic_label"] == "grounded"
        )
    )

    fn = sum(
        1
        for r in records
        if (
            r["human_label"] == "hallucinated"
            and
            r["automatic_label"] == "grounded"
        )
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def cohens_kappa(records):

    total = len(records)

    if total == 0:
        return None


    observed = safe_divide(
        sum(
            1
            for r in records
            if (
                r["human_label"]
                ==
                r["automatic_label"]
            )
        ),
        total
    )


    human_grounded = sum(
        1
        for r in records
        if r["human_label"] == "grounded"
    )

    human_hallucinated = sum(
        1
        for r in records
        if r["human_label"] == "hallucinated"
    )

    auto_grounded = sum(
        1
        for r in records
        if r["automatic_label"] == "grounded"
    )

    auto_hallucinated = sum(
        1
        for r in records
        if r["automatic_label"] == "hallucinated"
    )


    expected = (
        (
            human_grounded
            /
            total
        )
        *
        (
            auto_grounded
            /
            total
        )
        +
        (
            human_hallucinated
            /
            total
        )
        *
        (
            auto_hallucinated
            /
            total
        )
    )


    if expected == 1:
        return 1.0


    return (
        observed
        -
        expected
    ) / (
        1
        -
        expected
    )


def metrics_for_records(records):

    counts = confusion_counts(
        records
    )

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    total = len(records)

    correct = (
        tp
        +
        tn
    )


    accuracy = safe_divide(
        correct,
        total
    )


    hallucination_precision = safe_divide(
        tp,
        tp + fp
    )

    hallucination_recall = safe_divide(
        tp,
        tp + fn
    )


    hallucination_f1 = None

    if (
        hallucination_precision is not None
        and
        hallucination_recall is not None
        and
        (
            hallucination_precision
            +
            hallucination_recall
        ) > 0
    ):

        hallucination_f1 = (
            2
            *
            hallucination_precision
            *
            hallucination_recall
            /
            (
                hallucination_precision
                +
                hallucination_recall
            )
        )


    grounded_precision = safe_divide(
        tn,
        tn + fn
    )

    grounded_recall = safe_divide(
        tn,
        tn + fp
    )


    grounded_f1 = None

    if (
        grounded_precision is not None
        and
        grounded_recall is not None
        and
        (
            grounded_precision
            +
            grounded_recall
        ) > 0
    ):

        grounded_f1 = (
            2
            *
            grounded_precision
            *
            grounded_recall
            /
            (
                grounded_precision
                +
                grounded_recall
            )
        )


    f1_values = [
        value
        for value in [
            hallucination_f1,
            grounded_f1
        ]
        if value is not None
    ]


    macro_f1 = (
        sum(f1_values)
        /
        len(f1_values)
        if f1_values
        else None
    )


    false_positive_rate = safe_divide(
        fp,
        fp + tn
    )

    false_negative_rate = safe_divide(
        fn,
        fn + tp
    )


    return {

        "n":
            total,

        "correct":
            correct,

        "accuracy":
            accuracy,

        "cohens_kappa":
            cohens_kappa(
                records
            ),

        "confusion": {
            "true_positive":
                tp,

            "false_positive":
                fp,

            "true_negative":
                tn,

            "false_negative":
                fn
        },

        "hallucination": {
            "precision":
                hallucination_precision,

            "recall":
                hallucination_recall,

            "f1":
                hallucination_f1
        },

        "grounded": {
            "precision":
                grounded_precision,

            "recall_specificity":
                grounded_recall,

            "f1":
                grounded_f1
        },

        "macro_f1":
            macro_f1,

        "false_positive_rate":
            false_positive_rate,

        "false_negative_rate":
            false_negative_rate
    }


# 3. LOAD DATA

print("=" * 78)

print(
    "SECTION 28 - SYSTEM-BALANCED HUMAN VALIDATION EVALUATION"
)

print("=" * 78)


rows = load_csv(
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
    f"Human annotation rows : "
    f"{len(rows)}"
)


print(
    f"Hidden key records     : "
    f"{len(key)}"
)


# 4.VALIDATION

print(
    "\n[2] INPUT VALIDATION"
)


if len(rows) != 160:

    raise ValueError(
        f"Expected 160 rows, found "
        f"{len(rows)}"
    )


validation_ids = [

    row[
        "validation_id"
    ].strip()

    for row in rows
]


if len(set(validation_ids)) != 160:

    raise ValueError(
        "Duplicate validation IDs found."
    )


if set(validation_ids) != set(key.keys()):

    raise ValueError(
        "CSV validation IDs do not match "
        "the hidden key IDs."
    )


records = []


for row in rows:

    validation_id = (
        row[
            "validation_id"
        ].strip()
    )


    human_label = clean(
        row[
            "human_label"
        ]
    )


    if human_label not in VALID_LABELS:

        raise ValueError(
            f"Invalid or missing human label "
            f"for {validation_id}: "
            f"'{human_label}'"
        )


    automatic_label = clean(
        key[
            validation_id
        ][
            "automatic_binary"
        ]
    )


    if automatic_label not in VALID_LABELS:

        raise ValueError(
            f"Invalid automatic binary label "
            f"for {validation_id}: "
            f"'{automatic_label}'"
        )


    records.append(
        {

            "validation_id":
                validation_id,

            "question_id":
                str(
                    key[
                        validation_id
                    ][
                        "question_id"
                    ]
                ),

            "system":
                key[
                    validation_id
                ][
                    "system"
                ],

            "human_label":
                human_label,

            "automatic_label":
                automatic_label,

            "original_automatic_label":
                key[
                    validation_id
                ][
                    "automatic_label"
                ]
        }
    )


print(
    "PASS: All 160 labels are valid."
)


# 5. OVERALL DISTRIBUTIONS

print(
    "\n[3] OVERALL LABEL DISTRIBUTIONS"
)


human_grounded = sum(
    1
    for r in records
    if r["human_label"] == "grounded"
)

human_hallucinated = sum(
    1
    for r in records
    if r["human_label"] == "hallucinated"
)

auto_grounded = sum(
    1
    for r in records
    if r["automatic_label"] == "grounded"
)

auto_hallucinated = sum(
    1
    for r in records
    if r["automatic_label"] == "hallucinated"
)


print(
    f"Human grounded       : "
    f"{human_grounded}"
)

print(
    f"Human hallucinated   : "
    f"{human_hallucinated}"
)

print(
    f"Automatic grounded   : "
    f"{auto_grounded}"
)

print(
    f"Automatic hallucinated: "
    f"{auto_hallucinated}"
)


# 6. OVERALL PERFORMANCE

overall = metrics_for_records(
    records
)


print(
    "\n[4] OVERALL BINARY AGREEMENT"
)


print(
    f"Exact agreement : "
    f"{overall['correct']}/"
    f"{overall['n']}"
)


print(
    f"Accuracy        : "
    f"{overall['accuracy']:.4f} "
    f"({overall['accuracy'] * 100:.2f}%)"
)


print(
    f"Macro-F1        : "
    f"{overall['macro_f1']:.4f}"
)


print(
    f"Cohen's kappa   : "
    f"{overall['cohens_kappa']:.4f}"
)


print(
    f"Hallucination precision : "
    f"{overall['hallucination']['precision']:.4f}"
)


print(
    f"Hallucination recall    : "
    f"{overall['hallucination']['recall']:.4f}"
)


print(
    f"Grounded specificity    : "
    f"{overall['grounded']['recall_specificity']:.4f}"
)


print(
    f"False-positive rate     : "
    f"{overall['false_positive_rate']:.4f}"
)


print(
    f"False-negative rate     : "
    f"{overall['false_negative_rate']:.4f}"
)


# 7. OVERALL CONFUSION MATRIX

print(
    "\n[5] OVERALL CONFUSION MATRIX"
)


print(
    "Rows = HUMAN"
)

print(
    "Columns = AUTOMATIC V3"
)


c = overall[
    "confusion"
]


print(
    f"{'Human':<18}"
    f"{'Grounded':>14}"
    f"{'Hallucinated':>16}"
)


print(
    f"{'grounded':<18}"
    f"{c['true_negative']:>14}"
    f"{c['false_positive']:>16}"
)


print(
    f"{'hallucinated':<18}"
    f"{c['false_negative']:>14}"
    f"{c['true_positive']:>16}"
)


# 8. SYSTEM-LEVEL EVALUATION

print(
    "\n[6] SYSTEM-LEVEL BINARY AGREEMENT"
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


    if len(system_records) != 40:

        raise ValueError(
            f"{system_name} should have "
            f"40 claims but has "
            f"{len(system_records)}."
        )


    metrics = metrics_for_records(
        system_records
    )


    system_results[
        system_name
    ] = metrics


    print(
        f"\n{system_name}"
    )


    print(
        f"  Claims              : "
        f"{metrics['n']}"
    )


    print(
        f"  Agreement           : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )


    print(
        f"  Cohen's kappa       : "
        f"{metrics['cohens_kappa']:.4f}"
    )


    print(
        f"  Hallucination prec. : "
        f"{metrics['hallucination']['precision']:.4f}"
    )


    print(
        f"  Hallucination recall: "
        f"{metrics['hallucination']['recall']:.4f}"
    )


    print(
        f"  Specificity         : "
        f"{metrics['grounded']['recall_specificity']:.4f}"
    )


    print(
        f"  False-positive rate : "
        f"{metrics['false_positive_rate']:.4f}"
    )


    print(
        f"  False-negative rate : "
        f"{metrics['false_negative_rate']:.4f}"
    )


    print(
        f"  TP / FP / TN / FN   : "
        f"{metrics['confusion']['true_positive']} / "
        f"{metrics['confusion']['false_positive']} / "
        f"{metrics['confusion']['true_negative']} / "
        f"{metrics['confusion']['false_negative']}"
    )


# 9. AUTOMATIC-STRATUM CONFIRMATION BY SYSTEM

# Every system deliberately contains:
# 20 automatic grounded
# 20 automatic hallucinated
# This lets us directly ask:
# how often does a human confirm auto-grounded?
# how often does a human confirm auto-hallucinated?


print(
    "\n[7] AUTOMATIC-STRATUM HUMAN CONFIRMATION"
)


stratum_results = {}


for system_name in SYSTEM_ORDER:

    system_records = [

        record

        for record in records

        if record["system"] == system_name
    ]


    stratum_results[
        system_name
    ] = {}


    print(
        f"\n{system_name}"
    )


    for automatic_label in [
        "grounded",
        "hallucinated"
    ]:

        stratum_records = [

            record

            for record in system_records

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

            for record in stratum_records

            if (
                record[
                    "human_label"
                ]
                ==
                automatic_label
            )
        )


        confirmation_rate = (
            safe_divide(
                agreeing,
                len(
                    stratum_records
                )
            )
        )


        stratum_results[
            system_name
        ][
            automatic_label
        ] = {

            "sample":
                len(
                    stratum_records
                ),

            "human_confirmation":
                agreeing,

            "confirmation_rate":
                confirmation_rate
        }


        print(
            f"  Auto {automatic_label:<12}: "
            f"{agreeing}/"
            f"{len(stratum_records)} "
            f"confirmed "
            f"({confirmation_rate * 100:.2f}%)"
        )


# 10. FALSE POSITIVE CONSISTENCY

print(
    "\n[8] FALSE-POSITIVE CONSISTENCY ACROSS SYSTEMS"
)


fpr_values = {}


for system_name in SYSTEM_ORDER:

    value = (
        system_results[
            system_name
        ][
            "false_positive_rate"
        ]
    )

    fpr_values[
        system_name
    ] = value


    print(
        f"{system_name:<22}: "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


valid_fprs = [

    value

    for value in fpr_values.values()

    if value is not None
]


fpr_range = (
    max(valid_fprs)
    -
    min(valid_fprs)
)


print(
    f"\nFPR range across systems : "
    f"{fpr_range:.4f} "
    f"({fpr_range * 100:.2f} percentage points)"
)



# 11. FALSE-NEGATIVE CONSISTENCY


print(
    "\n[9] FALSE-NEGATIVE CONSISTENCY ACROSS SYSTEMS"
)


fnr_values = {}


for system_name in SYSTEM_ORDER:

    value = (
        system_results[
            system_name
        ][
            "false_negative_rate"
        ]
    )


    fnr_values[
        system_name
    ] = value


    print(
        f"{system_name:<22}: "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


valid_fnrs = [

    value

    for value in fnr_values.values()

    if value is not None
]


fnr_range = (
    max(valid_fnrs)
    -
    min(valid_fnrs)
)


print(
    f"\nFNR range across systems : "
    f"{fnr_range:.4f} "
    f"({fnr_range * 100:.2f} percentage points)"
)


# 12. SAVE OUTPUT

output = {

    "sample_size":
        len(
            records
        ),

    "design_note":
        (
            "System-balanced binary validation "
            "sample containing 20 automatic-grounded "
            "and 20 automatic-hallucinated claims "
            "from each of four systems. Results "
            "describe evaluator agreement under "
            "balanced challenge sampling and are "
            "not estimates of natural claim "
            "prevalence."
        ),

    "overall":
        overall,

    "system_results":
        system_results,

    "automatic_stratum_confirmation":
        stratum_results,

    "false_positive_rates":
        fpr_values,

    "false_positive_rate_range":
        fpr_range,

    "false_negative_rates":
        fnr_values,

    "false_negative_rate_range":
        fnr_range
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
    "\n[10] RESULTS SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 28 COMPLETE"
)


print(
    "=" * 78
)
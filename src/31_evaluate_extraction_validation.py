import csv
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
    / "human_validation_extraction"
)

ANNOTATION_PATH = (
    VALIDATION_DIR
    / "extraction_validation_statements.csv"
)

KEY_PATH = (
    VALIDATION_DIR
    / "extraction_validation_key.json"
)

OUTPUT_PATH = (
    VALIDATION_DIR
    / "extraction_validation_results.json"
)


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


VALID_LABELS = {
    "score",
    "exclude"
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
                f"CSV encoding used : "
                f"{encoding}"
            )


            return rows


        except UnicodeDecodeError as error:

            last_error = error


    raise last_error


def confusion_counts(records):

    # Positive class = SCORE

    tp = sum(
        1
        for r in records
        if (
            r["human_label"] == "score"
            and
            r["automatic_label"] == "score"
        )
    )

    fp = sum(
        1
        for r in records
        if (
            r["human_label"] == "exclude"
            and
            r["automatic_label"] == "score"
        )
    )

    tn = sum(
        1
        for r in records
        if (
            r["human_label"] == "exclude"
            and
            r["automatic_label"] == "exclude"
        )
    )

    fn = sum(
        1
        for r in records
        if (
            r["human_label"] == "score"
            and
            r["automatic_label"] == "exclude"
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


    human_score = sum(
        1
        for r in records
        if r["human_label"] == "score"
    )

    human_exclude = sum(
        1
        for r in records
        if r["human_label"] == "exclude"
    )

    auto_score = sum(
        1
        for r in records
        if r["automatic_label"] == "score"
    )

    auto_exclude = sum(
        1
        for r in records
        if r["automatic_label"] == "exclude"
    )


    expected = (
        (
            human_score
            /
            total
        )
        *
        (
            auto_score
            /
            total
        )
        +
        (
            human_exclude
            /
            total
        )
        *
        (
            auto_exclude
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


def metrics(records):

    counts = confusion_counts(
        records
    )

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    total = len(records)

    accuracy = safe_divide(
        tp + tn,
        total
    )

    score_precision = safe_divide(
        tp,
        tp + fp
    )

    score_recall = safe_divide(
        tp,
        tp + fn
    )

    exclude_precision = safe_divide(
        tn,
        tn + fn
    )

    exclude_recall = safe_divide(
        tn,
        tn + fp
    )


    score_f1 = None

    if (
        score_precision is not None
        and
        score_recall is not None
        and
        (
            score_precision
            +
            score_recall
        ) > 0
    ):

        score_f1 = (
            2
            *
            score_precision
            *
            score_recall
            /
            (
                score_precision
                +
                score_recall
            )
        )


    exclude_f1 = None

    if (
        exclude_precision is not None
        and
        exclude_recall is not None
        and
        (
            exclude_precision
            +
            exclude_recall
        ) > 0
    ):

        exclude_f1 = (
            2
            *
            exclude_precision
            *
            exclude_recall
            /
            (
                exclude_precision
                +
                exclude_recall
            )
        )


    macro_f1 = None

    if (
        score_f1 is not None
        and
        exclude_f1 is not None
    ):

        macro_f1 = (
            score_f1
            +
            exclude_f1
        ) / 2


    return {

        "n":
            total,

        "accuracy":
            accuracy,

        "cohens_kappa":
            cohens_kappa(
                records
            ),

        "confusion": {
            "tp":
                tp,

            "fp":
                fp,

            "tn":
                tn,

            "fn":
                fn
        },

        "score": {
            "precision":
                score_precision,

            "recall":
                score_recall,

            "f1":
                score_f1
        },

        "exclude": {
            "precision":
                exclude_precision,

            "recall":
                exclude_recall,

            "f1":
                exclude_f1
        },

        "macro_f1":
            macro_f1
    }


# 3. LOAD DATA

print("=" * 78)

print(
    "SECTION 31 - EXTRACTION / EXCLUSION VALIDATION EVALUATION"
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


# 4. INPUT VALIDATION

print(
    "\n[2] INPUT VALIDATION"
)


if len(rows) != 80:

    raise ValueError(
        f"Expected 80 rows, "
        f"found {len(rows)}"
    )


validation_ids = [

    row[
        "validation_id"
    ].strip()

    for row in rows
]


if len(set(validation_ids)) != 80:

    raise ValueError(
        "Duplicate validation IDs found."
    )


if set(validation_ids) != set(key.keys()):

    raise ValueError(
        "CSV validation IDs do not match "
        "the hidden key."
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
            f"Invalid/missing human label "
            f"for {validation_id}: "
            f"'{human_label}'"
        )


    automatic_label = clean(
        key[
            validation_id
        ][
            "automatic_scope_label"
        ]
    )


    if automatic_label not in VALID_LABELS:

        raise ValueError(
            f"Invalid automatic label "
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
                automatic_label
        }
    )


print(
    "PASS: All 80 annotations are valid."
)


# 5. OVERALL LABEL DISTRIBUTIONS

print(
    "\n[3] OVERALL LABEL DISTRIBUTIONS"
)


human_score = sum(
    1
    for r in records
    if r["human_label"] == "score"
)

human_exclude = sum(
    1
    for r in records
    if r["human_label"] == "exclude"
)

auto_score = sum(
    1
    for r in records
    if r["automatic_label"] == "score"
)

auto_exclude = sum(
    1
    for r in records
    if r["automatic_label"] == "exclude"
)


print(
    f"Human score   : "
    f"{human_score}"
)

print(
    f"Human exclude : "
    f"{human_exclude}"
)

print(
    f"Auto score    : "
    f"{auto_score}"
)

print(
    f"Auto exclude  : "
    f"{auto_exclude}"
)


# 6. OVERALL PERFORMANCE

overall = metrics(
    records
)


print(
    "\n[4] OVERALL EXTRACTION AGREEMENT"
)


print(
    f"Accuracy      : "
    f"{overall['accuracy']:.4f} "
    f"({overall['accuracy'] * 100:.2f}%)"
)


print(
    f"Macro-F1      : "
    f"{overall['macro_f1']:.4f}"
)


print(
    f"Cohen's kappa : "
    f"{overall['cohens_kappa']:.4f}"
)


print(
    f"SCORE precision : "
    f"{overall['score']['precision']:.4f}"
)


print(
    f"SCORE recall    : "
    f"{overall['score']['recall']:.4f}"
)


print(
    f"EXCLUDE precision : "
    f"{overall['exclude']['precision']:.4f}"
)


print(
    f"EXCLUDE recall    : "
    f"{overall['exclude']['recall']:.4f}"
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
    f"{'Score':>12}"
    f"{'Exclude':>12}"
)


print(
    f"{'score':<18}"
    f"{c['tp']:>12}"
    f"{c['fn']:>12}"
)


print(
    f"{'exclude':<18}"
    f"{c['fp']:>12}"
    f"{c['tn']:>12}"
)


# 8. SYSTEM-LEVEL RESULTS

print(
    "\n[6] SYSTEM-LEVEL EXTRACTION AGREEMENT"
)


system_results = {}


for system in SYSTEM_ORDER:

    system_records = [

        record

        for record in records

        if (
            record[
                "system"
            ]
            ==
            system
        )
    ]


    if len(system_records) != 20:

        raise ValueError(
            f"{system} should contain "
            f"20 validation statements, "
            f"found {len(system_records)}."
        )


    result = metrics(
        system_records
    )


    system_results[
        system
    ] = result


    print(
        f"\n{system}"
    )


    print(
        f"  Statements          : "
        f"{result['n']}"
    )


    print(
        f"  Accuracy            : "
        f"{result['accuracy']:.4f} "
        f"({result['accuracy'] * 100:.2f}%)"
    )


    print(
        f"  Cohen's kappa       : "
        f"{result['cohens_kappa']:.4f}"
    )


    print(
        f"  SCORE precision     : "
        f"{result['score']['precision']:.4f}"
    )


    print(
        f"  SCORE recall        : "
        f"{result['score']['recall']:.4f}"
    )


    print(
        f"  EXCLUDE precision   : "
        f"{result['exclude']['precision']:.4f}"
    )


    print(
        f"  EXCLUDE recall      : "
        f"{result['exclude']['recall']:.4f}"
    )


    print(
        f"  TP / FP / TN / FN   : "
        f"{result['confusion']['tp']} / "
        f"{result['confusion']['fp']} / "
        f"{result['confusion']['tn']} / "
        f"{result['confusion']['fn']}"
    )


# 9. AUTOMATIC-STRATUM CONFIRMATION

print(
    "\n[7] AUTOMATIC-STRATUM HUMAN CONFIRMATION"
)


stratum_results = {}


for system in SYSTEM_ORDER:

    system_records = [

        record

        for record in records

        if record["system"] == system
    ]


    stratum_results[
        system
    ] = {}


    print(
        f"\n{system}"
    )


    for automatic_label in [
        "score",
        "exclude"
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


        rate = safe_divide(
            agreeing,
            len(
                stratum_records
            )
        )


        stratum_results[
            system
        ][
            automatic_label
        ] = {

            "sample":
                len(
                    stratum_records
                ),

            "confirmed":
                agreeing,

            "confirmation_rate":
                rate
        }


        print(
            f"  Auto {automatic_label:<7}: "
            f"{agreeing}/"
            f"{len(stratum_records)} "
            f"confirmed "
            f"({rate * 100:.2f}%)"
        )


# 10. ERROR CONSISTENCY ACROSS SYSTEMS

print(
    "\n[8] EXTRACTION ERROR CONSISTENCY"
)


false_score_rates = {}

false_exclude_rates = {}


for system in SYSTEM_ORDER:

    result = (
        system_results[
            system
        ]
    )


    # Human says EXCLUDE,
    # V3 incorrectly says SCORE.
    false_score_rate = safe_divide(
        result[
            "confusion"
        ][
            "fp"
        ],
        (
            result[
                "confusion"
            ][
                "fp"
            ]
            +
            result[
                "confusion"
            ][
                "tn"
            ]
        )
    )


    # Human says SCORE,
    # V3 incorrectly says EXCLUDE.
    false_exclude_rate = safe_divide(
        result[
            "confusion"
        ][
            "fn"
        ],
        (
            result[
                "confusion"
            ][
                "fn"
            ]
            +
            result[
                "confusion"
            ][
                "tp"
            ]
        )
    )


    false_score_rates[
        system
    ] = false_score_rate


    false_exclude_rates[
        system
    ] = false_exclude_rate


    print(
        f"\n{system}"
    )


    print(
        f"  False SCORE rate   : "
        f"{false_score_rate:.4f} "
        f"({false_score_rate * 100:.2f}%)"
    )


    print(
        f"  False EXCLUDE rate : "
        f"{false_exclude_rate:.4f} "
        f"({false_exclude_rate * 100:.2f}%)"
    )


valid_false_score = [

    value

    for value
    in false_score_rates.values()

    if value is not None
]


valid_false_exclude = [

    value

    for value
    in false_exclude_rates.values()

    if value is not None
]


false_score_range = (
    max(valid_false_score)
    -
    min(valid_false_score)
)


false_exclude_range = (
    max(valid_false_exclude)
    -
    min(valid_false_exclude)
)


print(
    f"\nFalse SCORE rate range   : "
    f"{false_score_range:.4f} "
    f"({false_score_range * 100:.2f} pp)"
)


print(
    f"False EXCLUDE rate range : "
    f"{false_exclude_range:.4f} "
    f"({false_exclude_range * 100:.2f} pp)"
)


# 11. SAVE RESULTS

output = {

    "design_note":
        (
            "Balanced human validation of V3 "
            "claim extraction/exclusion decisions. "
            "Each system contributes 10 automatic "
            "SCORE and 10 automatic EXCLUDE "
            "statements. Therefore overall accuracy "
            "is a balanced challenge-set metric, "
            "not natural population accuracy."
        ),

    "sample_size":
        len(
            records
        ),

    "overall":
        overall,

    "system_results":
        system_results,

    "automatic_stratum_confirmation":
        stratum_results,

    "false_score_rates":
        false_score_rates,

    "false_exclude_rates":
        false_exclude_rates,

    "false_score_rate_range":
        false_score_range,

    "false_exclude_rate_range":
        false_exclude_range
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


# 12. FINISH

print(
    "\n[9] RESULTS SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 31 COMPLETE"
)


print(
    "=" * 78
)
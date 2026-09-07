import csv
import json
import random
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

HALLUCINATION_PATH = (
    RESULTS_DIR
    / "development_hallucination_v3.json"
)

EVALUATION_INPUT_PATH = (
    RESULTS_DIR
    / "hallucination_evaluation_inputs.json"
)

V3_PILOT_PATH = (
    RESULTS_DIR
    / "hallucination_judge_v3_pilot.json"
)

FIRST_VALIDATION_KEY_PATH = (
    RESULTS_DIR
    / "human_validation"
    / "human_validation_key.json"
)


SECOND_VALIDATION_DIR = (
    RESULTS_DIR
    / "human_validation_system_balanced"
)

SECOND_VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ANNOTATION_PATH = (
    SECOND_VALIDATION_DIR
    / "system_balanced_validation_claims.csv"
)

KEY_PATH = (
    SECOND_VALIDATION_DIR
    / "system_balanced_validation_key.json"
)

SUMMARY_PATH = (
    SECOND_VALIDATION_DIR
    / "system_balanced_validation_summary.json"
)


# 2. SETTINGS

RANDOM_SEED = 2026

CLAIMS_PER_STRATUM = 20


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
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


def to_binary(
    automatic_label
):

    automatic_label = (
        str(
            automatic_label
        )
        .strip()
        .lower()
    )


    if automatic_label == "supported":

        return "grounded"


    if automatic_label in {
        "unsupported",
        "contradicted"
    }:

        return "hallucinated"


    raise ValueError(
        f"Unexpected automatic label: "
        f"{automatic_label}"
    )


def format_evidence(
    evidence
):

    blocks = []


    for passage in evidence:

        label = passage.get(
            "label",
            "G?"
        )

        section = passage.get(
            "section",
            ""
        )

        text = passage.get(
            "text",
            ""
        )


        if section:

            blocks.append(
                f"[{label}] "
                f"{section}: "
                f"{text}"
            )

        else:

            blocks.append(
                f"[{label}] "
                f"{text}"
            )


    return "\n\n".join(
        blocks
    )


# 4. LOAD DATA

print("=" * 78)

print(
    "SECTION 27 - SYSTEM-BALANCED HUMAN VALIDATION"
)

print("=" * 78)


hallucination_data = load_json(
    HALLUCINATION_PATH
)

evaluation_inputs = load_json(
    EVALUATION_INPUT_PATH
)

pilot_data = load_json(
    V3_PILOT_PATH
)

first_validation_key = load_json(
    FIRST_VALIDATION_KEY_PATH
)


print(
    "\n[1] DATA LOADED"
)


print(
    f"Development questions : "
    f"{len(hallucination_data['judgments'])}"
)


# 5. EXCLUDE PREVIOUSLY SEEN QUESTIONS

pilot_question_ids = {

    str(
        question_id
    )

    for question_id
    in pilot_data[
        "selected_question_ids"
    ]
}


first_validation_question_ids = {

    str(
        record[
            "question_id"
        ]
    )

    for record
    in first_validation_key[
        "key"
    ].values()
}


excluded_question_ids = (
    pilot_question_ids
    |
    first_validation_question_ids
)


print(
    "\n[2] QUESTION EXCLUSIONS"
)


print(
    f"V3 pilot questions           : "
    f"{len(pilot_question_ids)}"
)

print(
    f"First validation questions   : "
    f"{len(first_validation_question_ids)}"
)

print(
    f"Total unique excluded        : "
    f"{len(excluded_question_ids)}"
)


# 6. INDEX GOLD EVIDENCE

evidence_by_question = {}


for record in evaluation_inputs:

    question_id = str(
        record[
            "question_id"
        ]
    )


    if question_id not in evidence_by_question:

        evidence_by_question[
            question_id
        ] = record[
            "gold_evidence"
        ]


# 7. BUILD SYSTEM × BINARY STRATA

print(
    "\n[3] BUILDING CLAIM STRATA"
)


claim_pool = {

    system_name: {

        binary_label: []

        for binary_label
        in BINARY_LABELS
    }

    for system_name
    in SYSTEM_ORDER
}


for judgment in hallucination_data[
    "judgments"
]:

    question_id = str(
        judgment[
            "question_id"
        ]
    )


    if question_id in excluded_question_ids:

        continue


    question = judgment[
        "question"
    ]


    for system_name in SYSTEM_ORDER:

        system_result = (
            judgment[
                "systems"
            ][
                system_name
            ]
        )


        claims = system_result.get(
            "claims",
            []
        )


        for claim_index, claim in enumerate(
            claims,
            start=1
        ):

            automatic_label = (
                str(
                    claim[
                        "label"
                    ]
                )
                .strip()
                .lower()
            )


            automatic_binary = (
                to_binary(
                    automatic_label
                )
            )


            claim_pool[
                system_name
            ][
                automatic_binary
            ].append(
                {

                    "question_id":
                        question_id,

                    "question":
                        question,

                    "system":
                        system_name,

                    "claim_index":
                        claim_index,

                    "claim":
                        claim[
                            "claim"
                        ],

                    "automatic_label":
                        automatic_label,

                    "automatic_binary":
                        automatic_binary,

                    "automatic_evidence_labels":
                        claim.get(
                            "evidence_labels",
                            []
                        ),

                    "automatic_rationale":
                        claim.get(
                            "rationale",
                            ""
                        ),

                    "gold_evidence":
                        evidence_by_question[
                            question_id
                        ]
                }
            )


for system_name in SYSTEM_ORDER:

    print(
        f"\n{system_name}"
    )


    for binary_label in BINARY_LABELS:

        print(
            f"  {binary_label:<13}: "
            f"{len(claim_pool[system_name][binary_label])}"
        )


# 8. CHECK SAMPLE AVAILABILITY

for system_name in SYSTEM_ORDER:

    for binary_label in BINARY_LABELS:

        available = len(
            claim_pool[
                system_name
            ][
                binary_label
            ]
        )


        if available < CLAIMS_PER_STRATUM:

            raise ValueError(
                f"Not enough claims for "
                f"{system_name} / "
                f"{binary_label}. "
                f"Available={available}, "
                f"needed={CLAIMS_PER_STRATUM}"
            )


# 9. SAMPLE EACH STRATUM

print(
    "\n[4] SELECTING BALANCED SAMPLE"
)


rng = random.Random(
    RANDOM_SEED
)


selected = []


for system_name in SYSTEM_ORDER:

    for binary_label in BINARY_LABELS:

        sampled = rng.sample(
            claim_pool[
                system_name
            ][
                binary_label
            ],
            CLAIMS_PER_STRATUM
        )


        selected.extend(
            sampled
        )


rng.shuffle(
    selected
)


print(
    f"Claims per system × binary stratum : "
    f"{CLAIMS_PER_STRATUM}"
)

print(
    f"Systems                           : "
    f"{len(SYSTEM_ORDER)}"
)

print(
    f"Binary strata                     : "
    f"{len(BINARY_LABELS)}"
)

print(
    f"Total claims                      : "
    f"{len(selected)}"
)


# 10. PREPARE BLINDED ANNOTATION FILE

annotation_rows = []

hidden_key = {}


for index, item in enumerate(
    selected,
    start=1
):

    validation_id = (
        f"SB{index:03d}"
    )


    annotation_rows.append(
        {

            "validation_id":
                validation_id,

            "question":
                item[
                    "question"
                ],

            "claim_to_evaluate":
                item[
                    "claim"
                ],

            "gold_evidence":
                format_evidence(
                    item[
                        "gold_evidence"
                    ]
                ),

            "human_label":
                "",

            "human_notes":
                ""
        }
    )


    hidden_key[
        validation_id
    ] = {

        "question_id":
            item[
                "question_id"
            ],

        "system":
            item[
                "system"
            ],

        "claim_index":
            item[
                "claim_index"
            ],

        "automatic_label":
            item[
                "automatic_label"
            ],

        "automatic_binary":
            item[
                "automatic_binary"
            ],

        "automatic_evidence_labels":
            item[
                "automatic_evidence_labels"
            ],

        "automatic_rationale":
            item[
                "automatic_rationale"
            ]
    }


# 11. WRITE ANNOTATION CSV

with ANNOTATION_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    fieldnames = [
        "validation_id",
        "question",
        "claim_to_evaluate",
        "gold_evidence",
        "human_label",
        "human_notes"
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        annotation_rows
    )


# 12. SAVE HIDDEN KEY

save_json(
    {

        "random_seed":
            RANDOM_SEED,

        "claims_per_stratum":
            CLAIMS_PER_STRATUM,

        "systems":
            SYSTEM_ORDER,

        "binary_labels":
            BINARY_LABELS,

        "total_claims":
            len(
                selected
            ),

        "excluded_question_ids":
            sorted(
                excluded_question_ids
            ),

        "key":
            hidden_key
    },
    KEY_PATH
)


# 13. VERIFY BALANCE

balance = {

    system_name: {

        binary_label: 0

        for binary_label
        in BINARY_LABELS
    }

    for system_name
    in SYSTEM_ORDER
}


for item in selected:

    balance[
        item[
            "system"
        ]
    ][
        item[
            "automatic_binary"
        ]
    ] += 1


unique_questions = len({

    item[
        "question_id"
    ]

    for item
    in selected
})


summary = {

    "total_claims":
        len(
            selected
        ),

    "unique_questions":
        unique_questions,

    "claims_per_system_binary_stratum":
        CLAIMS_PER_STRATUM,

    "balance_hidden_from_annotator":
        balance,

    "excluded_previous_questions":
        len(
            excluded_question_ids
        )
}


save_json(
    summary,
    SUMMARY_PATH
)


# 14. OUTPUT

print(
    "\n[5] SAMPLE BALANCE"
)


for system_name in SYSTEM_ORDER:

    print(
        f"\n{system_name}"
    )


    for binary_label in BINARY_LABELS:

        print(
            f"  {binary_label:<13}: "
            f"{balance[system_name][binary_label]}"
        )


print(
    f"\nUnique questions : "
    f"{unique_questions}"
)


print(
    "\n[6] FILES SAVED"
)


print(
    "ANNOTATE THIS FILE:"
)

print(
    ANNOTATION_PATH
)


print(
    "\nDO NOT OPEN UNTIL ANNOTATION IS COMPLETE:"
)

print(
    KEY_PATH
)


print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 27 COMPLETE"
)

print(
    "=" * 78
)
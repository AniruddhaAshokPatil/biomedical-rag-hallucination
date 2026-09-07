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


HUMAN_VALIDATION_DIR = (
    RESULTS_DIR
    / "human_validation"
)

HUMAN_VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ANNOTATION_PATH = (
    HUMAN_VALIDATION_DIR
    / "human_validation_claims.csv"
)

KEY_PATH = (
    HUMAN_VALIDATION_DIR
    / "human_validation_key.json"
)

SUMMARY_PATH = (
    HUMAN_VALIDATION_DIR
    / "human_validation_sample_summary.json"
)


# 2. SETTINGS

RANDOM_SEED = 42

CLAIMS_PER_LABEL = 40


LABELS = [
    "supported",
    "unsupported",
    "contradicted"
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


def format_evidence(
    evidence
):

    blocks = []


    for passage in evidence:

        blocks.append(
            f"[{passage['label']}] "
            f"{passage['section']}: "
            f"{passage['text']}"
        )


    return "\n\n".join(
        blocks
    )


# 4. LOAD DATA

print("=" * 75)

print(
    "SECTION 24 - PREPARE HUMAN VALIDATION"
)

print("=" * 75)


hallucination_data = load_json(
    HALLUCINATION_PATH
)

evaluation_inputs = load_json(
    EVALUATION_INPUT_PATH
)

v3_pilot = load_json(
    V3_PILOT_PATH
)


pilot_question_ids = set(
    v3_pilot[
        "selected_question_ids"
    ]
)


print(
    "\n[1] DATA LOADED"
)


print(
    f"Full questions         : "
    f"{len(hallucination_data['judgments'])}"
)

print(
    f"V3 pilot questions     : "
    f"{len(pilot_question_ids)}"
)


# 5. INDEX GOLD EVIDENCE

evidence_by_question = {}


for record in evaluation_inputs:

    question_id = (
        record[
            "question_id"
        ]
    )


    if (
        question_id
        not in
        evidence_by_question
    ):

        evidence_by_question[
            question_id
        ] = record[
            "gold_evidence"
        ]


# 6. BUILD CLAIM POOL

print(
    "\n[2] BUILDING CLAIM POOL"
)


claim_pool = {

    label: []

    for label in LABELS
}


excluded_pilot_claims = 0


for judgment in hallucination_data[
    "judgments"
]:

    question_id = (
        judgment[
            "question_id"
        ]
    )


    # Do NOT validate on questions used to tune V1/V2/V3.

    if (
        question_id
        in pilot_question_ids
    ):

        for system_result in (
            judgment[
                "systems"
            ].values()
        ):

            excluded_pilot_claims += len(
                system_result[
                    "claims"
                ]
            )

        continue


    for system_name, system_result in (
        judgment[
            "systems"
        ].items()
    ):

        for claim_index, claim in enumerate(
            system_result[
                "claims"
            ],
            start=1
        ):

            automatic_label = (
                claim[
                    "label"
                ]
            )


            claim_pool[
                automatic_label
            ].append(
                {

                    "question_id":
                        question_id,

                    "question":
                        judgment[
                            "question"
                        ],

                    "system":
                        system_name,

                    "generated_answer":
                        system_result[
                            "generated_answer"
                        ],

                    "claim_index":
                        claim_index,

                    "claim":
                        claim[
                            "claim"
                        ],

                    "automatic_label":
                        automatic_label,

                    "automatic_evidence_labels":
                        claim[
                            "evidence_labels"
                        ],

                    "automatic_rationale":
                        claim[
                            "rationale"
                        ],

                    "gold_evidence":
                        evidence_by_question[
                            question_id
                        ]
                }
            )


print(
    f"Pilot claims excluded : "
    f"{excluded_pilot_claims}"
)


for label in LABELS:

    print(
        f"{label:<13}: "
        f"{len(claim_pool[label])}"
    )


# 7. CHECK SAMPLE AVAILABILITY

for label in LABELS:

    if (
        len(
            claim_pool[
                label
            ]
        )
        <
        CLAIMS_PER_LABEL
    ):

        raise ValueError(
            f"Not enough {label} claims "
            f"for human validation."
        )


# 8. STRATIFIED RANDOM SAMPLE

print(
    "\n[3] SELECTING HUMAN VALIDATION SAMPLE"
)


rng = random.Random(
    RANDOM_SEED
)


selected_claims = []


for label in LABELS:

    sampled = rng.sample(
        claim_pool[
            label
        ],
        CLAIMS_PER_LABEL
    )


    selected_claims.extend(
        sampled
    )


# Shuffle all 120 so automatic categories
# are not grouped together.

rng.shuffle(
    selected_claims
)


print(
    f"Selected supported     : "
    f"{CLAIMS_PER_LABEL}"
)

print(
    f"Selected unsupported   : "
    f"{CLAIMS_PER_LABEL}"
)

print(
    f"Selected contradicted  : "
    f"{CLAIMS_PER_LABEL}"
)

print(
    f"Total validation claims: "
    f"{len(selected_claims)}"
)


# 9. CREATE BLINDED VALIDATION IDS

annotation_rows = []

validation_key = {}


for index, item in enumerate(
    selected_claims,
    start=1
):

    validation_id = (
        f"HV{index:03d}"
    )


    evidence_text = (
        format_evidence(
            item[
                "gold_evidence"
            ]
        )
    )


    # What YOU see.
    # Automatic label and system identity are intentionally hidden.

    annotation_rows.append(
        {

            "validation_id":
                validation_id,

            "question":
                item[
                    "question"
                ],

            "generated_answer":
                item[
                    "generated_answer"
                ],

            "claim_to_evaluate":
                item[
                    "claim"
                ],

            "gold_evidence":
                evidence_text,

            "human_label":
                "",

            "human_notes":
                ""
        }
    )


    # Hidden answer key.
    # DO NOT inspect this file until
    # human annotation is complete.

    validation_key[
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

        "automatic_evidence_labels":
            item[
                "automatic_evidence_labels"
            ],

        "automatic_rationale":
            item[
                "automatic_rationale"
            ]
    }


# 10. WRITE ANNOTATION CSV

with ANNOTATION_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    fieldnames = [

        "validation_id",
        "question",
        "generated_answer",
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


# 11. SAVE HIDDEN KEY

save_json(
    {

        "random_seed":
            RANDOM_SEED,

        "claims_per_label":
            CLAIMS_PER_LABEL,

        "total_claims":
            len(
                selected_claims
            ),

        "excluded_v3_pilot_questions":
            sorted(
                pilot_question_ids
            ),

        "key":
            validation_key
    },
    KEY_PATH
)



# 12. SAMPLE SUMMARY


system_counts = {}


for item in selected_claims:

    system_name = (
        item[
            "system"
        ]
    )


    system_counts[
        system_name
    ] = (
        system_counts.get(
            system_name,
            0
        )
        +
        1
    )


unique_questions = len({

    item[
        "question_id"
    ]

    for item
    in selected_claims
})


summary = {

    "total_claims":
        len(
            selected_claims
        ),

    "automatic_label_sampling": {

        "supported":
            CLAIMS_PER_LABEL,

        "unsupported":
            CLAIMS_PER_LABEL,

        "contradicted":
            CLAIMS_PER_LABEL
    },

    "unique_questions":
        unique_questions,

    "system_distribution_hidden_from_annotator":
        system_counts,

    "pilot_questions_excluded":
        len(
            pilot_question_ids
        )
}


save_json(
    summary,
    SUMMARY_PATH
)


# 13. OUTPUT

print(
    "\n[4] HUMAN VALIDATION SAMPLE"
)


print(
    f"Validation claims : "
    f"{len(selected_claims)}"
)

print(
    f"Unique questions  : "
    f"{unique_questions}"
)


print(
    "\nSystem distribution "
    "(for preparation only):"
)


for system_name, count in (
    system_counts.items()
):

    print(
        f"  {system_name:<22}: "
        f"{count}"
    )


print(
    "\n[5] FILES SAVED"
)


print(
    f"ANNOTATE THIS FILE:"
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
    + "=" * 75
)

print(
    "SECTION 24 COMPLETE"
)

print(
    "=" * 75
)
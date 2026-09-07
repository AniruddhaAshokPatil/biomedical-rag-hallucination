import csv
import json
import random
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

HALLUCINATION_PATH = (
    GENERATION_DIR
    / "development_hallucination_v3.json"
)

V3_PILOT_PATH = (
    GENERATION_DIR
    / "hallucination_judge_v3_pilot.json"
)

FIRST_VALIDATION_KEY_PATH = (
    GENERATION_DIR
    / "human_validation"
    / "human_validation_key.json"
)

SECOND_VALIDATION_KEY_PATH = (
    GENERATION_DIR
    / "human_validation_system_balanced"
    / "system_balanced_validation_key.json"
)

OUTPUT_DIR = (
    GENERATION_DIR
    / "human_validation_extraction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATION_PATH = (
    OUTPUT_DIR
    / "extraction_validation_statements.csv"
)

KEY_PATH = (
    OUTPUT_DIR
    / "extraction_validation_key.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "extraction_validation_summary.json"
)


# 2. SETTINGS

RANDOM_SEED = 2027

STATEMENTS_PER_STRATUM = 10

SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]

SCOPE_LABELS = [
    "score",
    "exclude"
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


def clean_text(value):

    if value is None:
        return ""

    return str(
        value
    ).strip()


def get_claim_text(claim):

    if isinstance(
        claim,
        str
    ):

        return clean_text(
            claim
        )


    if isinstance(
        claim,
        dict
    ):

        for field in [
            "claim",
            "statement",
            "text"
        ]:

            if field in claim:

                value = clean_text(
                    claim[
                        field
                    ]
                )

                if value:

                    return value


    return ""


def get_excluded_text(item):

    if isinstance(
        item,
        str
    ):

        return clean_text(
            item
        )


    if isinstance(
        item,
        dict
    ):

        for field in [
            "statement",
            "text",
            "claim",
            "excluded_statement"
        ]:

            if field in item:

                value = clean_text(
                    item[
                        field
                    ]
                )

                if value:

                    return value


    return ""


def previous_question_ids(
    pilot_data,
    first_key,
    second_key
):

    pilot_ids = {

        str(
            value
        )

        for value
        in pilot_data.get(
            "selected_question_ids",
            []
        )
    }


    first_ids = {

        str(
            record[
                "question_id"
            ]
        )

        for record
        in first_key[
            "key"
        ].values()
    }


    second_ids = {

        str(
            record[
                "question_id"
            ]
        )

        for record
        in second_key[
            "key"
        ].values()
    }


    return (
        pilot_ids,
        first_ids,
        second_ids,
        pilot_ids | first_ids | second_ids
    )


def sample_unique_questions(
    candidates,
    number,
    rng,
    globally_used
):

    shuffled = list(
        candidates
    )

    rng.shuffle(
        shuffled
    )


    chosen = []

    local_used = set()


    # First preference: question has not appeared anywhere else in the sample.

    for item in shuffled:

        question_id = (
            item[
                "question_id"
            ]
        )


        if question_id in globally_used:

            continue


        if question_id in local_used:

            continue


        chosen.append(
            item
        )

        local_used.add(
            question_id
        )

        globally_used.add(
            question_id
        )


        if len(chosen) == number:

            return chosen


    # Fallback: allow a question used in another stratum, while still keeping question IDs unique inside this stratum.


    for item in shuffled:

        question_id = (
            item[
                "question_id"
            ]
        )


        if question_id in local_used:

            continue


        chosen.append(
            item
        )

        local_used.add(
            question_id
        )


        if len(chosen) == number:

            return chosen


    raise ValueError(
        "Not enough unique-question statements "
        "to construct requested sample."
    )


# 4. LOAD DATA

print("=" * 78)

print(
    "SECTION 30 - PREPARE CLAIM EXTRACTION VALIDATION"
)

print("=" * 78)


hallucination_data = load_json(
    HALLUCINATION_PATH
)

pilot_data = load_json(
    V3_PILOT_PATH
)

first_validation_key = load_json(
    FIRST_VALIDATION_KEY_PATH
)

second_validation_key = load_json(
    SECOND_VALIDATION_KEY_PATH
)


print(
    "\n[1] DATA LOADED"
)


print(
    f"Development judgments : "
    f"{len(hallucination_data['judgments'])}"
)


# 5. EXCLUDE ALL QUESTIONS PREVIOUSLY SEEN BY HUMAN

(
    pilot_ids,
    first_ids,
    second_ids,
    excluded_question_ids
) = previous_question_ids(
    pilot_data,
    first_validation_key,
    second_validation_key
)


print(
    "\n[2] PREVIOUS QUESTION EXCLUSIONS"
)


print(
    f"V3 pilot questions          : "
    f"{len(pilot_ids)}"
)

print(
    f"First human validation      : "
    f"{len(first_ids)}"
)

print(
    f"Second human validation     : "
    f"{len(second_ids)}"
)

print(
    f"Unique questions excluded   : "
    f"{len(excluded_question_ids)}"
)


# 6. BUILD SCORE / EXCLUDE POOLS

pool = {

    system: {

        scope_label: []

        for scope_label
        in SCOPE_LABELS
    }

    for system
    in SYSTEM_ORDER
}


empty_claim_text = 0

empty_excluded_text = 0


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


    question = clean_text(
        judgment.get(
            "question",
            ""
        )
    )


    for system in SYSTEM_ORDER:

        system_result = (
            judgment[
                "systems"
            ][
                system
            ]
        )


        # Claims V3 decided should enter the denominator.
        # Automatic extraction decision = SCORE.

        claims = system_result.get(
            "claims",
            []
        )


        for claim_index, claim in enumerate(
            claims,
            start=1
        ):

            statement = get_claim_text(
                claim
            )


            if not statement:

                empty_claim_text += 1
                continue


            pool[
                system
            ][
                "score"
            ].append(
                {

                    "question_id":
                        question_id,

                    "question":
                        question,

                    "system":
                        system,

                    "automatic_scope_label":
                        "score",

                    "source_index":
                        claim_index,

                    "statement":
                        statement
                }
            )


        # Statements V3 explicitly decided should NOT enter
        # the hallucination denominator.
        # Automatic extraction decision = EXCLUDE.

        excluded_statements = (
            system_result.get(
                "excluded_statements",
                []
            )
        )


        for excluded_index, item in enumerate(
            excluded_statements,
            start=1
        ):

            statement = get_excluded_text(
                item
            )


            if not statement:

                empty_excluded_text += 1
                continue


            pool[
                system
            ][
                "exclude"
            ].append(
                {

                    "question_id":
                        question_id,

                    "question":
                        question,

                    "system":
                        system,

                    "automatic_scope_label":
                        "exclude",

                    "source_index":
                        excluded_index,

                    "statement":
                        statement
                }
            )


print(
    "\n[3] AVAILABLE STATEMENT POOL"
)


for system in SYSTEM_ORDER:

    print(
        f"\n{system}"
    )

    print(
        f"  score   : "
        f"{len(pool[system]['score'])}"
    )

    print(
        f"  exclude : "
        f"{len(pool[system]['exclude'])}"
    )


print(
    f"\nEmpty claim texts skipped    : "
    f"{empty_claim_text}"
)

print(
    f"Empty excluded texts skipped : "
    f"{empty_excluded_text}"
)


# 7. CHECK AVAILABILITY

for system in SYSTEM_ORDER:

    for scope_label in SCOPE_LABELS:

        available = len(
            pool[
                system
            ][
                scope_label
            ]
        )


        if available < STATEMENTS_PER_STRATUM:

            raise ValueError(
                f"Not enough statements for "
                f"{system}/{scope_label}. "
                f"Available={available}, "
                f"required={STATEMENTS_PER_STRATUM}"
            )


# 8. BALANCED SAMPLING

rng = random.Random(
    RANDOM_SEED
)


selected = []

globally_used_questions = set()


for system in SYSTEM_ORDER:

    for scope_label in SCOPE_LABELS:

        sample = sample_unique_questions(
            candidates=pool[
                system
            ][
                scope_label
            ],
            number=STATEMENTS_PER_STRATUM,
            rng=rng,
            globally_used=globally_used_questions
        )


        selected.extend(
            sample
        )


rng.shuffle(
    selected
)


print(
    "\n[4] SAMPLE SELECTED"
)


print(
    f"Statements per stratum : "
    f"{STATEMENTS_PER_STRATUM}"
)

print(
    f"Systems                : "
    f"{len(SYSTEM_ORDER)}"
)

print(
    f"Scope labels           : "
    f"{len(SCOPE_LABELS)}"
)

print(
    f"Total statements       : "
    f"{len(selected)}"
)


# 9. BUILD BLINDED CSV + HIDDEN KEY

annotation_rows = []

hidden_key = {}


for index, item in enumerate(
    selected,
    start=1
):

    validation_id = (
        f"EX{index:03d}"
    )


    annotation_rows.append(
        {

            "validation_id":
                validation_id,

            "question":
                item[
                    "question"
                ],

            "statement_to_classify":
                item[
                    "statement"
                ],

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

        "source_index":
            item[
                "source_index"
            ],

        "automatic_scope_label":
            item[
                "automatic_scope_label"
            ]
    }


# 10. WRITE BLINDED ANNOTATION CSV

with ANNOTATION_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    fieldnames = [
        "validation_id",
        "question",
        "statement_to_classify",
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

        "statements_per_stratum":
            STATEMENTS_PER_STRATUM,

        "systems":
            SYSTEM_ORDER,

        "scope_labels":
            SCOPE_LABELS,

        "total_statements":
            len(
                selected
            ),

        "key":
            hidden_key
    },
    KEY_PATH
)


# 12. VERIFY HIDDEN BALANCE

balance = {

    system: {
        "score": 0,
        "exclude": 0
    }

    for system
    in SYSTEM_ORDER
}


for item in selected:

    balance[
        item[
            "system"
        ]
    ][
        item[
            "automatic_scope_label"
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

    "total_statements":
        len(
            selected
        ),

    "unique_questions":
        unique_questions,

    "statements_per_system_scope_stratum":
        STATEMENTS_PER_STRATUM,

    "hidden_balance":
        balance,

    "previously_seen_questions_excluded":
        len(
            excluded_question_ids
        )
}


save_json(
    summary,
    SUMMARY_PATH
)


# 13. OUTPUT

print(
    "\n[5] HIDDEN SAMPLE BALANCE"
)


for system in SYSTEM_ORDER:

    print(
        f"\n{system}"
    )

    print(
        f"  score   : "
        f"{balance[system]['score']}"
    )

    print(
        f"  exclude : "
        f"{balance[system]['exclude']}"
    )


print(
    f"\nUnique questions represented : "
    f"{unique_questions}"
)


print(
    "\n[6] FILES SAVED"
)


print(
    "ANNOTATE:"
)

print(
    ANNOTATION_PATH
)


print(
    "\nDO NOT OPEN:"
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
    "SECTION 30 COMPLETE"
)

print(
    "=" * 78
)
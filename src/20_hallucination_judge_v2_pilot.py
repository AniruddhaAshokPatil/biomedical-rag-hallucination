import json
import random
from pathlib import Path

from openai import OpenAI


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

INPUT_PATH = (
    RESULTS_DIR
    / "hallucination_evaluation_inputs.json"
)

V1_PILOT_PATH = (
    RESULTS_DIR
    / "hallucination_judge_pilot.json"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "hallucination_judge_v2_pilot.json"
)


# 2. FROZEN JUDGE SETTINGS

JUDGE_MODEL = (
    "gpt-5.4-2026-03-05"
)

REASONING_EFFORT = "low"

MAX_OUTPUT_TOKENS = 3000

RANDOM_SEED = 42


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


ANONYMOUS_IDS = [
    "A",
    "B",
    "C",
    "D"
]


# 3. STRUCTURED OUTPUT SCHEMA

JUDGE_SCHEMA = {

    "type": "object",

    "properties": {

        "answers": {

            "type": "array",

            "minItems": 4,
            "maxItems": 4,

            "items": {

                "type": "object",

                "properties": {

                    "answer_id": {

                        "type": "string",

                        "enum": [
                            "A",
                            "B",
                            "C",
                            "D"
                        ]
                    },

                    "claims": {

                        "type": "array",

                        "items": {

                            "type": "object",

                            "properties": {

                                "claim": {
                                    "type": "string"
                                },

                                "label": {

                                    "type": "string",

                                    "enum": [
                                        "supported",
                                        "unsupported",
                                        "contradicted"
                                    ]
                                },

                                "evidence_labels": {

                                    "type": "array",

                                    "items": {
                                        "type": "string"
                                    }
                                },

                                "rationale": {
                                    "type": "string"
                                }
                            },

                            "required": [
                                "claim",
                                "label",
                                "evidence_labels",
                                "rationale"
                            ],

                            "additionalProperties":
                                False
                        }
                    }
                },

                "required": [
                    "answer_id",
                    "claims"
                ],

                "additionalProperties":
                    False
            }
        }
    },

    "required": [
        "answers"
    ],

    "additionalProperties":
        False
}


# 4. IMPROVED JUDGE INSTRUCTIONS

JUDGE_INSTRUCTIONS = """
You are performing claim-level hallucination evaluation for
biomedical question answering.

You will receive:

1. One biomedical question.
2. Gold evidence passages from the benchmark article.
3. Four anonymous generated answers labelled A, B, C, and D.

IMPORTANT:

The four answers are NOT evidence.

Judge every factual claim ONLY against the supplied GOLD EVIDENCE.

Do not use outside medical knowledge.
Do not use the gold answer, because it is not provided.
Do not assume which system produced an answer.


============================================================
CLAIM EXTRACTION
============================================================

For each answer, extract atomic factual claims.

An atomic factual claim is one independently verifiable
proposition.

Split sentences containing multiple factual propositions.

Preserve meaningful qualifiers such as:

- may
- suggests
- associated with
- increased
- decreased
- statistically significant
- specific to
- does not establish
- cannot determine

Do not create claims from:

- politeness
- conversational wording
- simple expressions of uncertainty such as
  "I am not sure"

However, an explicit statement ABOUT THE EVIDENCE such as:

"The evidence does not show an association"

IS a factual claim and must be evaluated.


============================================================
SUPPORTED
============================================================

Label a claim SUPPORTED when the gold evidence directly states
the claim OR reasonably entails it.

Support does NOT require identical wording.

A cautious and natural interpretation of reported findings can
count as supported.

For example, if evidence reports that one group has
significantly more underestimation than another group, a
carefully worded statement that estimates were less accurate
for that measured outcome can be supported.

Do not require the article to use the exact same adjective or
sentence if the meaning follows reasonably from the reported
results.


============================================================
CONTRADICTED
============================================================

Label a claim CONTRADICTED only when the gold evidence supports
the opposite conclusion or is clearly incompatible with the
claim.

Example:

Claim:
"The evidence provides no study results."

If a gold evidence passage explicitly reports study results,
the claim is contradicted.


============================================================
UNSUPPORTED
============================================================

Label a claim UNSUPPORTED when:

- the evidence does not establish it,
- it introduces information absent from the evidence,
- or it makes an interpretation that does not reasonably follow
  from the evidence.

A medically plausible statement is still unsupported if the
supplied evidence does not support it.


============================================================
CONSISTENCY REQUIREMENT
============================================================

Apply the SAME evidential threshold to all four anonymous
answers.

Semantically equivalent claims should receive the same label
unless a meaningful qualifier changes the claim.

For example, do not mark essentially the same factual
interpretation supported in Answer A but unsupported in
Answer C without a real semantic reason.

Different wording alone is NOT a reason for a different label.


============================================================
EVIDENCE LABELS
============================================================

For SUPPORTED and CONTRADICTED claims, provide the relevant gold
evidence labels such as G1, G2, etc.

For UNSUPPORTED claims, evidence_labels should normally be [].


============================================================
CRITICAL RULE
============================================================

Other generated answers must NEVER be used as evidence.

Only GOLD EVIDENCE determines support.

Return only the required structured output.
""".strip()


# 5. HELPERS

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


def format_gold_evidence(
    evidence
):

    blocks = []


    for passage in evidence:

        blocks.append(
            f"[{passage['label']}]\n"
            f"Section: {passage['section']}\n"
            f"{passage['text']}"
        )


    return "\n\n".join(
        blocks
    )


# 6. LOAD DATA

print("=" * 75)

print(
    "SECTION 20 - HALLUCINATION JUDGE V2 CONSISTENCY PILOT"
)

print("=" * 75)


records = load_json(
    INPUT_PATH
)

v1_pilot = load_json(
    V1_PILOT_PATH
)


selected_question_ids = (
    v1_pilot[
        "selected_question_ids"
    ]
)


print(
    "\n[1] DATA LOADED"
)

print(
    f"Evaluation records : "
    f"{len(records)}"
)

print(
    f"Pilot questions    : "
    f"{len(selected_question_ids)}"
)


if len(selected_question_ids) != 24:

    raise ValueError(
        "Expected the same 24 questions "
        "used in Section 19."
    )


# 7. INDEX RECORDS

records_by_question = {}


for record in records:

    question_id = (
        record[
            "question_id"
        ]
    )


    records_by_question.setdefault(
        question_id,
        {}
    )


    records_by_question[
        question_id
    ][
        record["system"]
    ] = record


# 8. VALIDATE FOUR SYSTEMS

print(
    "\n[2] INPUT VALIDATION"
)


invalid_questions = []


for question_id in selected_question_ids:

    systems = set(

        records_by_question[
            question_id
        ].keys()
    )


    if systems != set(
        SYSTEM_ORDER
    ):

        invalid_questions.append(
            question_id
        )


print(
    f"Questions missing systems: "
    f"{len(invalid_questions)}"
)


if invalid_questions:

    raise ValueError(
        "Pilot questions do not contain "
        "all four systems."
    )


print(
    "PASS: Every pilot question has "
    "all four system answers."
)


# 9. CREATE BLINDED ANSWER MAPPINGS
# Actual system names are NEVER sent to the judge.
# Example:
# A = advanced_rag
# B = baseline_llm
# C = gold_context_control
# D = basic_rag
# Mapping differs deterministically for each question.

answer_mappings = {}


for question_id in selected_question_ids:

    shuffled_systems = (
        SYSTEM_ORDER.copy()
    )


    question_random = random.Random(
        RANDOM_SEED
        +
        int(question_id)
    )


    question_random.shuffle(
        shuffled_systems
    )


    answer_mappings[
        question_id
    ] = {

        anonymous_id:
            system_name

        for anonymous_id, system_name
        in zip(
            ANONYMOUS_IDS,
            shuffled_systems
        )
    }


print(
    "\n[3] BLINDING"
)

print(
    "PASS: System identities are hidden "
    "from the judge."
)


# 10. LOAD CHECKPOINT

if OUTPUT_PATH.exists():

    existing = load_json(
        OUTPUT_PATH
    )

    judgments = existing.get(
        "judgments",
        []
    )


    print(
        "\n[4] CHECKPOINT FOUND"
    )

    print(
        f"Completed questions: "
        f"{len(judgments)}"
    )


else:

    judgments = []


    print(
        "\n[4] NO EXISTING CHECKPOINT"
    )


completed_question_ids = {

    item[
        "question_id"
    ]

    for item
    in judgments
}


# 11. OPENAI CLIENT

client = OpenAI()


# 12. RUN V2 JUDGE

print(
    "\n[5] RUNNING V2 CONSISTENCY JUDGE"
)


for question_number, question_id in enumerate(
    selected_question_ids,
    start=1
):

    if (
        question_id
        in completed_question_ids
    ):

        print(
            f"Question "
            f"{question_number}/24 "
            f"- {question_id} "
            f"-> already completed"
        )

        continue


    systems_for_question = (
        records_by_question[
            question_id
        ]
    )


    first_record = next(
        iter(
            systems_for_question.values()
        )
    )


    evidence_text = (
        format_gold_evidence(
            first_record[
                "gold_evidence"
            ]
        )
    )


    mapping = (
        answer_mappings[
            question_id
        ]
    )


    answer_blocks = []


    for anonymous_id in ANONYMOUS_IDS:

        system_name = (
            mapping[
                anonymous_id
            ]
        )


        answer = (

            systems_for_question[
                system_name
            ][
                "generated_answer"
            ]
        )


        answer_blocks.append(
            f"ANSWER {anonymous_id}:\n"
            f"{answer}"
        )


    answers_text = "\n\n".join(
        answer_blocks
    )


    user_prompt = (
        f"QUESTION:\n"
        f"{first_record['question']}\n\n"

        f"GOLD EVIDENCE:\n"
        f"{evidence_text}\n\n"

        f"GENERATED ANSWERS:\n"
        f"{answers_text}"
    )


    try:

        response = client.responses.create(

            model=JUDGE_MODEL,

            instructions=(
                JUDGE_INSTRUCTIONS
            ),

            input=user_prompt,

            reasoning={
                "effort":
                    REASONING_EFFORT
            },

            max_output_tokens=(
                MAX_OUTPUT_TOKENS
            ),

            text={

                "verbosity":
                    "low",

                "format": {

                    "type":
                        "json_schema",

                    "name":
                        "hallucination_v2",

                    "strict":
                        True,

                    "schema":
                        JUDGE_SCHEMA
                }
            }
        )


        parsed = json.loads(
            response.output_text
        )


        # 13. VALIDATE ANSWER IDs

        returned_ids = [

            answer[
                "answer_id"
            ]

            for answer
            in parsed[
                "answers"
            ]
        ]


        if (
            set(returned_ids)
            !=
            set(ANONYMOUS_IDS)
        ):

            raise ValueError(
                "Judge did not return exactly "
                "A, B, C, and D."
            )


        # 14. VALIDATE EVIDENCE LABELS

        valid_evidence_labels = {

            passage[
                "label"
            ]

            for passage
            in first_record[
                "gold_evidence"
            ]
        }


        system_results = {}


        for answer_result in parsed[
            "answers"
        ]:

            anonymous_id = (
                answer_result[
                    "answer_id"
                ]
            )


            system_name = (
                mapping[
                    anonymous_id
                ]
            )


            claims = (
                answer_result[
                    "claims"
                ]
            )


            invalid_labels = []


            for claim in claims:

                for evidence_label in (
                    claim[
                        "evidence_labels"
                    ]
                ):

                    if (
                        evidence_label
                        not in
                        valid_evidence_labels
                    ):

                        invalid_labels.append(
                            evidence_label
                        )


            # CLAIM COUNTS

            supported = sum(

                1

                for claim in claims

                if (
                    claim["label"]
                    ==
                    "supported"
                )
            )


            unsupported = sum(

                1

                for claim in claims

                if (
                    claim["label"]
                    ==
                    "unsupported"
                )
            )


            contradicted = sum(

                1

                for claim in claims

                if (
                    claim["label"]
                    ==
                    "contradicted"
                )
            )


            total_claims = len(
                claims
            )


            if total_claims > 0:

                hallucination_rate = (

                    (
                        unsupported
                        +
                        contradicted
                    )

                    /
                    total_claims
                )


                groundedness = (

                    supported
                    /
                    total_claims
                )


            else:

                hallucination_rate = None

                groundedness = None


            system_results[
                system_name
            ] = {

                "anonymous_answer_id":
                    anonymous_id,

                "generated_answer":

                    systems_for_question[
                        system_name
                    ][
                        "generated_answer"
                    ],

                "claims":
                    claims,

                "claim_counts": {

                    "supported":
                        supported,

                    "unsupported":
                        unsupported,

                    "contradicted":
                        contradicted,

                    "total":
                        total_claims
                },

                "hallucination_rate":
                    hallucination_rate,

                "groundedness":
                    groundedness,

                "invalid_evidence_labels":
                    invalid_labels
            }


        # 15. STORE QUESTION JUDGMENT

        input_tokens = (

            response.usage.input_tokens

            if response.usage

            else 0
        )


        output_tokens = (

            response.usage.output_tokens

            if response.usage

            else 0
        )


        question_judgment = {

            "question_id":
                question_id,

            "gold_decision":
                first_record[
                    "gold_decision"
                ],

            "blind_mapping":
                mapping,

            "systems":
                system_results,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens
        }


        judgments.append(
            question_judgment
        )


        completed_question_ids.add(
            question_id
        )


        print(
            f"Question "
            f"{question_number}/24 "
            f"- {question_id} "
            f"-> complete"
        )


        # Checkpoint after every question

        save_json(
            {

                "judge_version":
                    "v2_consistency",

                "judge_model":
                    JUDGE_MODEL,

                "reasoning_effort":
                    REASONING_EFFORT,

                "random_seed":
                    RANDOM_SEED,

                "selected_question_ids":
                    selected_question_ids,

                "judgments":
                    judgments
            },
            OUTPUT_PATH
        )


    except Exception as error:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "JUDGE ERROR"
        )

        print(
            "=" * 75
        )

        print(
            f"Question ID: "
            f"{question_id}"
        )

        print(
            f"Error: "
            f"{error}"
        )

        print(
            "\nCompleted questions are "
            "already checkpointed."
        )

        print(
            "Fix the issue and rerun "
            "the script to resume."
        )

        raise


# 16. VALIDATION

print(
    "\n[6] V2 PILOT VALIDATION"
)


print(
    f"Completed questions : "
    f"{len(judgments)}"
)


total_system_judgments = sum(

    len(
        judgment[
            "systems"
        ]
    )

    for judgment
    in judgments
)


print(
    f"System judgments    : "
    f"{total_system_judgments}"
)


invalid_evidence_cases = 0

zero_claim_cases = 0


for judgment in judgments:

    for system_name in SYSTEM_ORDER:

        result = (

            judgment[
                "systems"
            ][
                system_name
            ]
        )


        if result[
            "invalid_evidence_labels"
        ]:

            invalid_evidence_cases += 1


        if (
            result[
                "claim_counts"
            ][
                "total"
            ]
            ==
            0
        ):

            zero_claim_cases += 1


print(
    f"Invalid evidence cases: "
    f"{invalid_evidence_cases}"
)

print(
    f"Zero-claim answers     : "
    f"{zero_claim_cases}"
)


if len(judgments) != 24:

    raise ValueError(
        "Expected 24 completed questions."
    )


if total_system_judgments != 96:

    raise ValueError(
        "Expected 96 system judgments."
    )


# 17. SYSTEM SUMMARIES

print(
    "\n[7] V2 HALLUCINATION SUMMARY"
)


system_summaries = {}


for system_name in SYSTEM_ORDER:

    system_results = []


    for judgment in judgments:

        system_results.append(

            judgment[
                "systems"
            ][
                system_name
            ]
        )


    valid_results = [

        result

        for result
        in system_results

        if (
            result[
                "hallucination_rate"
            ]
            is not None
        )
    ]


    total_claims = sum(

        result[
            "claim_counts"
        ][
            "total"
        ]

        for result
        in valid_results
    )


    total_supported = sum(

        result[
            "claim_counts"
        ][
            "supported"
        ]

        for result
        in valid_results
    )


    total_unsupported = sum(

        result[
            "claim_counts"
        ][
            "unsupported"
        ]

        for result
        in valid_results
    )


    total_contradicted = sum(

        result[
            "claim_counts"
        ][
            "contradicted"
        ]

        for result
        in valid_results
    )


    mean_hallucination = (

        sum(

            result[
                "hallucination_rate"
            ]

            for result
            in valid_results
        )

        /
        len(
            valid_results
        )
    )


    mean_groundedness = (

        sum(

            result[
                "groundedness"
            ]

            for result
            in valid_results
        )

        /
        len(
            valid_results
        )
    )


    micro_hallucination = (

        (
            total_unsupported
            +
            total_contradicted
        )

        /
        total_claims
    )


    micro_groundedness = (

        total_supported
        /
        total_claims
    )


    summary = {

        "answers":
            len(
                valid_results
            ),

        "total_claims":
            total_claims,

        "supported_claims":
            total_supported,

        "unsupported_claims":
            total_unsupported,

        "contradicted_claims":
            total_contradicted,

        "mean_answer_hallucination_rate":
            mean_hallucination,

        "mean_answer_groundedness":
            mean_groundedness,

        "micro_claim_hallucination_rate":
            micro_hallucination,

        "micro_claim_groundedness":
            micro_groundedness
    }


    system_summaries[
        system_name
    ] = summary


    print(
        f"\n{system_name}"
    )

    print(
        f"  Answers evaluated   : "
        f"{summary['answers']}"
    )

    print(
        f"  Total claims        : "
        f"{summary['total_claims']}"
    )

    print(
        f"  Supported           : "
        f"{summary['supported_claims']}"
    )

    print(
        f"  Unsupported         : "
        f"{summary['unsupported_claims']}"
    )

    print(
        f"  Contradicted        : "
        f"{summary['contradicted_claims']}"
    )

    print(
        f"  Mean hallucination  : "
        f"{summary['mean_answer_hallucination_rate']:.4f}"
    )

    print(
        f"  Mean groundedness   : "
        f"{summary['mean_answer_groundedness']:.4f}"
    )

    print(
        f"  Micro hallucination : "
        f"{summary['micro_claim_hallucination_rate']:.4f}"
    )


# 18. V1 VS V2 COMPARISON

print(
    "\n[8] V1 VS V2 MICRO HALLUCINATION"
)


v1_summaries = (
    v1_pilot[
        "system_summaries"
    ]
)


print(
    f"{'System':<24}"
    f"{'V1':>10}"
    f"{'V2':>10}"
    f"{'Change':>10}"
)

print(
    "-" * 54
)


for system_name in SYSTEM_ORDER:

    v1_rate = (

        v1_summaries[
            system_name
        ][
            "micro_claim_hallucination_rate"
        ]
    )


    v2_rate = (

        system_summaries[
            system_name
        ][
            "micro_claim_hallucination_rate"
        ]
    )


    change = (
        v2_rate
        -
        v1_rate
    )


    print(
        f"{system_name:<24}"
        f"{v1_rate:>10.4f}"
        f"{v2_rate:>10.4f}"
        f"{change:>+10.4f}"
    )


# 19. TARGETED CONSISTENCY EXAMPLES

TARGET_IDS = [
    "17453263",
    "15879722"
]


print(
    "\n[9] TARGETED CONSISTENCY CHECK"
)


for target_id in TARGET_IDS:

    target = next(

        (
            judgment

            for judgment
            in judgments

            if (
                judgment[
                    "question_id"
                ]
                ==
                target_id
            )
        ),

        None
    )


    if target is None:

        continue


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"QUESTION ID: "
        f"{target_id}"
    )

    print(
        "=" * 70
    )


    for system_name in SYSTEM_ORDER:

        result = (

            target[
                "systems"
            ][
                system_name
            ]
        )


        print(
            f"\nSYSTEM: "
            f"{system_name}"
        )


        print(
            "ANSWER:"
        )

        print(
            result[
                "generated_answer"
            ]
        )


        print(
            "\nCLAIMS:"
        )


        if not result[
            "claims"
        ]:

            print(
                "  [NO FACTUAL CLAIMS]"
            )


        for claim in result[
            "claims"
        ]:

            print(
                f"  - "
                f"{claim['label'].upper()}: "
                f"{claim['claim']}"
            )


# 20. TOKEN USAGE

total_input_tokens = sum(

    judgment.get(
        "input_tokens",
        0
    )

    for judgment
    in judgments
)


total_output_tokens = sum(

    judgment.get(
        "output_tokens",
        0
    )

    for judgment
    in judgments
)


print(
    "\n[10] TOKEN USAGE"
)


print(
    f"Input tokens  : "
    f"{total_input_tokens}"
)

print(
    f"Output tokens : "
    f"{total_output_tokens}"
)

print(
    f"Total tokens  : "
    f"{total_input_tokens + total_output_tokens}"
)


# 21. SAVE FINAL RESULT

save_json(
    {

        "judge_version":
            "v2_consistency",

        "judge_model":
            JUDGE_MODEL,

        "reasoning_effort":
            REASONING_EFFORT,

        "random_seed":
            RANDOM_SEED,

        "selected_question_ids":
            selected_question_ids,

        "system_summaries":
            system_summaries,

        "judgments":
            judgments
    },
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
    + "=" * 75
)

print(
    "SECTION 20 COMPLETE"
)

print(
    "=" * 75
)
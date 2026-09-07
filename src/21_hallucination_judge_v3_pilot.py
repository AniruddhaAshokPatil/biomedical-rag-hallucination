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

V2_PILOT_PATH = (
    RESULTS_DIR
    / "hallucination_judge_v2_pilot.json"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "hallucination_judge_v3_pilot.json"
)


# 2. FROZEN SETTINGS

JUDGE_MODEL = (
    "gpt-5.4-2026-03-05"
)

REASONING_EFFORT = "low"

MAX_OUTPUT_TOKENS = 3500

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

                    # ------------------------------------------------
                    # These are the claims that ENTER the
                    # hallucination denominator.
                    # ------------------------------------------------

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
                    },

                    # Statements deliberately excluded from
                    # hallucination scoring.
                  

                    "excluded_statements": {

                        "type": "array",

                        "items": {

                            "type": "object",

                            "properties": {

                                "statement": {
                                    "type": "string"
                                },

                                "reason": {

                                    "type": "string",

                                    "enum": [
                                        "abstention_or_uncertainty",
                                        "evidence_availability_meta_statement",
                                        "non_factual"
                                    ]
                                }
                            },

                            "required": [
                                "statement",
                                "reason"
                            ],

                            "additionalProperties":
                                False
                        }
                    }
                },

                "required": [
                    "answer_id",
                    "claims",
                    "excluded_statements"
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


# 4. V3 JUDGE INSTRUCTION

JUDGE_INSTRUCTIONS = """
You are performing claim-level hallucination evaluation for
biomedical question answering.

You will receive:

1. One biomedical research question.
2. Gold evidence passages from the benchmark article.
3. Four anonymous generated answers labelled A, B, C, and D.

The generated answers are NOT evidence.

Judge claims ONLY against the supplied GOLD EVIDENCE.

Do not use outside medical knowledge.
Do not infer which system produced an answer.
Apply exactly the same evidential standard to A, B, C, and D.


============================================================
PRIMARY EVALUATION TARGET
============================================================

The hallucination metric is intended to measure unsupported
or contradicted BIOMEDICAL OR RESEARCH FACTUAL CONTENT.

Only factual claims that describe:

- patients
- diseases
- interventions
- outcomes
- associations
- diagnostic performance
- numerical findings
- study design
- study populations
- research findings
- substantive interpretations of study findings

should normally enter the hallucination denominator.


============================================================
CLAIM EXTRACTION
============================================================

Split each answer into atomic factual claims.

An atomic claim contains one independently verifiable
biomedical or research proposition.

If a sentence contains multiple propositions, split them.

Preserve important qualifiers such as:

- may
- suggests
- associated with
- increased
- decreased
- statistically significant
- specific to
- does not establish
- cannot demonstrate


============================================================
EXCLUDE PURE ABSTENTION / UNCERTAINTY
============================================================

Do NOT count pure uncertainty or refusal statements as
biomedical hallucination claims.

Examples to EXCLUDE:

"I am not sure."

"I cannot answer confidently."

"Possibly."

"I cannot confirm this from the information provided."

These should appear under excluded_statements with reason:

abstention_or_uncertainty


============================================================
EXCLUDE EVIDENCE-AVAILABILITY META STATEMENTS
============================================================

Statements that ONLY describe whether the prompt contains
enough evidence or whether results are present should NOT
enter the biomedical hallucination denominator.

Examples to EXCLUDE:

"The provided evidence does not include the results."

"There is not enough information here to answer."

"The supplied passages do not allow a conclusion."

These should appear under excluded_statements with reason:

evidence_availability_meta_statement


============================================================
IMPORTANT DISTINCTION
============================================================

Do NOT exclude substantive statements about the actual
scientific findings merely because they mention evidence or
the study.

For example:

"The study did not establish a causal relationship."

"The results suggest an association."

"The study found no significant difference."

These ARE substantive research factual claims and should be
evaluated normally as supported, unsupported, or contradicted.

The distinction is:

META / ACCESS statement:
"The supplied evidence does not contain results."
-> EXCLUDE

SUBSTANTIVE scientific statement:
"The study did not find an effect."
-> EVALUATE


============================================================
SUPPORTED
============================================================

SUPPORTED means the gold evidence directly states or
reasonably entails the claim.

Support does not require identical wording.

Natural and cautious interpretations of reported findings
may count as supported.

Do not require the article to use the exact same adjective
when the interpretation reasonably follows from the data.


============================================================
CONTRADICTED
============================================================

CONTRADICTED means the gold evidence supports the opposite
scientific conclusion or is clearly incompatible with the
claim.

Use this category only for substantive biomedical or research
factual claims that enter the hallucination denominator.


============================================================
UNSUPPORTED
============================================================

UNSUPPORTED means:

- the claim introduces information absent from the evidence,
- the evidence does not establish the claim,
- or the interpretation goes beyond what reasonably follows.

Medical plausibility alone is not sufficient.


============================================================
CONSISTENCY
============================================================

Apply the SAME evidential threshold to all four answers.

Semantically equivalent claims should receive the same label
unless an important qualifier genuinely changes their meaning.

Different wording alone must not cause different labels.


============================================================
EVIDENCE LABELS
============================================================

For supported and contradicted claims, provide relevant gold
labels such as G1, G2, etc.

For unsupported claims, evidence_labels should normally be [].


============================================================
OTHER GENERATED ANSWERS
============================================================

Never use one generated answer to judge another generated
answer.

Only the supplied GOLD EVIDENCE determines factual support.


============================================================
OUTPUT
============================================================

For each answer return:

1. claims
   - ONLY claims entering hallucination scoring

2. excluded_statements
   - abstention/uncertainty
   - evidence-availability meta-statements
   - non-factual wording

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
    "SECTION 21 - HALLUCINATION JUDGE V3 FINAL PILOT"
)

print("=" * 75)


records = load_json(
    INPUT_PATH
)

v2_pilot = load_json(
    V2_PILOT_PATH
)


selected_question_ids = (
    v2_pilot[
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
        "Expected the same 24 pilot questions."
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


# 8. VALIDATE INPUT

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
        "Some pilot questions do not "
        "contain all four systems."
    )


print(
    "PASS: All 24 questions contain "
    "four system outputs."
)


# 9. RECREATE SAME BLINDED ORDER AS V2

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
    "PASS: Same deterministic blinded "
    "presentation design as V2."
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


# 12. RUN V3 JUDGE

print(
    "\n[5] RUNNING V3 JUDGE"
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


        generated_answer = (

            systems_for_question[
                system_name
            ][
                "generated_answer"
            ]
        )


        answer_blocks.append(
            f"ANSWER {anonymous_id}:\n"
            f"{generated_answer}"
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
                        "hallucination_v3",

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


        # 14. PROCESS EACH ANONYMOUS ANSWER

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


            excluded_statements = (

                answer_result[
                    "excluded_statements"
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


            # 15. CLAIM COUNTS

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


            # 16. EXCLUSION COUNTS

            exclusion_counts = {

                "abstention_or_uncertainty":
                    0,

                "evidence_availability_meta_statement":
                    0,

                "non_factual":
                    0
            }


            for excluded in (
                excluded_statements
            ):

                exclusion_counts[
                    excluded[
                        "reason"
                    ]
                ] += 1


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

                "excluded_statements":
                    excluded_statements,

                "exclusion_counts":
                    exclusion_counts,

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


        # 17. SAVE QUESTION JUDGMENT

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


        # Checkpoint after every successful question

        save_json(
            {

                "judge_version":
                    "v3_final",

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
            "\nCompleted questions "
            "are already saved."
        )

        print(
            "Rerun this script "
            "after fixing the issue."
        )

        raise


# 18. VALIDATION

print(
    "\n[6] V3 PILOT VALIDATION"
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

zero_claim_answers = 0

total_excluded_statements = 0


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

            zero_claim_answers += 1


        total_excluded_statements += len(

            result[
                "excluded_statements"
            ]
        )


print(
    f"Invalid evidence cases : "
    f"{invalid_evidence_cases}"
)

print(
    f"Zero-claim answers      : "
    f"{zero_claim_answers}"
)

print(
    f"Excluded statements     : "
    f"{total_excluded_statements}"
)


if len(judgments) != 24:

    raise ValueError(
        "Expected 24 completed questions."
    )


if total_system_judgments != 96:

    raise ValueError(
        "Expected 96 system judgments."
    )


# 19. SYSTEM SUMMARIES

print(
    "\n[7] V3 HALLUCINATION SUMMARY"
)


system_summaries = {}


for system_name in SYSTEM_ORDER:

    system_results = [

        judgment[
            "systems"
        ][
            system_name
        ]

        for judgment
        in judgments
    ]


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


    excluded_total = sum(

        len(
            result[
                "excluded_statements"
            ]
        )

        for result
        in system_results
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

        if valid_results

        else None
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

        if valid_results

        else None
    )


    micro_hallucination = (

        (
            total_unsupported
            +
            total_contradicted
        )

        /
        total_claims

        if total_claims > 0

        else None
    )


    micro_groundedness = (

        total_supported
        /
        total_claims

        if total_claims > 0

        else None
    )


    summary = {

        "answers_total":
            len(
                system_results
            ),

        "answers_with_factual_claims":
            len(
                valid_results
            ),

        "zero_claim_answers":
            (
                len(system_results)
                -
                len(valid_results)
            ),

        "total_claims":
            total_claims,

        "supported_claims":
            total_supported,

        "unsupported_claims":
            total_unsupported,

        "contradicted_claims":
            total_contradicted,

        "excluded_statements":
            excluded_total,

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
        f"  Answers total          : "
        f"{summary['answers_total']}"
    )

    print(
        f"  Answers with claims    : "
        f"{summary['answers_with_factual_claims']}"
    )

    print(
        f"  Zero-claim answers     : "
        f"{summary['zero_claim_answers']}"
    )

    print(
        f"  Excluded statements    : "
        f"{summary['excluded_statements']}"
    )

    print(
        f"  Total claims           : "
        f"{summary['total_claims']}"
    )

    print(
        f"  Supported              : "
        f"{summary['supported_claims']}"
    )

    print(
        f"  Unsupported            : "
        f"{summary['unsupported_claims']}"
    )

    print(
        f"  Contradicted           : "
        f"{summary['contradicted_claims']}"
    )

    print(
        f"  Mean hallucination     : "
        f"{summary['mean_answer_hallucination_rate']:.4f}"
    )

    print(
        f"  Mean groundedness      : "
        f"{summary['mean_answer_groundedness']:.4f}"
    )

    print(
        f"  Micro hallucination    : "
        f"{summary['micro_claim_hallucination_rate']:.4f}"
    )


# 20. V2 VS V3 COMPARISON

print(
    "\n[8] V2 VS V3 MICRO HALLUCINATION"
)


v2_summaries = (
    v2_pilot[
        "system_summaries"
    ]
)


print(
    f"{'System':<24}"
    f"{'V2':>10}"
    f"{'V3':>10}"
    f"{'Change':>10}"
)

print(
    "-" * 54
)


for system_name in SYSTEM_ORDER:

    v2_rate = (

        v2_summaries[
            system_name
        ][
            "micro_claim_hallucination_rate"
        ]
    )


    v3_rate = (

        system_summaries[
            system_name
        ][
            "micro_claim_hallucination_rate"
        ]
    )


    change = (
        v3_rate
        -
        v2_rate
    )


    print(
        f"{system_name:<24}"
        f"{v2_rate:>10.4f}"
        f"{v3_rate:>10.4f}"
        f"{change:>+10.4f}"
    )


# 21. TARGETED EXCLUSION CHECK

TARGET_IDS = [
    "17453263",
    "21745056",
    "19664156"
]


print(
    "\n[9] TARGETED EXCLUSION CHECK"
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
            "\nSCORED CLAIMS:"
        )


        if not result[
            "claims"
        ]:

            print(
                "  [NO SCORED FACTUAL CLAIMS]"
            )


        for claim in result[
            "claims"
        ]:

            print(
                f"  - "
                f"{claim['label'].upper()}: "
                f"{claim['claim']}"
            )


        print(
            "\nEXCLUDED:"
        )


        if not result[
            "excluded_statements"
        ]:

            print(
                "  [NONE]"
            )


        for excluded in (
            result[
                "excluded_statements"
            ]
        ):

            print(
                f"  - "
                f"{excluded['reason']}: "
                f"{excluded['statement']}"
            )


# 22. TOKEN USAGE

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


# 23. SAVE FINAL PILOT

save_json(
    {

        "judge_version":
            "v3_final",

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
    "SECTION 21 COMPLETE"
)

print(
    "=" * 75
)
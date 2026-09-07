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

OUTPUT_PATH = (
    RESULTS_DIR
    / "development_hallucination_v3.json"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "development_hallucination_v3_summary.json"
)


# 2. FROZEN JUDGE SETTINGS

JUDGE_VERSION = "v3_final"

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


# 4. FROZEN V3 JUDGE INSTRUCTIONS

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


# 6. LOAD DEVELOPMENT EVALUATION INPUTS

print("=" * 75)

print(
    "SECTION 22 - FULL DEVELOPMENT HALLUCINATION EVALUATION"
)

print("=" * 75)


records = load_json(
    INPUT_PATH
)


print(
    "\n[1] DATA LOADED"
)

print(
    f"Evaluation records: "
    f"{len(records)}"
)


if len(records) != 2000:

    raise ValueError(
        "Expected exactly 2,000 records."
    )


# 7. GROUP BY QUESTION

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


question_ids = sorted(
    records_by_question.keys()
)


print(
    f"Unique questions  : "
    f"{len(question_ids)}"
)


if len(question_ids) != 500:

    raise ValueError(
        "Expected exactly 500 "
        "development questions."
    )


# 8. INPUT VALIDATION

print(
    "\n[2] INPUT VALIDATION"
)


missing_system_questions = []

wrong_split_questions = []


for question_id in question_ids:

    systems = set(

        records_by_question[
            question_id
        ].keys()
    )


    if systems != set(
        SYSTEM_ORDER
    ):

        missing_system_questions.append(
            question_id
        )


    first_record = next(
        iter(
            records_by_question[
                question_id
            ].values()
        )
    )


    if not first_record[
        "gold_evidence"
    ]:

        raise ValueError(
            f"No gold evidence for "
            f"{question_id}"
        )


print(
    f"Questions missing systems : "
    f"{len(missing_system_questions)}"
)


if missing_system_questions:

    raise ValueError(
        "Some questions do not contain "
        "all four systems."
    )


print(
    "PASS: 500 questions x 4 systems "
    "are ready."
)


# 9. DETERMINISTIC BLINDING

answer_mappings = {}


for question_id in question_ids:

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
    "PASS: Every question uses "
    "deterministic anonymous A/B/C/D labels."
)


# 10. LOAD CHECKPOINT

if OUTPUT_PATH.exists():

    existing = load_json(
        OUTPUT_PATH
    )


    if (
        existing.get(
            "judge_version"
        )
        != JUDGE_VERSION
    ):

        raise ValueError(
            "Existing checkpoint uses "
            "a different judge version."
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

    print(
        "Starting full hallucination evaluation."
    )


completed_question_ids = {

    judgment[
        "question_id"
    ]

    for judgment
    in judgments
}


print(
    f"Remaining questions: "
    f"{500 - len(completed_question_ids)}"
)


# 11. OPENAI CLIENT

client = OpenAI()


# 12. FULL JUDGE RUN

print(
    "\n[5] RUNNING FULL V3 JUDGE"
)


for question_number, question_id in enumerate(
    question_ids,
    start=1
):

    if (
        question_id
        in completed_question_ids
    ):

        print(
            f"Question "
            f"{question_number}/500 "
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


        # 13. VALIDATE RETURNED ANSWER IDS

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
            len(returned_ids) != 4
            or
            set(returned_ids)
            !=
            set(ANONYMOUS_IDS)
        ):

            raise ValueError(
                "Judge did not return exactly "
                "A, B, C, and D."
            )


        # 14. VALID EVIDENCE LABELS

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


        # 15. PROCESS EACH ANSWER

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


            # Claim counts

            supported = sum(

                1

                for claim in claims

                if (
                    claim[
                        "label"
                    ]
                    ==
                    "supported"
                )
            )


            unsupported = sum(

                1

                for claim in claims

                if (
                    claim[
                        "label"
                    ]
                    ==
                    "unsupported"
                )
            )


            contradicted = sum(

                1

                for claim in claims

                if (
                    claim[
                        "label"
                    ]
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


                contradiction_rate = (

                    contradicted
                    /
                    total_claims
                )


            else:

                hallucination_rate = None

                groundedness = None

                contradiction_rate = None


            # Exclusion counts

            exclusion_counts = {

                "abstention_or_uncertainty":
                    0,

                "evidence_availability_meta_statement":
                    0,

                "non_factual":
                    0
            }


            for excluded in excluded_statements:

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

                "generated_decision":

                    systems_for_question[
                        system_name
                    ][
                        "generated_decision"
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

                "contradiction_rate":
                    contradiction_rate,

                "invalid_evidence_labels":
                    invalid_labels
            }


        # 16. TOKEN USAGE

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


        # 17. STORE QUESTION

        question_judgment = {

            "question_id":
                question_id,

            "question":
                first_record[
                    "question"
                ],

            # Stored for later analysis.
            # This was NOT sent to the judge.
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
            f"{question_number}/500 "
            f"- {question_id} "
            f"-> complete"
        )


        # 18. CHECKPOINT AFTER EVERY SUCCESS

        checkpoint = {

            "split":
                "development",

            "judge_version":
                JUDGE_VERSION,

            "judge_model":
                JUDGE_MODEL,

            "reasoning_effort":
                REASONING_EFFORT,

            "max_output_tokens":
                MAX_OUTPUT_TOKENS,

            "random_seed":
                RANDOM_SEED,

            "expected_questions":
                500,

            "completed_questions":
                len(judgments),

            "judgments":
                judgments
        }


        save_json(
            checkpoint,
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
            "\nEverything completed before "
            "this error is already saved."
        )


        print(
            "Fix the issue and rerun this "
            "same script to resume."
        )


        raise


# 19. FINAL COMPLETION CHECK

print(
    "\n[6] FINAL VALIDATION"
)


print(
    f"Completed questions : "
    f"{len(judgments)}"
)


if len(judgments) != 500:

    raise ValueError(
        "Full hallucination evaluation "
        "is incomplete."
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


if total_system_judgments != 2000:

    raise ValueError(
        "Expected exactly 2,000 "
        "system judgments."
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


# 20. FULL SYSTEM SUMMARIES

print(
    "\n[7] FULL HALLUCINATION SUMMARY"
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


    supported_claims = sum(

        result[
            "claim_counts"
        ][
            "supported"
        ]

        for result
        in valid_results
    )


    unsupported_claims = sum(

        result[
            "claim_counts"
        ][
            "unsupported"
        ]

        for result
        in valid_results
    )


    contradicted_claims = sum(

        result[
            "claim_counts"
        ][
            "contradicted"
        ]

        for result
        in valid_results
    )


    excluded_statements = sum(

        len(
            result[
                "excluded_statements"
            ]
        )

        for result
        in system_results
    )


    abstention_exclusions = sum(

        result[
            "exclusion_counts"
        ][
            "abstention_or_uncertainty"
        ]

        for result
        in system_results
    )


    evidence_meta_exclusions = sum(

        result[
            "exclusion_counts"
        ][
            "evidence_availability_meta_statement"
        ]

        for result
        in system_results
    )


    non_factual_exclusions = sum(

        result[
            "exclusion_counts"
        ][
            "non_factual"
        ]

        for result
        in system_results
    )


    if valid_results:

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


    else:

        mean_hallucination = None

        mean_groundedness = None


    if total_claims > 0:

        micro_hallucination = (

            (
                unsupported_claims
                +
                contradicted_claims
            )

            /
            total_claims
        )


        micro_groundedness = (

            supported_claims
            /
            total_claims
        )


        micro_contradiction = (

            contradicted_claims
            /
            total_claims
        )


    else:

        micro_hallucination = None

        micro_groundedness = None

        micro_contradiction = None


    zero_claim_count = (

        len(
            system_results
        )

        -
        len(
            valid_results
        )
    )


    zero_claim_rate = (

        zero_claim_count
        /
        len(
            system_results
        )
    )


    summary = {

        "answers_total":
            len(
                system_results
            ),

        "answers_with_scored_claims":
            len(
                valid_results
            ),

        "zero_claim_answers":
            zero_claim_count,

        "zero_claim_rate":
            zero_claim_rate,

        "total_claims":
            total_claims,

        "supported_claims":
            supported_claims,

        "unsupported_claims":
            unsupported_claims,

        "contradicted_claims":
            contradicted_claims,

        "excluded_statements":
            excluded_statements,

        "abstention_exclusions":
            abstention_exclusions,

        "evidence_meta_exclusions":
            evidence_meta_exclusions,

        "non_factual_exclusions":
            non_factual_exclusions,

        "mean_answer_hallucination_rate":
            mean_hallucination,

        "mean_answer_groundedness":
            mean_groundedness,

        "micro_claim_hallucination_rate":
            micro_hallucination,

        "micro_claim_groundedness":
            micro_groundedness,

        "micro_claim_contradiction_rate":
            micro_contradiction
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
        f"{summary['answers_with_scored_claims']}"
    )


    print(
        f"  Zero-claim answers     : "
        f"{summary['zero_claim_answers']} "
        f"({summary['zero_claim_rate'] * 100:.2f}%)"
    )


    print(
        f"  Total factual claims   : "
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


    print(
        f"  Micro groundedness     : "
        f"{summary['micro_claim_groundedness']:.4f}"
    )


    print(
        f"  Micro contradiction    : "
        f"{summary['micro_claim_contradiction_rate']:.4f}"
    )


    print(
        f"  Excluded statements    : "
        f"{summary['excluded_statements']}"
    )


# 21. MAIN COMPARISON TABLE

print(
    "\n[8] MAIN HALLUCINATION COMPARISON"
)


print(
    f"{'System':<24}"
    f"{'Micro Hall.':>14}"
    f"{'Grounded':>12}"
    f"{'Mean Hall.':>12}"
    f"{'Zero Claim':>12}"
)

print(
    "-" * 74
)


for system_name in SYSTEM_ORDER:

    summary = (
        system_summaries[
            system_name
        ]
    )


    print(
        f"{system_name:<24}"
        f"{summary['micro_claim_hallucination_rate']:>14.4f}"
        f"{summary['micro_claim_groundedness']:>12.4f}"
        f"{summary['mean_answer_hallucination_rate']:>12.4f}"
        f"{summary['zero_claim_rate']:>12.4f}"
    )


# 22. GOLD-LABEL STRATIFIED SUMMARY

print(
    "\n[9] HALLUCINATION BY GOLD DECISION"
)


label_summaries = {}


for system_name in SYSTEM_ORDER:

    label_summaries[
        system_name
    ] = {}


    print(
        f"\n{system_name}"
    )


    for gold_label in [
        "yes",
        "no",
        "maybe"
    ]:

        matching_results = []


        for judgment in judgments:

            if (
                judgment[
                    "gold_decision"
                ]
                != gold_label
            ):

                continue


            result = (

                judgment[
                    "systems"
                ][
                    system_name
                ]
            )


            if (
                result[
                    "hallucination_rate"
                ]
                is not None
            ):

                matching_results.append(
                    result
                )


        total_claims = sum(

            result[
                "claim_counts"
            ][
                "total"
            ]

            for result
            in matching_results
        )


        hallucinated_claims = sum(

            result[
                "claim_counts"
            ][
                "unsupported"
            ]

            +
            result[
                "claim_counts"
            ][
                "contradicted"
            ]

            for result
            in matching_results
        )


        if total_claims > 0:

            rate = (
                hallucinated_claims
                /
                total_claims
            )

        else:

            rate = None


        label_summaries[
            system_name
        ][
            gold_label
        ] = {

            "answers_with_claims":
                len(
                    matching_results
                ),

            "total_claims":
                total_claims,

            "hallucinated_claims":
                hallucinated_claims,

            "micro_hallucination_rate":
                rate
        }


        rate_text = (

            f"{rate:.4f}"

            if rate is not None

            else "N/A"
        )


        print(
            f"  {gold_label:<6} "
            f"answers={len(matching_results):>3} "
            f"claims={total_claims:>4} "
            f"hallucination={rate_text}"
        )


# 23. TOKEN USAGE

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
    "\n[10] JUDGE TOKEN USAGE"
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


# 24. SAVE FINAL SUMMARY

summary_output = {

    "split":
        "development",

    "questions":
        500,

    "system_judgments":
        2000,

    "judge_version":
        JUDGE_VERSION,

    "judge_model":
        JUDGE_MODEL,

    "reasoning_effort":
        REASONING_EFFORT,

    "max_output_tokens":
        MAX_OUTPUT_TOKENS,

    "random_seed":
        RANDOM_SEED,

    "metric_definition": {

        "hallucination":
            (
                "(unsupported claims + "
                "contradicted claims) / "
                "scored factual claims"
            ),

        "groundedness":
            (
                "supported claims / "
                "scored factual claims"
            ),

        "zero_claim_answers":
            (
                "Answers containing no scored "
                "biomedical/research factual claims"
            )
    },

    "validation": {

        "invalid_evidence_cases":
            invalid_evidence_cases,

        "zero_claim_answers_all_systems":
            zero_claim_answers,

        "excluded_statements_all_systems":
            total_excluded_statements
    },

    "system_summaries":
        system_summaries,

    "gold_label_summaries":
        label_summaries,

    "token_usage": {

        "input":
            total_input_tokens,

        "output":
            total_output_tokens,

        "total":
            (
                total_input_tokens
                +
                total_output_tokens
            )
    }
}


save_json(
    summary_output,
    SUMMARY_PATH
)


# Re-save completed detailed results

save_json(
    {

        "split":
            "development",

        "judge_version":
            JUDGE_VERSION,

        "judge_model":
            JUDGE_MODEL,

        "reasoning_effort":
            REASONING_EFFORT,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "random_seed":
            RANDOM_SEED,

        "expected_questions":
            500,

        "completed_questions":
            len(judgments),

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
    SUMMARY_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 22 COMPLETE"
)

print(
    "=" * 75
)
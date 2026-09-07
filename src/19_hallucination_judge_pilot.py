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
    / "hallucination_judge_pilot.json"
)


# 2. JUDGE SETTINGS

JUDGE_MODEL = (
    "gpt-5.4-2026-03-05"
)

REASONING_EFFORT = "low"

MAX_OUTPUT_TOKENS = 1200

RANDOM_SEED = 42

# Balanced pilot:
# 8 YES
# 8 NO
# 8 MAYBE
# = 24 questions
# Each question has four systems:
# 24 x 4 = 96 judge requests

QUESTIONS_PER_LABEL = 8


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]

# 3. STRUCTURED OUTPUT SCHEMA

JUDGE_SCHEMA = {

    "type": "object",

    "properties": {

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
        "claims"
    ],

    "additionalProperties":
        False
}


# 4. JUDGE INSTRUCTIONS

JUDGE_INSTRUCTIONS = """
You are evaluating hallucination in biomedical question answering.

You will receive:
1. A biomedical question.
2. A generated answer.
3. Gold evidence passages from the benchmark article.

Evaluate ONLY against the supplied gold evidence.
Do not use outside medical knowledge.
Do not use assumptions about what is generally true in medicine.

First split the generated answer into atomic factual claims.

An atomic claim should contain one independently verifiable factual proposition.
If a sentence contains multiple factual propositions, split it into multiple claims.

Preserve important qualifiers such as:
- may
- suggests
- associated with
- specific to
- increased
- decreased
- not established

For every factual claim assign exactly one label:

SUPPORTED:
The gold evidence directly supports or reasonably entails the claim.

CONTRADICTED:
The gold evidence supports the opposite or is clearly incompatible with the claim.

UNSUPPORTED:
The claim is neither supported nor contradicted by the gold evidence.

Do not classify opinions, formatting, or purely conversational phrases as factual claims.

For supported or contradicted claims, provide the relevant gold evidence labels.
For unsupported claims, evidence_labels should normally be empty.

Be conservative. Do not mark a claim supported merely because it is medically plausible.

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

    parts = []


    for passage in evidence:

        parts.append(
            f"[{passage['label']}] "
            f"Section: {passage['section']}\n"
            f"{passage['text']}"
        )


    return "\n\n".join(
        parts
    )


# 6. LOAD DATA

print("=" * 75)

print(
    "SECTION 19 - CLAIM-LEVEL HALLUCINATION JUDGE PILOT"
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


# 7. GROUP RECORDS BY QUESTION

records_by_question = {}


for record in records:

    question_id = record[
        "question_id"
    ]


    records_by_question.setdefault(
        question_id,
        {}
    )


    records_by_question[
        question_id
    ][
        record["system"]
    ] = record


print(
    f"Unique questions  : "
    f"{len(records_by_question)}"
)


# 8. BUILD BALANCED QUESTION SAMPLE

questions_by_label = {

    "yes": [],
    "no": [],
    "maybe": []
}


for question_id, systems in (
    records_by_question.items()
):

    first_record = next(
        iter(
            systems.values()
        )
    )


    gold_label = (
        first_record[
            "gold_decision"
        ]
    )


    questions_by_label[
        gold_label
    ].append(
        question_id
    )


random_generator = random.Random(
    RANDOM_SEED
)


selected_question_ids = []


for label in [
    "yes",
    "no",
    "maybe"
]:

    available_ids = sorted(
        questions_by_label[
            label
        ]
    )


    selected = (
        random_generator.sample(
            available_ids,
            QUESTIONS_PER_LABEL
        )
    )


    selected_question_ids.extend(
        selected
    )


print(
    "\n[2] BALANCED PILOT SAMPLE"
)


print(
    f"Yes questions   : "
    f"{QUESTIONS_PER_LABEL}"
)

print(
    f"No questions    : "
    f"{QUESTIONS_PER_LABEL}"
)

print(
    f"Maybe questions : "
    f"{QUESTIONS_PER_LABEL}"
)


print(
    f"Total questions : "
    f"{len(selected_question_ids)}"
)

print(
    f"Judge requests  : "
    f"{len(selected_question_ids) * len(SYSTEM_ORDER)}"
)


# 9. LOAD CHECKPOINT

if OUTPUT_PATH.exists():

    existing = load_json(
        OUTPUT_PATH
    )

    judgments = existing.get(
        "judgments",
        []
    )


    print(
        "\n[3] CHECKPOINT FOUND"
    )

    print(
        f"Existing judgments: "
        f"{len(judgments)}"
    )


else:

    judgments = []


    print(
        "\n[3] NO EXISTING CHECKPOINT"
    )


completed_keys = {

    (
        item["question_id"],
        item["system"]
    )

    for item in judgments
}

# 10. OPENAI CLIENT

client = OpenAI()


# 11. RUN JUDGE

print(
    "\n[4] RUNNING CLAIM-LEVEL JUDGE"
)


total_input_tokens = sum(

    item.get(
        "input_tokens",
        0
    )

    for item in judgments
)


total_output_tokens = sum(

    item.get(
        "output_tokens",
        0
    )

    for item in judgments
)


for question_number, question_id in enumerate(
    selected_question_ids,
    start=1
):

    print(
        f"\nQuestion "
        f"{question_number}/"
        f"{len(selected_question_ids)} "
        f"- {question_id}"
    )


    for system_name in SYSTEM_ORDER:

        request_key = (
            question_id,
            system_name
        )


        if request_key in completed_keys:

            print(
                f"  {system_name:<22}"
                f" -> already completed"
            )

            continue


        record = (

            records_by_question[
                question_id
            ][
                system_name
            ]
        )


        evidence_text = (
            format_gold_evidence(
                record[
                    "gold_evidence"
                ]
            )
        )


        user_prompt = (
            f"QUESTION:\n"
            f"{record['question']}\n\n"

            f"GENERATED ANSWER:\n"
            f"{record['generated_answer']}\n\n"

            f"GOLD EVIDENCE:\n"
            f"{evidence_text}"
        )


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
                        "hallucination_judgment",

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


        claims = parsed[
            "claims"
        ]


        # 12. VALIDATE EVIDENCE LABELS

        valid_evidence_labels = {

            passage[
                "label"
            ]

            for passage in record[
                "gold_evidence"
            ]
        }


        invalid_evidence_labels = []


        for claim in claims:

            for label in claim[
                "evidence_labels"
            ]:

                if (
                    label
                    not in
                    valid_evidence_labels
                ):

                    invalid_evidence_labels.append(
                        label
                    )


        # 13. CLAIM COUNTS

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


            contradiction_rate = (

                contradicted
                /
                total_claims
            )


        else:

            hallucination_rate = None

            groundedness = None

            contradiction_rate = None


        # 14. TOKEN USAGE

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


        total_input_tokens += (
            input_tokens
        )

        total_output_tokens += (
            output_tokens
        )


        # 15. STORE

        judgment = {

            "question_id":
                question_id,

            "system":
                system_name,

            "gold_decision":
                record[
                    "gold_decision"
                ],

            "generated_answer":
                record[
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

            "contradiction_rate":
                contradiction_rate,

            "invalid_evidence_labels":
                invalid_evidence_labels,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens
        }


        judgments.append(
            judgment
        )


        completed_keys.add(
            request_key
        )


        print(
            f"  {system_name:<22}"
            f" -> "
            f"claims={total_claims}, "
            f"hallucination="
            f"{hallucination_rate}"
        )


        # Checkpoint after every call

        save_json(
            {

                "judge_model":
                    JUDGE_MODEL,

                "reasoning_effort":
                    REASONING_EFFORT,

                "random_seed":
                    RANDOM_SEED,

                "questions_per_label":
                    QUESTIONS_PER_LABEL,

                "selected_question_ids":
                    selected_question_ids,

                "judgments":
                    judgments
            },
            OUTPUT_PATH
        )


# 16. PILOT VALIDATION

print(
    "\n[5] PILOT VALIDATION"
)


invalid_label_cases = [

    judgment

    for judgment in judgments

    if judgment[
        "invalid_evidence_labels"
    ]
]


zero_claim_cases = [

    judgment

    for judgment in judgments

    if (
        judgment[
            "claim_counts"
        ]["total"]
        ==
        0
    )
]


print(
    f"Completed judgments      : "
    f"{len(judgments)}"
)

print(
    f"Invalid evidence labels  : "
    f"{len(invalid_label_cases)}"
)

print(
    f"Zero-claim answers       : "
    f"{len(zero_claim_cases)}"
)


# 17. SYSTEM SUMMARIES

print(
    "\n[6] PILOT HALLUCINATION SUMMARY"
)


system_summaries = {}


for system_name in SYSTEM_ORDER:

    system_judgments = [

        judgment

        for judgment in judgments

        if (
            judgment[
                "system"
            ]
            ==
            system_name
        )
    ]


    valid_judgments = [

        judgment

        for judgment
        in system_judgments

        if (
            judgment[
                "hallucination_rate"
            ]
            is not None
        )
    ]


    total_claims = sum(

        judgment[
            "claim_counts"
        ]["total"]

        for judgment
        in valid_judgments
    )


    total_supported = sum(

        judgment[
            "claim_counts"
        ]["supported"]

        for judgment
        in valid_judgments
    )


    total_unsupported = sum(

        judgment[
            "claim_counts"
        ]["unsupported"]

        for judgment
        in valid_judgments
    )


    total_contradicted = sum(

        judgment[
            "claim_counts"
        ]["contradicted"]

        for judgment
        in valid_judgments
    )


    # Mean answer-level hallucination rate

    mean_hallucination_rate = (

        sum(
            judgment[
                "hallucination_rate"
            ]

            for judgment
            in valid_judgments
        )

        /
        len(
            valid_judgments
        )
    )


    mean_groundedness = (

        sum(
            judgment[
                "groundedness"
            ]

            for judgment
            in valid_judgments
        )

        /
        len(
            valid_judgments
        )
    )


    # Micro claim-level hallucination rate

    micro_hallucination_rate = (

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
                valid_judgments
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
            mean_hallucination_rate,

        "mean_answer_groundedness":
            mean_groundedness,

        "micro_claim_hallucination_rate":
            micro_hallucination_rate,

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
        f"  Answers evaluated      : "
        f"{summary['answers']}"
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


# 18. TOKEN USAGE

print(
    "\n[7] JUDGE TOKEN USAGE"
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


# 19. SAVE FINAL PILOT

save_json(
    {

        "judge_model":
            JUDGE_MODEL,

        "reasoning_effort":
            REASONING_EFFORT,

        "random_seed":
            RANDOM_SEED,

        "questions_per_label":
            QUESTIONS_PER_LABEL,

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
    "\n[8] RESULTS SAVED"
)

print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 19 COMPLETE"
)

print(
    "=" * 75
)
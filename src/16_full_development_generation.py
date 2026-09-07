import json
import time
from pathlib import Path

from openai import OpenAI


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "generation"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILES = {

    "baseline_llm":
        GENERATION_DIR
        / "baseline_dev_inputs.json",

    "basic_rag":
        GENERATION_DIR
        / "basic_rag_dev_inputs.json",

    "advanced_rag":
        GENERATION_DIR
        / "advanced_rag_dev_inputs.json",

    "gold_context_control":
        GENERATION_DIR
        / "gold_context_dev_inputs.json"
}


OUTPUT_PATH = (
    RESULTS_DIR
    / "development_generations.json"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "development_generation_summary.json"
)


# 2. FROZEN MODEL SETTINGS

MODEL_NAME = (
    "gpt-5.4-mini-2026-03-17"
)

REASONING_EFFORT = "none"

MAX_OUTPUT_TOKENS = 300


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


# 3. STRUCTURED OUTPUT SCHEMA

ANSWER_SCHEMA = {

    "type": "object",

    "properties": {

        "decision": {

            "type": "string",

            "enum": [
                "yes",
                "no",
                "maybe"
            ]
        },

        "answer": {
            "type": "string"
        },

        "citations": {

            "type": "array",

            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "decision",
        "answer",
        "citations"
    ],

    "additionalProperties": False
}


# 4. HELPERS

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


# 5. LOAD INPUT FILES

print("=" * 75)

print(
    "SECTION 16 - FULL DEVELOPMENT GENERATION"
)

print("=" * 75)


datasets = {

    system_name:
        load_json(path)

    for system_name, path
    in INPUT_FILES.items()
}


print(
    "\n[1] INPUT FILES LOADED"
)


for system_name in SYSTEM_ORDER:

    print(
        f"{system_name:<22}: "
        f"{len(datasets[system_name])}"
    )


# 6. INDEX EACH SYSTEM BY QUESTION ID

indexed_datasets = {}


for system_name, records in datasets.items():

    indexed_datasets[
        system_name
    ] = {

        record["question_id"]:
            record

        for record in records
    }


question_sets = [

    set(
        records.keys()
    )

    for records
    in indexed_datasets.values()
]


if not all(

    question_set
    ==
    question_sets[0]

    for question_set
    in question_sets

):

    raise ValueError(
        "Generation conditions contain "
        "different question IDs."
    )


question_ids = sorted(
    question_sets[0]
)


print(
    f"\nCommon development questions: "
    f"{len(question_ids)}"
)


if len(question_ids) != 500:

    raise ValueError(
        "Expected exactly 500 "
        "development questions."
    )


# 7. LOAD EXISTING CHECKPOINT

if OUTPUT_PATH.exists():

    existing_output = load_json(
        OUTPUT_PATH
    )

    results = existing_output.get(
        "results",
        []
    )

    print(
        "\n[2] CHECKPOINT FOUND"
    )

    print(
        f"Existing completed responses: "
        f"{len(results)}"
    )

else:

    results = []

    print(
        "\n[2] NO EXISTING CHECKPOINT"
    )

    print(
        "Starting a new development run."
    )


# 8. IDENTIFY COMPLETED REQUESTS

# A unique request is: (question_id, system)

completed_keys = {

    (
        result["question_id"],
        result["system"]
    )

    for result
    in results
}


TOTAL_EXPECTED = (

    len(question_ids)
    *
    len(SYSTEM_ORDER)
)


print(
    f"Total expected responses: "
    f"{TOTAL_EXPECTED}"
)


print(
    f"Already completed       : "
    f"{len(completed_keys)}"
)


print(
    f"Remaining               : "
    f"{TOTAL_EXPECTED - len(completed_keys)}"
)

# 9. OPENAI CLIENT

client = OpenAI()

# 10. RUN FULL DEVELOPMENT GENERATION

print(
    "\n[3] RUNNING DEVELOPMENT GENERATION"
)


new_requests_this_run = 0


for question_number, question_id in enumerate(
    question_ids,
    start=1
):

    print(
        f"\nQuestion "
        f"{question_number}/"
        f"{len(question_ids)} "
        f"- {question_id}"
    )


    for system_name in SYSTEM_ORDER:

        request_key = (
            question_id,
            system_name
        )


        # Resume support

        if request_key in completed_keys:

            print(
                f"  {system_name:<22}"
                f" -> already completed"
            )

            continue


        item = (
            indexed_datasets[
                system_name
            ][
                question_id
            ]
        )


        # API request

        try:

            response = client.responses.create(

                model=MODEL_NAME,

                instructions=item[
                    "system_prompt"
                ],

                input=item[
                    "user_prompt"
                ],

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
                            "biomedical_qa_answer",

                        "strict":
                            True,

                        "schema":
                            ANSWER_SCHEMA
                    }
                }
            )


            parsed = json.loads(
                response.output_text
            )

            # Token usage

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

            # Citation validation

            available_labels = {

                context[
                    "context_label"
                ]

                for context
                in item["contexts"]
            }


            returned_citations = (
                parsed[
                    "citations"
                ]
            )


            invalid_citations = [

                citation

                for citation
                in returned_citations

                if citation
                not in available_labels
            ]


            # Baseline must have no citations

            baseline_citation_violation = (

                system_name
                ==
                "baseline_llm"

                and

                len(
                    returned_citations
                ) > 0
            )


            # Store response

            result_record = {

                "question_id":
                    question_id,

                "system":
                    system_name,

                "question":
                    item[
                        "question"
                    ],

                "decision":
                    parsed[
                        "decision"
                    ],

                "answer":
                    parsed[
                        "answer"
                    ],

                "citations":
                    returned_citations,

                "invalid_citations":
                    invalid_citations,

                "baseline_citation_violation":
                    baseline_citation_violation,

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens,

                "model":
                    MODEL_NAME
            }


            results.append(
                result_record
            )


            completed_keys.add(
                request_key
            )


            new_requests_this_run += 1


            print(
                f"  {system_name:<22}"
                f" -> "
                f"{parsed['decision']}"
            )

            # SAVE AFTER EVERY SUCCESSFUL REQUEST

            checkpoint = {

                "model":
                    MODEL_NAME,

                "reasoning_effort":
                    REASONING_EFFORT,

                "max_output_tokens":
                    MAX_OUTPUT_TOKENS,

                "expected_responses":
                    TOTAL_EXPECTED,

                "completed_responses":
                    len(results),

                "results":
                    results
            }


            save_json(
                checkpoint,
                OUTPUT_PATH
            )


            # Small pause.
            # This does NOT affect model output.
            time.sleep(
                0.05
            )


        except Exception as error:


            print(
                "\n"
                + "=" * 75
            )

            print(
                "API ERROR"
            )

            print(
                "=" * 75
            )


            print(
                f"Question ID: "
                f"{question_id}"
            )


            print(
                f"System: "
                f"{system_name}"
            )


            print(
                f"Error: "
                f"{error}"
            )


            print(
                "\nCompleted responses have "
                "already been saved."
            )

            print(
                "Fix the issue and rerun this "
                "same script to resume."
            )


            raise


# 11. FINAL VALIDATION

print(
    "\n[4] FINAL VALIDATION"
)


print(
    f"Completed responses: "
    f"{len(results)}"
)


if len(results) != TOTAL_EXPECTED:

    raise ValueError(
        "Full generation is incomplete."
    )

# Ensure each question has exactly four systems.

question_system_counts = {}


for result in results:

    question_id = result[
        "question_id"
    ]

    question_system_counts.setdefault(
        question_id,
        set()
    )

    question_system_counts[
        question_id
    ].add(
        result[
            "system"
        ]
    )


incomplete_questions = [

    question_id

    for question_id, systems
    in question_system_counts.items()

    if len(systems)
    != len(SYSTEM_ORDER)
]


print(
    f"Incomplete questions: "
    f"{len(incomplete_questions)}"
)


if incomplete_questions:

    raise ValueError(
        "Some questions do not contain "
        "all four system outputs."
    )


# 12. CITATION VALIDATION

invalid_citation_records = [

    result

    for result in results

    if result[
        "invalid_citations"
    ]
]


baseline_citation_records = [

    result

    for result in results

    if result[
        "baseline_citation_violation"
    ]
]


print(
    f"Invalid citation records : "
    f"{len(invalid_citation_records)}"
)


print(
    f"Baseline citation records: "
    f"{len(baseline_citation_records)}"
)


# 13. TOKEN USAGE

total_input_tokens = sum(

    result[
        "input_tokens"
    ]

    for result
    in results
)


total_output_tokens = sum(

    result[
        "output_tokens"
    ]

    for result
    in results
)


print(
    "\n[5] TOKEN USAGE"
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


# 14. DECISION DISTRIBUTIONS

print(
    "\n[6] DECISION DISTRIBUTIONS"
)


decision_distributions = {}


for system_name in SYSTEM_ORDER:

    counts = {

        "yes": 0,
        "no": 0,
        "maybe": 0
    }


    for result in results:

        if (
            result[
                "system"
            ]
            ==
            system_name
        ):

            counts[
                result[
                    "decision"
                ]
            ] += 1


    decision_distributions[
        system_name
    ] = counts


    print(
        f"\n{system_name}"
    )


    print(
        f"  yes   : "
        f"{counts['yes']}"
    )

    print(
        f"  no    : "
        f"{counts['no']}"
    )

    print(
        f"  maybe : "
        f"{counts['maybe']}"
    )


# 15. SAVE SUMMARY

summary = {

    "split":
        "development",

    "questions":
        len(question_ids),

    "systems":
        SYSTEM_ORDER,

    "expected_responses":
        TOTAL_EXPECTED,

    "completed_responses":
        len(results),

    "model":
        MODEL_NAME,

    "reasoning_effort":
        REASONING_EFFORT,

    "max_output_tokens":
        MAX_OUTPUT_TOKENS,

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
    },

    "invalid_citation_records":
        len(
            invalid_citation_records
        ),

    "baseline_citation_records":
        len(
            baseline_citation_records
        ),

    "decision_distributions":
        decision_distributions
}


save_json(
    summary,
    SUMMARY_PATH
)


print(
    "\n[7] RESULTS SAVED"
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
    "SECTION 16 COMPLETE"
)

print(
    "=" * 75
)
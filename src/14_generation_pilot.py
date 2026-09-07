import json
import random
from pathlib import Path

from openai import OpenAI


# 1. PATHS

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
    / "generation_pilot_25.json"
)


# 2. FROZEN MODEL SETTINGS

MODEL_NAME = (
    "gpt-5.4-mini-2026-03-17"
)

REASONING_EFFORT = "none"

MAX_OUTPUT_TOKENS = 300


# 3. PILOT SETTINGS

PILOT_SIZE = 25

RANDOM_SEED = 42


# Current API pricing checked for this experiment.
# Used only for estimating generation cost.

INPUT_PRICE_PER_MILLION = 0.75

OUTPUT_PRICE_PER_MILLION = 4.50


# 4. STRUCTURED OUTPUT SCHEMA

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


# 6. LOAD ALL FOUR CONDITIONS

print("=" * 75)
print("SECTION 14 - GENERATION PILOT")
print("=" * 75)


datasets = {

    system_name:
        load_json(path)

    for system_name, path
    in INPUT_FILES.items()
}


print("\n[1] INPUT FILES LOADED")


for system_name, records in datasets.items():

    print(
        f"{system_name:<22}: "
        f"{len(records)}"
    )


# 7. INDEX BY QUESTION ID

indexed_datasets = {}


for system_name, records in datasets.items():

    indexed_datasets[
        system_name
    ] = {

        record["question_id"]:
            record

        for record
        in records
    }


# All systems must contain identical questions.

question_sets = [

    set(records.keys())

    for records
    in indexed_datasets.values()
]


if not all(
    question_set == question_sets[0]
    for question_set
    in question_sets
):

    raise ValueError(
        "Generation conditions contain "
        "different question IDs."
    )


all_question_ids = sorted(
    question_sets[0]
)


print(
    f"\nCommon question IDs: "
    f"{len(all_question_ids)}"
)


# 8. SELECT REPRODUCIBLE PILOT QUESTIONS

random_generator = random.Random(
    RANDOM_SEED
)


pilot_question_ids = (
    random_generator.sample(
        all_question_ids,
        PILOT_SIZE
    )
)


print(
    f"\n[2] PILOT SAMPLE"
)

print(
    f"Questions selected: "
    f"{len(pilot_question_ids)}"
)

print(
    f"Random seed: "
    f"{RANDOM_SEED}"
)


# 9. OPENAI CLIENT

client = OpenAI()


# 10. STORAGE

results = []

total_input_tokens = 0

total_output_tokens = 0

invalid_citation_cases = []

baseline_citation_cases = []


# 11. RUN PILOT

print(
    "\n[3] RUNNING GENERATION PILOT"
)


total_requests = (
    PILOT_SIZE
    *
    len(indexed_datasets)
)


completed_requests = 0


for question_number, question_id in enumerate(
    pilot_question_ids,
    start=1
):

    print(
        f"\nQuestion "
        f"{question_number}/"
        f"{PILOT_SIZE}"
        f" - {question_id}"
    )


    for system_name, dataset in (
        indexed_datasets.items()
    ):

        item = dataset[
            question_id
        ]


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


        # Token counts

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


        if invalid_citations:

            invalid_citation_cases.append(
                {
                    "question_id":
                        question_id,

                    "system":
                        system_name,

                    "citations":
                        invalid_citations
                }
            )


        # Baseline should never cite anything.

        if (
            system_name
            ==
            "baseline_llm"

            and

            returned_citations
        ):

            baseline_citation_cases.append(
                {
                    "question_id":
                        question_id,

                    "citations":
                        returned_citations
                }
            )


        # Store output

        results.append(
            {
                "question_id":
                    question_id,

                "system":
                    system_name,

                "question":
                    item["question"],

                "decision":
                    parsed["decision"],

                "answer":
                    parsed["answer"],

                "citations":
                    parsed[
                        "citations"
                    ],

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens,

                "model":
                    MODEL_NAME
            }
        )


        completed_requests += 1


        print(
            f"  {system_name:<22}"
            f" -> "
            f"{parsed['decision']}"
        )


        # Checkpoint after every successful request.

        save_json(
            {
                "pilot_size":
                    PILOT_SIZE,

                "random_seed":
                    RANDOM_SEED,

                "model":
                    MODEL_NAME,

                "results":
                    results
            },
            OUTPUT_PATH
        )


# 12. COST ESTIMATE

input_cost = (

    total_input_tokens
    /
    1_000_000

    *
    INPUT_PRICE_PER_MILLION
)


output_cost = (

    total_output_tokens
    /
    1_000_000

    *
    OUTPUT_PRICE_PER_MILLION
)


total_cost = (
    input_cost
    +
    output_cost
)


# 13. SUMMARY

print(
    "\n[4] PILOT SUMMARY"
)


print(
    f"Completed requests : "
    f"{completed_requests}/"
    f"{total_requests}"
)


print(
    f"Input tokens       : "
    f"{total_input_tokens}"
)

print(
    f"Output tokens      : "
    f"{total_output_tokens}"
)

print(
    f"Total tokens       : "
    f"{total_input_tokens + total_output_tokens}"
)


print(
    f"\nEstimated API cost : "
    f"${total_cost:.4f}"
)


# 14. VALIDATION RESULTS

print(
    "\n[5] OUTPUT VALIDATION"
)


print(
    f"Invalid citation cases : "
    f"{len(invalid_citation_cases)}"
)


print(
    f"Baseline citation cases: "
    f"{len(baseline_citation_cases)}"
)


if (
    not invalid_citation_cases
    and
    not baseline_citation_cases
):

    print(
        "PASS: Citation format is valid."
    )


# 15. DECISION DISTRIBUTIONS

print(
    "\n[6] PILOT DECISION DISTRIBUTIONS"
)


for system_name in indexed_datasets:

    system_results = [

        result

        for result in results

        if result[
            "system"
        ] == system_name
    ]


    counts = {

        "yes": 0,
        "no": 0,
        "maybe": 0
    }


    for result in system_results:

        counts[
            result["decision"]
        ] += 1


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


# 16. SAVE FINAL PILOT SUMMARY

final_output = {

    "pilot_size":
        PILOT_SIZE,

    "random_seed":
        RANDOM_SEED,

    "model":
        MODEL_NAME,

    "requests":
        completed_requests,

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

    "estimated_cost_usd":
        total_cost,

    "invalid_citation_cases":
        invalid_citation_cases,

    "baseline_citation_cases":
        baseline_citation_cases,

    "results":
        results
}


save_json(
    final_output,
    OUTPUT_PATH
)


print(
    "\n[7] RESULTS SAVED"
)

print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 14 COMPLETE"
)

print(
    "=" * 75
)
import json
from pathlib import Path

from openai import OpenAI


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "generation"
)


BASELINE_PATH = (
    GENERATION_DIR
    / "baseline_dev_inputs.json"
)

BASIC_PATH = (
    GENERATION_DIR
    / "basic_rag_dev_inputs.json"
)

ADVANCED_PATH = (
    GENERATION_DIR
    / "advanced_rag_dev_inputs.json"
)

GOLD_PATH = (
    GENERATION_DIR
    / "gold_context_dev_inputs.json"
)


# 2. FROZEN MODEL SETTINGS

MODEL_NAME = (
    "gpt-5.4-mini-2026-03-17"
)

REASONING_EFFORT = "none"

MAX_OUTPUT_TOKENS = 300


TARGET_QUESTION_ID = "17610439"


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


def find_question(
    records,
    question_id
):

    for record in records:

        if (
            record["question_id"]
            ==
            question_id
        ):

            return record

    raise ValueError(
        f"Question not found: "
        f"{question_id}"
    )


# 4. STRUCTURED OUTPUT SCHEMA

# The model must return exactly:
# {
#   "decision": "yes/no/maybe",
#   "answer": "...",
#   "citations": [...]
# }

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


# 5. LOAD FOUR CONDITIONS

print("=" * 75)
print("SECTION 13 - LLM GENERATION SMOKE TEST")
print("=" * 75)


baseline_inputs = load_json(
    BASELINE_PATH
)

basic_inputs = load_json(
    BASIC_PATH
)

advanced_inputs = load_json(
    ADVANCED_PATH
)

gold_inputs = load_json(
    GOLD_PATH
)


conditions = {

    "BASELINE":
        find_question(
            baseline_inputs,
            TARGET_QUESTION_ID
        ),

    "BASIC RAG":
        find_question(
            basic_inputs,
            TARGET_QUESTION_ID
        ),

    "ADVANCED RAG":
        find_question(
            advanced_inputs,
            TARGET_QUESTION_ID
        ),

    "GOLD CONTROL":
        find_question(
            gold_inputs,
            TARGET_QUESTION_ID
        )
}


print(
    f"\nModel: "
    f"{MODEL_NAME}"
)

print(
    f"Reasoning effort: "
    f"{REASONING_EFFORT}"
)

print(
    f"Question ID: "
    f"{TARGET_QUESTION_ID}"
)


# 6. CREATE OPENAI CLIENT

client = OpenAI()



# 7. RUN ONE QUESTION THROUGH ALL FOUR SYSTEMS


for condition_name, item in conditions.items():

    print(
        "\n"
        + "=" * 75
    )

    print(
        condition_name
    )

    print(
        "=" * 75
    )


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


    raw_output = (
        response.output_text
    )


    parsed_output = json.loads(
        raw_output
    )


    print(
        "\nDecision:"
    )

    print(
        parsed_output[
            "decision"
        ]
    )


    print(
        "\nAnswer:"
    )

    print(
        parsed_output[
            "answer"
        ]
    )


    print(
        "\nCitations:"
    )

    print(
        parsed_output[
            "citations"
        ]
    )

   
    # Token usage

    if response.usage:

        print(
            "\nToken usage:"
        )

        print(
            f"  Input  : "
            f"{response.usage.input_tokens}"
        )

        print(
            f"  Output : "
            f"{response.usage.output_tokens}"
        )

        print(
            f"  Total  : "
            f"{response.usage.total_tokens}"
        )


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 13 COMPLETE"
)

print(
    "=" * 75
)
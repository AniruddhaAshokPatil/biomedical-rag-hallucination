import ast
import hashlib
import json
import time

from collections import Counter
from pathlib import Path

from openai import OpenAI


# 1. PROJECT PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)


FROZEN_GENERATOR_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "16_full_development_generation.py"
)


GENERATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
)


BASELINE_INPUT_PATH = (
    GENERATION_DIR
    / "baseline_test_inputs.json"
)


BASIC_INPUT_PATH = (
    GENERATION_DIR
    / "basic_rag_test_inputs.json"
)


ADVANCED_INPUT_PATH = (
    GENERATION_DIR
    / "advanced_rag_test_inputs.json"
)


GOLD_INPUT_PATH = (
    GENERATION_DIR
    / "gold_context_test_inputs.json"
)


OUTPUT_PATH = (
    GENERATION_DIR
    / "final_test_generation_results.json"
)


SUMMARY_PATH = (
    GENERATION_DIR
    / "final_test_generation_summary.json"
)


# 2. HELPERS

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


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


def sha256_file(path):

    hasher = hashlib.sha256()


    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )


            if not chunk:

                break


            hasher.update(
                chunk
            )


    return hasher.hexdigest()


def sha256_text(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# 3. START

print("=" * 78)

print(
    "SECTION 36 - FINAL TEST GENERATION"
)

print("=" * 78)


# 4. VERIFY ORIGINAL FROZEN PROTOCOL

print(
    "\n[1] VERIFYING ORIGINAL FROZEN PROTOCOL"
)


manifest = load_json(
    MANIFEST_PATH
)


changed_files = []

missing_files = []


for relative_path, expected_hash in (
    manifest[
        "file_hashes"
    ].items()
):

    path = (
        PROJECT_ROOT
        / relative_path
    )


    if not path.exists():

        missing_files.append(
            relative_path
        )

        continue


    current_hash = sha256_file(
        path
    )


    if current_hash != expected_hash:

        changed_files.append(
            relative_path
        )


if missing_files:

    print(
        "\nMissing frozen files:"
    )


    for filename in missing_files:

        print(
            f"  {filename}"
        )


if changed_files:

    print(
        "\nChanged frozen files:"
    )


    for filename in changed_files:

        print(
            f"  {filename}"
        )


if (
    missing_files
    or
    changed_files
):

    raise RuntimeError(
        "Original frozen protocol "
        "integrity check failed."
    )


print(
    "PASS: All original frozen files "
    "remain unchanged."
)


# 5. EXTRACT EXACT FROZEN GENERATOR SETTINGS
# Rather than retyping these values, load them directly from the frozen full-development generator:
#   MODEL_NAME
#   REASONING_EFFORT
#   MAX_OUTPUT_TOKENS
#   SYSTEM_ORDER
#   ANSWER_SCHEMA


print(
    "\n[2] LOADING FROZEN GENERATOR CONFIGURATION"
)


source = FROZEN_GENERATOR_SOURCE.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source,
    filename=str(
        FROZEN_GENERATOR_SOURCE
    )
)


required_assignments = {

    "MODEL_NAME",
    "REASONING_EFFORT",
    "MAX_OUTPUT_TOKENS",
    "SYSTEM_ORDER",
    "ANSWER_SCHEMA"
}


selected_nodes = []


for node in tree.body:

    if not isinstance(
        node,
        ast.Assign
    ):

        continue


    for target in node.targets:

        if (
            isinstance(
                target,
                ast.Name
            )
            and
            target.id
            in
            required_assignments
        ):

            selected_nodes.append(
                node
            )

            break


module = ast.Module(
    body=selected_nodes,
    type_ignores=[]
)


ast.fix_missing_locations(
    module
)


frozen_namespace = {}


exec(
    compile(
        module,
        filename=str(
            FROZEN_GENERATOR_SOURCE
        ),
        mode="exec"
    ),
    frozen_namespace
)


missing_settings = (

    required_assignments

    -

    set(
        frozen_namespace.keys()
    )
)


if missing_settings:

    raise RuntimeError(
        "Could not extract frozen "
        "generator settings: "
        f"{sorted(missing_settings)}"
    )


MODEL_NAME = frozen_namespace[
    "MODEL_NAME"
]


REASONING_EFFORT = frozen_namespace[
    "REASONING_EFFORT"
]


MAX_OUTPUT_TOKENS = frozen_namespace[
    "MAX_OUTPUT_TOKENS"
]


SYSTEM_ORDER = frozen_namespace[
    "SYSTEM_ORDER"
]


ANSWER_SCHEMA = frozen_namespace[
    "ANSWER_SCHEMA"
]


print(
    f"Model             : "
    f"{MODEL_NAME}"
)


print(
    f"Reasoning effort  : "
    f"{REASONING_EFFORT}"
)


print(
    f"Max output tokens : "
    f"{MAX_OUTPUT_TOKENS}"
)


print(
    f"System order      : "
    f"{SYSTEM_ORDER}"
)


print(
    "Text verbosity    : low"
)


print(
    "Structured output : strict JSON schema"
)


# 6. SAFETY CHECK FROZEN MODEL

EXPECTED_MODEL = (
    "gpt-5.4-mini-2026-03-17"
)


EXPECTED_REASONING = (
    "none"
)


EXPECTED_MAX_OUTPUT = 300


if MODEL_NAME != EXPECTED_MODEL:

    raise RuntimeError(
        "Frozen generator model does "
        "not match the expected "
        "development configuration."
    )


if (
    REASONING_EFFORT
    !=
    EXPECTED_REASONING
):

    raise RuntimeError(
        "Frozen reasoning setting "
        "does not match development."
    )


if (
    MAX_OUTPUT_TOKENS
    !=
    EXPECTED_MAX_OUTPUT
):

    raise RuntimeError(
        "Frozen max-output setting "
        "does not match development."
    )


print(
    "PASS: Generator configuration "
    "matches the frozen protocol."
)


# 7. LOAD FOUR FINAL TEST INPUT FILES

print(
    "\n[3] LOADING FINAL TEST GENERATION INPUTS"
)


baseline_inputs = load_json(
    BASELINE_INPUT_PATH
)


basic_inputs = load_json(
    BASIC_INPUT_PATH
)


advanced_inputs = load_json(
    ADVANCED_INPUT_PATH
)


gold_inputs = load_json(
    GOLD_INPUT_PATH
)


input_datasets = {

    "baseline_llm":
        baseline_inputs,

    "basic_rag":
        basic_inputs,

    "advanced_rag":
        advanced_inputs,

    "gold_context_control":
        gold_inputs
}


for system_name in SYSTEM_ORDER:

    if system_name not in input_datasets:

        raise ValueError(
            f"Frozen system "
            f"{system_name} "
            f"is missing from "
            f"test inputs."
        )


for system_name, dataset in (
    input_datasets.items()
):

    print(
        f"{system_name:<22}: "
        f"{len(dataset)}"
    )


    if len(dataset) != 500:

        raise ValueError(
            f"{system_name} does not "
            f"contain exactly "
            f"500 questions."
        )


TOTAL_EXPECTED = (
    500
    *
    len(
        SYSTEM_ORDER
    )
)


if TOTAL_EXPECTED != 2000:

    raise ValueError(
        "Expected exactly 2,000 "
        "final generation requests."
    )


print(
    f"Total expected responses : "
    f"{TOTAL_EXPECTED}"
)

# 8. HASH FINAL GENERATION INPUT FILES
# Section 35 outputs were created after the original protocol manifest, so bind the final generation run to these exact four input files here.


input_file_hashes = {

    "baseline_test_inputs.json":
        sha256_file(
            BASELINE_INPUT_PATH
        ),

    "basic_rag_test_inputs.json":
        sha256_file(
            BASIC_INPUT_PATH
        ),

    "advanced_rag_test_inputs.json":
        sha256_file(
            ADVANCED_INPUT_PATH
        ),

    "gold_context_test_inputs.json":
        sha256_file(
            GOLD_INPUT_PATH
        )
}


print(
    "\nInput-file hashes captured."
)


# 9. VALIDATE QUESTION ALIGNMENT

print(
    "\n[4] VALIDATING FINAL INPUT ALIGNMENT"
)


reference_ids = [

    str(
        item[
            "question_id"
        ]
    )

    for item
    in baseline_inputs
]


if len(
    set(
        reference_ids
    )
) != 500:

    raise ValueError(
        "Baseline contains duplicate "
        "question IDs."
    )


for system_name, dataset in (
    input_datasets.items()
):

    system_ids = [

        str(
            item[
                "question_id"
            ]
        )

        for item
        in dataset
    ]


    if system_ids != reference_ids:

        raise ValueError(
            f"{system_name} question "
            f"ordering does not match "
            f"baseline ordering."
        )


print(
    "PASS: All systems contain the "
    "same 500 questions in the "
    "same order."
)


# 10. VALIDATE PROMPT CONSISTENCY

system_prompt_hashes = set()


for dataset in input_datasets.values():

    for item in dataset:

        system_prompt_hashes.add(
            sha256_text(
                item[
                    "system_prompt"
                ]
            )
        )


if len(
    system_prompt_hashes
) != 1:

    raise ValueError(
        "Generation inputs do not "
        "share one frozen "
        "system prompt."
    )


frozen_prompt_hash = next(
    iter(
        system_prompt_hashes
    )
)


print(
    f"Frozen prompt SHA256 : "
    f"{frozen_prompt_hash}"
)


print(
    "PASS: All 2,000 inputs use "
    "the same frozen prompt."
)


# 11. BUILD LOOKUP BY SYSTEM + QUESTION

input_lookup = {}


for system_name, dataset in (
    input_datasets.items()
):

    input_lookup[
        system_name
    ] = {

        str(
            item[
                "question_id"
            ]
        ):
            item

        for item
        in dataset
    }


# 12. LOAD EXISTING CHECKPOINT IF PRESENT

if OUTPUT_PATH.exists():

    existing = load_json(
        OUTPUT_PATH
    )


    print(
        "\n[5] CHECKPOINT FOUND"
    )

  
    # Verify the checkpoint belongs to this exact run.
    

    if (
        existing.get(
            "model"
        )
        !=
        MODEL_NAME
    ):

        raise RuntimeError(
            "Existing checkpoint uses "
            "a different generator model."
        )


    if (
        existing.get(
            "reasoning_effort"
        )
        !=
        REASONING_EFFORT
    ):

        raise RuntimeError(
            "Existing checkpoint uses "
            "a different reasoning setting."
        )


    if (
        existing.get(
            "max_output_tokens"
        )
        !=
        MAX_OUTPUT_TOKENS
    ):

        raise RuntimeError(
            "Existing checkpoint uses "
            "a different max-output "
            "setting."
        )


    saved_hashes = (
        existing.get(
            "input_file_hashes"
        )
    )


    if (
        saved_hashes
        !=
        input_file_hashes
    ):

        raise RuntimeError(
            "Generation input files "
            "changed since the "
            "checkpoint was created."
        )


    results = existing.get(
        "results",
        []
    )


    print(
        f"Existing completed responses: "
        f"{len(results)}"
    )


else:

    print(
        "\n[5] NO EXISTING CHECKPOINT"
    )


    print(
        "Starting a new final-test "
        "generation run."
    )


    results = []


# 13. VALIDATE EXISTING CHECKPOINT RESULTS

completed_keys = set()


for result in results:

    key = (
        str(
            result[
                "question_id"
            ]
        ),
        result[
            "system"
        ]
    )


    if key in completed_keys:

        raise RuntimeError(
            "Duplicate completed "
            f"checkpoint key: {key}"
        )


    completed_keys.add(
        key
    )


print(
    f"Already completed : "
    f"{len(completed_keys)}"
)


print(
    f"Remaining         : "
    f"{TOTAL_EXPECTED - len(completed_keys)}"
)


# 14. CREATE OPENAI CLIENT

print(
    "\n[6] CREATING OPENAI CLIENT"
)


client = OpenAI()


print(
    "OpenAI client created."
)


# 15. CITATION VALIDATION

def validate_citations(
    item,
    parsed
):

    citations = parsed[
        "citations"
    ]


    allowed_citations = {

        context[
            "context_label"
        ]

        for context
        in item[
            "contexts"
        ]
    }


    invalid_citations = [

        citation

        for citation
        in citations

        if citation
        not in
        allowed_citations
    ]


    baseline_violation = (

        item[
            "system"
        ]
        ==
        "baseline_llm"

        and

        len(
            citations
        )
        >
        0
    )


    return (
        invalid_citations,
        baseline_violation
    )


# 16. CHECKPOINT FUNCTION

def save_checkpoint():

    checkpoint = {

        "split":
            "test",

        "model":
            MODEL_NAME,

        "reasoning_effort":
            REASONING_EFFORT,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "verbosity":
            "low",

        "structured_output":
            "strict_json_schema",

        "system_order":
            SYSTEM_ORDER,

        "expected_responses":
            TOTAL_EXPECTED,

        "completed_responses":
            len(
                results
            ),

        "input_file_hashes":
            input_file_hashes,

        "frozen_prompt_sha256":
            frozen_prompt_hash,

        "results":
            results
    }


    save_json(
        checkpoint,
        OUTPUT_PATH
    )


# 17. RUN FINAL TEST GENERATION

# Ordering:
# question 1:
#   baseline
#   basic
#   advanced
#   gold

# question 2:
#   baseline
#   basic
#   advanced
#   gold

# This gives every question all four paired responses.


print(
    "\n[7] RUNNING FINAL TEST GENERATION"
)


request_number = len(
    completed_keys
)


for question_number, question_id in enumerate(
    reference_ids,
    start=1
):

    for system_name in SYSTEM_ORDER:

        key = (
            question_id,
            system_name
        )


        # Resume safely: never repeat an already-completed call.

        if key in completed_keys:

            continue


        item = (
            input_lookup[
                system_name
            ][
                question_id
            ]
        )


        request_number += 1


        print(
            f"\n"
            f"[{request_number}/{TOTAL_EXPECTED}] "
            f"Question "
            f"{question_number}/500 | "
            f"{system_name}"
        )


        try:

            response = client.responses.create(

                model=
                    MODEL_NAME,

                instructions=
                    item[
                        "system_prompt"
                    ],

                input=
                    item[
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


            # Parse strict structured output.

            raw_output = (
                response.output_text
            )


            if not raw_output:

                raise RuntimeError(
                    "OpenAI response contained "
                    "no output_text."
                )


            parsed = json.loads(
                raw_output
            )

            # Validate required fields.

            required_output_fields = {

                "decision",
                "answer",
                "citations"
            }


            missing_output_fields = (

                required_output_fields

                -

                set(
                    parsed.keys()
                )
            )


            if missing_output_fields:

                raise ValueError(
                    "Structured output missing "
                    f"fields: "
                    f"{missing_output_fields}"
                )


            if parsed[
                "decision"
            ] not in {

                "yes",
                "no",
                "maybe"

            }:

                raise ValueError(
                    "Invalid decision value: "
                    f"{parsed['decision']}"
                )


            if not isinstance(
                parsed[
                    "answer"
                ],
                str
            ):

                raise TypeError(
                    "Answer is not a string."
                )


            if not isinstance(
                parsed[
                    "citations"
                ],
                list
            ):

                raise TypeError(
                    "Citations is not a list."
                )


            # Citation validation.

            (
                invalid_citations,
                baseline_citation_violation
            ) = validate_citations(
                item,
                parsed
            )


            # Token usage.

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

            # Store result.
            # No test label/reference answer is present.

            result_record = {

                "question_id":
                    question_id,

                "system":
                    system_name,

                "decision":
                    parsed[
                        "decision"
                    ],

                "answer":
                    parsed[
                        "answer"
                    ],

                "citations":
                    parsed[
                        "citations"
                    ],

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
                key
            )


            # SAVE AFTER EVERY SUCCESSFUL REQUEST

            save_checkpoint()


            print(
                f"  Decision : "
                f"{parsed['decision']}"
            )


            print(
                f"  Citations: "
                f"{parsed['citations']}"
            )


            print(
                f"  Tokens   : "
                f"{input_tokens} in / "
                f"{output_tokens} out"
            )


            # Same small pause used in development.
            time.sleep(
                0.05
            )


        except Exception as error:

            print(
                "\n"
                + "=" * 78
            )


            print(
                "GENERATION STOPPED"
            )


            print(
                "=" * 78
            )


            print(
                f"Question ID : "
                f"{question_id}"
            )


            print(
                f"System      : "
                f"{system_name}"
            )


            print(
                f"Error type  : "
                f"{type(error).__name__}"
            )


            print(
                f"Error       : "
                f"{error}"
            )


            print(
                "\nCompleted responses have "
                "already been saved."
            )


            print(
                "Fix the issue and rerun "
                "this script. It will resume "
                "without repeating completed "
                "requests."
            )


            print(
                "\nIMPORTANT:"
            )


            print(
                "If the frozen model itself "
                "is unavailable, DO NOT "
                "substitute another model."
            )


            raise


# 18. FINAL COMPLETENESS CHECK

print(
    "\n[8] FINAL COMPLETENESS CHECK"
)


print(
    f"Completed responses : "
    f"{len(results)}"
)


print(
    f"Expected responses  : "
    f"{TOTAL_EXPECTED}"
)


if (
    len(
        results
    )
    !=
    TOTAL_EXPECTED
):

    raise RuntimeError(
        "Final test generation is "
        "incomplete."
    )


final_keys = {

    (
        str(
            result[
                "question_id"
            ]
        ),
        result[
            "system"
        ]
    )

    for result
    in results
}


if len(
    final_keys
) != TOTAL_EXPECTED:

    raise RuntimeError(
        "Final generation contains "
        "duplicate system-question "
        "records."
    )


for system_name in SYSTEM_ORDER:

    system_count = sum(

        1

        for result
        in results

        if result[
            "system"
        ]
        ==
        system_name
    )


    print(
        f"{system_name:<22}: "
        f"{system_count}"
    )


    if system_count != 500:

        raise RuntimeError(
            f"{system_name} does not "
            f"contain 500 final answers."
        )


print(
    "PASS: Exactly 500 answers per "
    "system and 2,000 total answers."
)


# 19. CITATION VALIDITY

print(
    "\n[9] CITATION VALIDATION"
)


invalid_citation_records = [

    result

    for result
    in results

    if result[
        "invalid_citations"
    ]
]


baseline_citation_violations = [

    result

    for result
    in results

    if result[
        "baseline_citation_violation"
    ]
]


print(
    f"Invalid citation records   : "
    f"{len(invalid_citation_records)}"
)


print(
    f"Baseline citation violations: "
    f"{len(baseline_citation_violations)}"
)


# 20. DECISION DISTRIBUTIONS

# This is NOT accuracy evaluation.
# We are only counting the model outputs.
# No gold labels are loaded.


print(
    "\n[10] DECISION DISTRIBUTIONS"
)


decision_distributions = {}


for system_name in SYSTEM_ORDER:

    decisions = [

        result[
            "decision"
        ]

        for result
        in results

        if result[
            "system"
        ]
        ==
        system_name
    ]


    counts = Counter(
        decisions
    )


    decision_distributions[
        system_name
    ] = {

        "yes":
            counts.get(
                "yes",
                0
            ),

        "no":
            counts.get(
                "no",
                0
            ),

        "maybe":
            counts.get(
                "maybe",
                0
            )
    }


    print(
        f"{system_name:<22} "
        f"yes={counts.get('yes', 0):<4} "
        f"no={counts.get('no', 0):<4} "
        f"maybe={counts.get('maybe', 0):<4}"
    )


# 21. TOKEN USAGE

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


total_tokens = (
    total_input_tokens
    +
    total_output_tokens
)


print(
    "\n[11] TOKEN USAGE"
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
    f"{total_tokens}"
)


# 22. FINAL LEAKAGE CHECK

print(
    "\n[12] FINAL LEAKAGE CHECK"
)


forbidden_fields = {

    "gold_decision",

    "reference_answer",

    "gold_context_ids",

    "final_decision",

    "LONG_ANSWER",

    "reasoning_required_pred",

    "reasoning_free_pred"
}


leakage_count = 0


for result in results:

    overlap = (

        forbidden_fields

        &

        set(
            result.keys()
        )
    )


    if overlap:

        leakage_count += 1


print(
    f"Evaluation-only result fields : "
    f"{leakage_count}"
)


if leakage_count != 0:

    raise RuntimeError(
        "Evaluation data leakage "
        "detected in final results."
    )


print(
    "PASS: Final generated answers "
    "contain no evaluation labels "
    "or reference answers."
)


# 23. SAVE FINAL CHECKPOINT ONE LAST TIME

save_checkpoint()


# 24. SAVE GENERATION SUMMARY

summary = {

    "split":
        "test",

    "systems":
        SYSTEM_ORDER,

    "expected_responses":
        TOTAL_EXPECTED,

    "completed_responses":
        len(
            results
        ),

    "model":
        MODEL_NAME,

    "reasoning_effort":
        REASONING_EFFORT,

    "max_output_tokens":
        MAX_OUTPUT_TOKENS,

    "verbosity":
        "low",

    "structured_output":
        "strict_json_schema",

    "frozen_prompt_sha256":
        frozen_prompt_hash,

    "input_file_hashes":
        input_file_hashes,

    "token_usage": {

        "input":
            total_input_tokens,

        "output":
            total_output_tokens,

        "total":
            total_tokens
    },

    "invalid_citation_records":
        len(
            invalid_citation_records
        ),

    "baseline_citation_violations":
        len(
            baseline_citation_violations
        ),

    "decision_distributions":
        decision_distributions,

    "gold_labels_loaded":
        False,

    "reference_answers_loaded":
        False,

    "accuracy_metrics_calculated":
        False,

    "hallucination_metrics_calculated":
        False
}


save_json(
    summary,
    SUMMARY_PATH
)


# 25. FINAL OUTPUT

print(
    "\n[13] FINAL TEST GENERATION SUMMARY"
)


print(
    f"Completed : "
    f"{len(results)}/{TOTAL_EXPECTED}"
)


print(
    f"Model     : "
    f"{MODEL_NAME}"
)


print(
    f"Reasoning : "
    f"{REASONING_EFFORT}"
)


print(
    f"Max output: "
    f"{MAX_OUTPUT_TOKENS}"
)


print(
    "Verbosity : low"
)


print(
    "Gold labels loaded     : NO"
)


print(
    "Reference answers loaded: NO"
)


print(
    "Accuracy evaluated     : NO"
)


print(
    "Hallucination evaluated: NO"
)


print(
    "\n[14] FILES SAVED"
)


print(
    OUTPUT_PATH
)


print(
    SUMMARY_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 36 COMPLETE - "
    "FINAL TEST GENERATION FROZEN"
)


print(
    "=" * 78
)
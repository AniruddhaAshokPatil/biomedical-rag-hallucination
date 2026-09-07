import ast
import hashlib
import json
import random
import time

from collections import Counter
from pathlib import Path

from openai import OpenAI


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


ORIGINAL_FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)


FINAL_TEST_FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "final_test_artifact_manifest.json"
)


FROZEN_JUDGE_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "22_full_hallucination_evaluation.py"
)


TEST_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_questions.json"
)


GENERATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "final_test_generation_results.json"
)


GOLD_INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "gold_context_test_inputs.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_test_hallucination_v3_results.json"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_test_hallucination_v3_summary.json"
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


# 3. START

print("=" * 78)

print(
    "SECTION 40 - FINAL TEST HALLUCINATION EVALUATION V3"
)

print("=" * 78)


# 4. VERIFY ORIGINAL JUDGE IS STILL FROZEN

print(
    "\n[1] VERIFYING FROZEN V3 JUDGE"
)


original_manifest = load_json(
    ORIGINAL_FREEZE_MANIFEST
)


judge_manifest_key = None


for relative_path in (
    original_manifest[
        "file_hashes"
    ]
):

    if relative_path.endswith(
        "22_full_hallucination_evaluation.py"
    ):

        judge_manifest_key = (
            relative_path
        )

        break


if judge_manifest_key is None:

    raise RuntimeError(
        "Section 22 was not found "
        "in the original freeze manifest."
    )


expected_judge_hash = (
    original_manifest[
        "file_hashes"
    ][
        judge_manifest_key
    ]
)


actual_judge_hash = sha256_file(
    FROZEN_JUDGE_SOURCE
)


if (
    actual_judge_hash
    !=
    expected_judge_hash
):

    raise RuntimeError(
        "Frozen Section 22 judge "
        "has changed."
    )


print(
    "PASS: Frozen Section 22 judge "
    "is unchanged."
)


# 5. VERIFY FINAL GENERATED ANSWERS REMAIN FROZEN

print(
    "\n[2] VERIFYING FINAL TEST ARTIFACT FREEZE"
)


test_manifest = load_json(
    FINAL_TEST_FREEZE_MANIFEST
)


changed_files = []

missing_files = []


for relative_path, expected_hash in (
    test_manifest[
        "sha256"
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


    actual_hash = sha256_file(
        path
    )


    if (
        actual_hash
        !=
        expected_hash
    ):

        changed_files.append(
            relative_path
        )


print(
    f"Frozen artifacts checked : "
    f"{len(test_manifest['sha256'])}"
)


print(
    f"Changed artifacts        : "
    f"{len(changed_files)}"
)


print(
    f"Missing artifacts        : "
    f"{len(missing_files)}"
)


if (
    changed_files
    or
    missing_files
):

    raise RuntimeError(
        "Final-test artifacts changed "
        "after generation."
    )


print(
    "PASS: Final generated answers "
    "remain unchanged."
)


# 6. EXTRACT EXACT FROZEN V3 OBJECTS

# Do not manually recreate the judge.
# Extract exact values from frozen Section 22:

#   JUDGE_VERSION
#   JUDGE_MODEL
#   REASONING_EFFORT
#   MAX_OUTPUT_TOKENS
#   RANDOM_SEED
#   SYSTEM_ORDER
#   ANONYMOUS_IDS
#   JUDGE_SCHEMA
#   JUDGE_INSTRUCTIONS
#   format_gold_evidence()



print(
    "\n[3] LOADING EXACT FROZEN V3 CONFIGURATION"
)


source = FROZEN_JUDGE_SOURCE.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source,
    filename=str(
        FROZEN_JUDGE_SOURCE
    )
)


assignment_names = {

    "JUDGE_VERSION",
    "JUDGE_MODEL",
    "REASONING_EFFORT",
    "MAX_OUTPUT_TOKENS",
    "RANDOM_SEED",
    "SYSTEM_ORDER",
    "ANONYMOUS_IDS",
    "JUDGE_SCHEMA",
    "JUDGE_INSTRUCTIONS"
}


selected_nodes = []


for node in tree.body:

    if isinstance(
        node,
        ast.Assign
    ):

        for target in node.targets:

            if (
                isinstance(
                    target,
                    ast.Name
                )
                and
                target.id
                in
                assignment_names
            ):

                selected_nodes.append(
                    node
                )

                break


    elif isinstance(
        node,
        ast.FunctionDef
    ):

        if (
            node.name
            ==
            "format_gold_evidence"
        ):

            selected_nodes.append(
                node
            )


module = ast.Module(
    body=selected_nodes,
    type_ignores=[]
)


ast.fix_missing_locations(
    module
)


namespace = {}


exec(
    compile(
        module,
        filename=str(
            FROZEN_JUDGE_SOURCE
        ),
        mode="exec"
    ),
    namespace
)


required_objects = (
    assignment_names
    |
    {
        "format_gold_evidence"
    }
)


missing_objects = (

    required_objects
    -
    set(
        namespace.keys()
    )
)


if missing_objects:

    raise RuntimeError(
        "Could not extract frozen "
        f"V3 objects: "
        f"{sorted(missing_objects)}"
    )


JUDGE_VERSION = (
    namespace[
        "JUDGE_VERSION"
    ]
)


JUDGE_MODEL = (
    namespace[
        "JUDGE_MODEL"
    ]
)


REASONING_EFFORT = (
    namespace[
        "REASONING_EFFORT"
    ]
)


MAX_OUTPUT_TOKENS = (
    namespace[
        "MAX_OUTPUT_TOKENS"
    ]
)


RANDOM_SEED = (
    namespace[
        "RANDOM_SEED"
    ]
)


SYSTEM_ORDER = (
    namespace[
        "SYSTEM_ORDER"
    ]
)


ANONYMOUS_IDS = (
    namespace[
        "ANONYMOUS_IDS"
    ]
)


JUDGE_SCHEMA = (
    namespace[
        "JUDGE_SCHEMA"
    ]
)


JUDGE_INSTRUCTIONS = (
    namespace[
        "JUDGE_INSTRUCTIONS"
    ]
)


format_gold_evidence = (
    namespace[
        "format_gold_evidence"
    ]
)


print(
    f"Judge version      : "
    f"{JUDGE_VERSION}"
)


print(
    f"Judge model        : "
    f"{JUDGE_MODEL}"
)


print(
    f"Reasoning effort   : "
    f"{REASONING_EFFORT}"
)


print(
    f"Max output tokens  : "
    f"{MAX_OUTPUT_TOKENS}"
)


print(
    f"Random seed        : "
    f"{RANDOM_SEED}"
)


print(
    f"Systems            : "
    f"{SYSTEM_ORDER}"
)


print(
    f"Anonymous IDs      : "
    f"{ANONYMOUS_IDS}"
)


# 7. SAFETY CHECK EXPECTED FROZEN SETTINGS

if (
    JUDGE_VERSION
    !=
    "v3_final"
):

    raise RuntimeError(
        "Unexpected judge version."
    )


if (
    JUDGE_MODEL
    !=
    "gpt-5.4-2026-03-05"
):

    raise RuntimeError(
        "Unexpected frozen judge model."
    )


if (
    REASONING_EFFORT
    !=
    "low"
):

    raise RuntimeError(
        "Unexpected reasoning setting."
    )


if (
    MAX_OUTPUT_TOKENS
    !=
    3500
):

    raise RuntimeError(
        "Unexpected maximum output "
        "token setting."
    )


if (
    RANDOM_SEED
    !=
    42
):

    raise RuntimeError(
        "Unexpected V3 random seed."
    )


print(
    "PASS: V3 configuration matches "
    "the frozen evaluator."
)


# 8. LOAD FROZEN TEST MATERIAL

print(
    "\n[4] LOADING FROZEN TEST MATERIAL"
)


test_questions = load_json(
    TEST_QUESTIONS_PATH
)


generation_data = load_json(
    GENERATION_RESULTS_PATH
)


generation_results = (
    generation_data[
        "results"
    ]
)


gold_inputs = load_json(
    GOLD_INPUT_PATH
)


print(
    f"Questions          : "
    f"{len(test_questions)}"
)


print(
    f"Generated answers  : "
    f"{len(generation_results)}"
)


print(
    f"Gold input records : "
    f"{len(gold_inputs)}"
)


if len(
    test_questions
) != 500:

    raise ValueError(
        "Expected 500 test questions."
    )


if len(
    generation_results
) != 2000:

    raise ValueError(
        "Expected 2,000 generated answers."
    )


if len(
    gold_inputs
) != 500:

    raise ValueError(
        "Expected 500 Gold-context records."
    )


# 9. INDEX TEST DATA

question_by_id = {

    str(
        item[
            "question_id"
        ]
    ):
        item[
            "question"
        ]

    for item
    in test_questions
}


gold_by_id = {

    str(
        item[
            "question_id"
        ]
    ):
        item

    for item
    in gold_inputs
}


answers_by_question = {}


for result in generation_results:

    question_id = str(
        result[
            "question_id"
        ]
    )


    system_name = (
        result[
            "system"
        ]
    )


    if system_name not in SYSTEM_ORDER:

        raise ValueError(
            f"Unexpected system: "
            f"{system_name}"
        )


    answers_by_question.setdefault(
        question_id,
        {}
    )


    if (
        system_name
        in
        answers_by_question[
            question_id
        ]
    ):

        raise ValueError(
            f"Duplicate answer for "
            f"{question_id} / "
            f"{system_name}"
        )


    answers_by_question[
        question_id
    ][
        system_name
    ] = result


question_ids = [

    str(
        item[
            "question_id"
        ]
    )

    for item
    in test_questions
]


# 10. VALIDATE COMPLETE PAIRED DESIGN

print(
    "\n[5] VALIDATING PAIRED HALLUCINATION INPUTS"
)


for question_id in question_ids:

    if question_id not in gold_by_id:

        raise KeyError(
            f"Missing Gold evidence "
            f"for {question_id}."
        )


    if question_id not in answers_by_question:

        raise KeyError(
            f"Missing generated answers "
            f"for {question_id}."
        )


    systems_present = set(
        answers_by_question[
            question_id
        ].keys()
    )


    if (
        systems_present
        !=
        set(
            SYSTEM_ORDER
        )
    ):

        raise ValueError(
            f"Question {question_id} "
            f"does not have all four "
            f"systems."
        )


print(
    "PASS: 500 questions x 4 systems "
    "are ready."
)


# 11. BUILD GOLD EVIDENCE

# Section 35 Gold contexts are already frozen.

# Relabel them G1, G2, ... exactly for the V3 judge.

# No generated system's retrieved context is used here.

# Every system is judged against the SAME gold evidence.



gold_evidence_by_question = {}


for question_id in question_ids:

    contexts = (
        gold_by_id[
            question_id
        ][
            "contexts"
        ]
    )


    evidence = []


    for index, context in enumerate(
        contexts,
        start=1
    ):

        evidence.append(
            {
                "label":
                    f"G{index}",

                "section":
                    context.get(
                        "section"
                    ),

                "text":
                    context[
                        "text"
                    ]
            }
        )


    if not evidence:

        raise ValueError(
            f"No Gold evidence "
            f"for {question_id}."
        )


    gold_evidence_by_question[
        question_id
    ] = evidence


print(
    "PASS: Common Gold evidence "
    "prepared for all systems."
)


# 12. DETERMINISTIC BLINDING

print(
    "\n[6] DETERMINISTIC BLINDING"
)


answer_mappings = {}


for question_id in question_ids:

    shuffled_systems = (
        SYSTEM_ORDER.copy()
    )


    question_random = random.Random(
        RANDOM_SEED
        +
        int(
            question_id
        )
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
    "PASS: Every question uses "
    "deterministic anonymous A/B/C/D labels."
)


# 13. CURRENT INPUT HASHES

current_input_hashes = {

    "generation_results":
        sha256_file(
            GENERATION_RESULTS_PATH
        ),

    "gold_context_inputs":
        sha256_file(
            GOLD_INPUT_PATH
        ),

    "frozen_judge_source":
        sha256_file(
            FROZEN_JUDGE_SOURCE
        )
}


# 14. LOAD CHECKPOINT

if OUTPUT_PATH.exists():

    print(
        "\n[7] CHECKPOINT FOUND"
    )


    existing = load_json(
        OUTPUT_PATH
    )


    if (
        existing.get(
            "judge_version"
        )
        !=
        JUDGE_VERSION
    ):

        raise RuntimeError(
            "Checkpoint judge version "
            "does not match."
        )


    if (
        existing.get(
            "judge_model"
        )
        !=
        JUDGE_MODEL
    ):

        raise RuntimeError(
            "Checkpoint judge model "
            "does not match."
        )


    if (
        existing.get(
            "input_hashes"
        )
        !=
        current_input_hashes
    ):

        raise RuntimeError(
            "Frozen inputs changed since "
            "the hallucination checkpoint "
            "was created."
        )


    judgments = existing.get(
        "judgments",
        []
    )


    print(
        f"Completed questions : "
        f"{len(judgments)}"
    )


else:

    print(
        "\n[7] NO EXISTING CHECKPOINT"
    )


    judgments = []


completed_question_ids = {

    str(
        judgment[
            "question_id"
        ]
    )

    for judgment
    in judgments
}


if (
    len(
        completed_question_ids
    )
    !=
    len(
        judgments
    )
):

    raise RuntimeError(
        "Duplicate questions detected "
        "in hallucination checkpoint."
    )


print(
    f"Already completed : "
    f"{len(completed_question_ids)}"
)


print(
    f"Remaining         : "
    f"{500 - len(completed_question_ids)}"
)


# 15. CHECKPOINT FUNCTION

def save_checkpoint():

    checkpoint = {

        "split":
            "test",

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
            len(
                judgments
            ),

        "expected_system_judgments":
            2000,

        "input_hashes":
            current_input_hashes,

        "judgments":
            judgments
    }


    save_json(
        checkpoint,
        OUTPUT_PATH
    )


# 16. OPENAI CLIENT

print(
    "\n[8] CREATING OPENAI JUDGE CLIENT"
)


client = OpenAI()


print(
    "OpenAI client created."
)


# 17. RUN FROZEN V3 JUDGE

print(
    "\n[9] RUNNING FINAL TEST V3 JUDGE"
)


for question_number, question_id in enumerate(
    question_ids,
    start=1
):

    if (
        question_id
        in
        completed_question_ids
    ):

        continue


    question = (
        question_by_id[
            question_id
        ]
    )


    gold_evidence = (
        gold_evidence_by_question[
            question_id
        ]
    )


    evidence_text = (
        format_gold_evidence(
            gold_evidence
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
            answers_by_question[
                question_id
            ][
                system_name
            ][
                "answer"
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
        f"{question}\n\n"
        f"GOLD EVIDENCE:\n"
        f"{evidence_text}\n\n"
        f"GENERATED ANSWERS:\n"
        f"{answers_text}"
    )


    print(
        f"\n[{question_number}/500] "
        f"Question ID: {question_id}"
    )


    try:

        response = client.responses.create(

            model=
                JUDGE_MODEL,

            instructions=(
                JUDGE_INSTRUCTIONS
            ),

            input=
                user_prompt,

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


        if not response.output_text:

            raise RuntimeError(
                "Judge returned no "
                "output_text."
            )


        parsed = json.loads(
            response.output_text
        )


        # VALIDATE RETURNED ANSWER IDS

        returned_ids = [

            answer_result[
                "answer_id"
            ]

            for answer_result
            in parsed[
                "answers"
            ]
        ]


        if (
            len(
                returned_ids
            )
            !=
            4
            or
            set(
                returned_ids
            )
            !=
            set(
                ANONYMOUS_IDS
            )
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
            in gold_evidence
        }


        system_results = {}


        for answer_result in (
            parsed[
                "answers"
            ]
        ):

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


            # Validate evidence labels

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

                for claim
                in claims

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

                for claim
                in claims

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

                for claim
                in claims

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


            for excluded in (
                excluded_statements
            ):

                reason = (
                    excluded[
                        "reason"
                    ]
                )


                if (
                    reason
                    not in
                    exclusion_counts
                ):

                    raise ValueError(
                        "Unexpected exclusion "
                        f"reason: {reason}"
                    )


                exclusion_counts[
                    reason
                ] += 1


            generation_record = (
                answers_by_question[
                    question_id
                ][
                    system_name
                ]
            )


            system_results[
                system_name
            ] = {

                "anonymous_answer_id":
                    anonymous_id,

                "generated_answer":
                    generation_record[
                        "answer"
                    ],

                "generated_decision":
                    generation_record[
                        "decision"
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


        if (
            set(
                system_results.keys()
            )
            !=
            set(
                SYSTEM_ORDER
            )
        ):

            raise ValueError(
                "Judge results could not be "
                "mapped back to all systems."
            )


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


        judgments.append(
            {

                "question_id":
                    question_id,

                "question":
                    question,

                "gold_evidence":
                    gold_evidence,

                "blind_mapping":
                    mapping,

                "systems":
                    system_results,

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens
            }
        )


        completed_question_ids.add(
            question_id
        )

        # CHECKPOINT AFTER EVERY SUCCESS

        save_checkpoint()


        print(
            f"  Completed systems : "
            f"{len(system_results)}"
        )


        print(
            f"  Tokens            : "
            f"{input_tokens} in / "
            f"{output_tokens} out"
        )


        time.sleep(
            0.05
        )


    except Exception as error:

        print(
            "\n"
            + "=" * 78
        )


        print(
            "V3 JUDGE STOPPED"
        )


        print(
            "=" * 78
        )


        print(
            f"Question ID : "
            f"{question_id}"
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
            "\nAll previously completed "
            "questions are checkpointed."
        )


        print(
            "Fix the issue and rerun "
            "the same script to resume."
        )


        print(
            "\nDo NOT change the judge "
            "model, prompt, schema, or "
            "classification rules."
        )


        raise


# 18. FINAL COMPLETENESS CHECK

print(
    "\n[10] FINAL COMPLETENESS CHECK"
)


print(
    f"Questions judged : "
    f"{len(judgments)}"
)


if len(
    judgments
) != 500:

    raise ValueError(
        "Expected exactly 500 "
        "question-level judgments."
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
    f"System judgments : "
    f"{total_system_judgments}"
)


if (
    total_system_judgments
    !=
    2000
):

    raise ValueError(
        "Expected exactly 2,000 "
        "system judgments."
    )


print(
    "PASS: 500 grouped judge calls "
    "produced 2,000 system judgments."
)


# 19. VALIDATION COUNTS

print(
    "\n[11] JUDGE OUTPUT VALIDATION"
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


        if (
            result[
                "invalid_evidence_labels"
            ]
        ):

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
    "\n[12] FINAL TEST HALLUCINATION SUMMARY"
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


# 21. SCORECARD

print(
    "\n[13] HALLUCINATION SCORECARD"
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


# 22. TOKEN USAGE

total_input_tokens = sum(

    judgment[
        "input_tokens"
    ]

    for judgment
    in judgments
)


total_output_tokens = sum(

    judgment[
        "output_tokens"
    ]

    for judgment
    in judgments
)


print(
    "\n[14] JUDGE TOKEN USAGE"
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


# 23. FINAL SAVE

save_checkpoint()


summary_output = {

    "split":
        "test",

    "analysis":
        "confirmatory_hallucination_evaluation",

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

    "blinding":
        "deterministic_anonymous_A_B_C_D",

    "evidence":
        "same_gold_evidence_for_all_systems",

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
                "answers containing no scored "
                "biomedical/research factual claims"
            )
    },

    "system_summaries":
        system_summaries,

    "validation": {

        "invalid_evidence_cases":
            invalid_evidence_cases,

        "zero_claim_answers_total":
            zero_claim_answers,

        "excluded_statements_total":
            total_excluded_statements
    },

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

    "statistical_tests_performed":
        False,

    "human_calibration_applied":
        False
}


save_json(
    summary_output,
    SUMMARY_PATH
)


print(
    "\n[15] FILES SAVED"
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
    "SECTION 40 COMPLETE - "
    "FINAL TEST HALLUCINATION V3 COMPLETE"
)


print(
    "=" * 78
)

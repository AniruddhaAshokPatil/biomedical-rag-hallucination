import ast
import hashlib
import json

from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


TEST_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_questions.json"
)


RETRIEVAL_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_corpus.json"
)


GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_ground_truth.json"
)


BASIC_CONTEXT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "basic_rag_test_contexts.json"
)


ADVANCED_CONTEXT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "advanced_rag_test_contexts.json"
)


FROZEN_SECTION_12_PATH = (
    PROJECT_ROOT
    / "src"
    / "12_prepare_llm_inputs.py"
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BASELINE_OUTPUT = (
    OUTPUT_DIR
    / "baseline_test_inputs.json"
)


BASIC_OUTPUT = (
    OUTPUT_DIR
    / "basic_rag_test_inputs.json"
)


ADVANCED_OUTPUT = (
    OUTPUT_DIR
    / "advanced_rag_test_inputs.json"
)


GOLD_OUTPUT = (
    OUTPUT_DIR
    / "gold_context_test_inputs.json"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "test_generation_input_summary.json"
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
    "SECTION 35 - PREPARE FINAL TEST LLM INPUTS"
)

print("=" * 78)


# 4. VERIFY ORIGINAL FROZEN PROTOCOL

print(
    "\n[1] VERIFYING FROZEN PROTOCOL"
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


    actual_hash = sha256_file(
        path
    )


    if actual_hash != expected_hash:

        changed_files.append(
            relative_path
        )


if missing_files:

    print(
        "Missing frozen files:"
    )


    for filename in missing_files:

        print(
            f"  {filename}"
        )


if changed_files:

    print(
        "Changed frozen files:"
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
        "Frozen protocol integrity "
        "check failed."
    )


print(
    "PASS: All original frozen files "
    "remain unchanged."
)


# 5. EXTRACT EXACT FROZEN SECTION 12 PROMPT CODE

# We extract:
# - SYSTEM_PROMPT
# - format_contexts()
# - build_user_prompt()
# directly from frozen Section 12.
# This prevents accidental prompt drift.


print(
    "\n[2] LOADING EXACT FROZEN PROMPT"
)


source = FROZEN_SECTION_12_PATH.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source,
    filename=str(
        FROZEN_SECTION_12_PATH
    )
)


selected_nodes = []


for node in tree.body:

    # SYSTEM_PROMPT assignment

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
                ==
                "SYSTEM_PROMPT"
            ):

                selected_nodes.append(
                    node
                )

    # Frozen helper functions

    if isinstance(
        node,
        ast.FunctionDef
    ):

        if node.name in {
            "format_contexts",
            "build_user_prompt"
        }:

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
            FROZEN_SECTION_12_PATH
        ),
        mode="exec"
    ),
    namespace
)


required_names = {
    "SYSTEM_PROMPT",
    "format_contexts",
    "build_user_prompt"
}


missing_names = (
    required_names
    -
    set(
        namespace.keys()
    )
)


if missing_names:

    raise RuntimeError(
        "Could not extract frozen "
        f"Section 12 objects: "
        f"{sorted(missing_names)}"
    )


SYSTEM_PROMPT = (
    namespace[
        "SYSTEM_PROMPT"
    ]
)


format_contexts = (
    namespace[
        "format_contexts"
    ]
)


build_user_prompt = (
    namespace[
        "build_user_prompt"
    ]
)


print(
    "PASS: Frozen SYSTEM_PROMPT loaded."
)


print(
    "PASS: Frozen format_contexts() loaded."
)


print(
    "PASS: Frozen build_user_prompt() loaded."
)


print(
    f"Prompt SHA256 : "
    f"{sha256_text(SYSTEM_PROMPT)}"
)


# 6. LOAD NON-EVALUATION INPUT DATA

print(
    "\n[3] LOADING TEST INPUT DATA"
)


test_questions = load_json(
    TEST_QUESTIONS_PATH
)


retrieval_corpus = load_json(
    RETRIEVAL_CORPUS_PATH
)


basic_contexts = load_json(
    BASIC_CONTEXT_PATH
)


advanced_contexts = load_json(
    ADVANCED_CONTEXT_PATH
)


print(
    f"Test questions   : "
    f"{len(test_questions)}"
)


print(
    f"Corpus documents : "
    f"{len(retrieval_corpus)}"
)


print(
    f"Basic contexts   : "
    f"{len(basic_contexts)}"
)


print(
    f"Advanced contexts: "
    f"{len(advanced_contexts)}"
)


if len(test_questions) != 500:

    raise ValueError(
        "Expected 500 test questions."
    )


if len(basic_contexts) != 500:

    raise ValueError(
        "Expected 500 Basic RAG "
        "context records."
    )


if len(advanced_contexts) != 500:

    raise ValueError(
        "Expected 500 Advanced RAG "
        "context records."
    )


# 7. INDEX DATA

corpus_by_document_id = {

    str(
        document[
            "document_id"
        ]
    ):
        document

    for document
    in retrieval_corpus
}


basic_by_id = {

    str(
        item[
            "question_id"
        ]
    ):
        item

    for item
    in basic_contexts
}


advanced_by_id = {

    str(
        item[
            "question_id"
        ]
    ):
        item

    for item
    in advanced_contexts
}


# 8. LOAD ONLY GOLD CONTEXT IDS

# The Gold-context control requires the known source
# passages.
# We therefore access ONLY:
#  gold_context_ids
# We deliberately DO NOT access:
# gold_decision
# reference_answer


print(
    "\n[4] LOADING GOLD-CONTEXT IDS ONLY"
)


raw_ground_truth = load_json(
    GROUND_TRUTH_PATH
)


gold_ids_by_question = {}


for question_record in test_questions:

    question_id = str(
        question_record[
            "question_id"
        ]
    )


    if question_id not in raw_ground_truth:

        raise KeyError(
            f"Question {question_id} "
            f"missing from evaluation "
            f"ground truth."
        )


    entry = raw_ground_truth[
        question_id
    ]


    if "gold_context_ids" not in entry:

        raise KeyError(
            f"Question {question_id} "
            f"has no gold_context_ids."
        )


    gold_context_ids = (
        entry[
            "gold_context_ids"
        ]
    )


    if not isinstance(
        gold_context_ids,
        list
    ):

        raise TypeError(
            f"gold_context_ids for "
            f"{question_id} is not a list."
        )


    gold_ids_by_question[
        question_id
    ] = [

        str(
            document_id
        )

        for document_id
        in gold_context_ids
    ]

# Delete full ground-truth object immediately.
# From this point onwards the program retains
# ONLY the context IDs required for Gold control.

del raw_ground_truth


print(
    "PASS: Only gold_context_ids retained."
)


print(
    "Gold decisions accessed     : NO"
)


print(
    "Reference answers accessed : NO"
)


# 9. OUTPUT STORAGE

baseline_inputs = []

basic_inputs = []

advanced_inputs = []

gold_inputs = []


# 10. BUILD FOUR FINAL TEST CONDITIONS

print(
    "\n[5] BUILDING FOUR GENERATION CONDITIONS"
)


for index, question_record in enumerate(
    test_questions,
    start=1
):

    question_id = str(
        question_record[
            "question_id"
        ]
    )


    question = (
        question_record[
            "question"
        ]
    )


    # A. BASELINE LLM

    baseline_inputs.append(
        {

            "question_id":
                question_id,

            "system":
                "baseline_llm",

            "question":
                question,

            "contexts":
                [],

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    []
                )
        }
    )


    # B. BASIC RAG

    if question_id not in basic_by_id:

        raise KeyError(
            f"Basic context missing for "
            f"{question_id}."
        )


    basic_raw_contexts = (
        basic_by_id[
            question_id
        ][
            "contexts"
        ]
    )


    if len(
        basic_raw_contexts
    ) != 5:

        raise ValueError(
            f"Basic RAG question "
            f"{question_id} does not "
            f"have exactly 5 contexts."
        )


    basic_formatted = format_contexts(
        basic_raw_contexts
    )


    basic_inputs.append(
        {

            "question_id":
                question_id,

            "system":
                "basic_rag",

            "question":
                question,

            "contexts":
                basic_formatted,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    basic_formatted
                )
        }
    )


    # C. ADVANCED RAG

    if question_id not in advanced_by_id:

        raise KeyError(
            f"Advanced context missing for "
            f"{question_id}."
        )


    advanced_raw_contexts = (
        advanced_by_id[
            question_id
        ][
            "contexts"
        ]
    )


    if len(
        advanced_raw_contexts
    ) != 5:

        raise ValueError(
            f"Advanced RAG question "
            f"{question_id} does not "
            f"have exactly 5 contexts."
        )


    advanced_formatted = format_contexts(
        advanced_raw_contexts
    )


    advanced_inputs.append(
        {

            "question_id":
                question_id,

            "system":
                "advanced_rag",

            "question":
                question,

            "contexts":
                advanced_formatted,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    advanced_formatted
                )
        }
    )


    # D. GOLD-CONTEXT CONTROL

    gold_context_ids = (
        gold_ids_by_question[
            question_id
        ]
    )


    if not gold_context_ids:

        raise ValueError(
            f"Question {question_id} "
            f"has no Gold contexts."
        )


    gold_raw_contexts = []


    for document_id in gold_context_ids:

        if (
            document_id
            not in
            corpus_by_document_id
        ):

            raise KeyError(
                f"Gold document "
                f"{document_id} "
                f"for question "
                f"{question_id} "
                f"is missing from corpus."
            )


        document = (
            corpus_by_document_id[
                document_id
            ]
        )


        gold_raw_contexts.append(
            {

                "section":
                    document.get(
                        "section"
                    ),

                "text":
                    document[
                        "text"
                    ]
            }
        )


    gold_formatted = format_contexts(
        gold_raw_contexts
    )


    gold_inputs.append(
        {

            "question_id":
                question_id,

            "system":
                "gold_context_control",

            "question":
                question,

            "contexts":
                gold_formatted,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    gold_formatted
                )
        }
    )


    if index % 100 == 0:

        print(
            f"Prepared "
            f"{index}/500"
        )


# 11. STRUCTURAL VALIDATION

print(
    "\n[6] STRUCTURAL VALIDATION"
)


datasets = {

    "baseline":
        baseline_inputs,

    "basic":
        basic_inputs,

    "advanced":
        advanced_inputs,

    "gold":
        gold_inputs
}


for name, dataset in datasets.items():

    if len(
        dataset
    ) != 500:

        raise ValueError(
            f"{name} contains "
            f"{len(dataset)} records, "
            f"not 500."
        )


test_ids = {

    str(
        record[
            "question_id"
        ]
    )

    for record
    in test_questions
}


for name, dataset in datasets.items():

    ids = {

        str(
            record[
                "question_id"
            ]
        )

        for record
        in dataset
    }


    if ids != test_ids:

        raise ValueError(
            f"{name} IDs do not match "
            f"the frozen test IDs."
        )


print(
    "PASS: All four systems contain "
    "exactly 500 questions."
)


print(
    "PASS: All four systems use "
    "identical test-question IDs."
)


# 12. CONTEXT VALIDATION

print(
    "\n[7] CONTEXT VALIDATION"
)


baseline_nonempty = sum(

    1

    for item
    in baseline_inputs

    if item[
        "contexts"
    ]
)


basic_wrong_count = sum(

    1

    for item
    in basic_inputs

    if len(
        item[
            "contexts"
        ]
    ) != 5
)


advanced_wrong_count = sum(

    1

    for item
    in advanced_inputs

    if len(
        item[
            "contexts"
        ]
    ) != 5
)


gold_empty = sum(

    1

    for item
    in gold_inputs

    if not item[
        "contexts"
    ]
)


print(
    f"Baseline with contexts : "
    f"{baseline_nonempty}"
)


print(
    f"Basic wrong count      : "
    f"{basic_wrong_count}"
)


print(
    f"Advanced wrong count   : "
    f"{advanced_wrong_count}"
)


print(
    f"Gold with no contexts  : "
    f"{gold_empty}"
)


if baseline_nonempty != 0:

    raise ValueError(
        "Baseline unexpectedly "
        "contains evidence."
    )


if basic_wrong_count != 0:

    raise ValueError(
        "Basic context-count "
        "validation failed."
    )


if advanced_wrong_count != 0:

    raise ValueError(
        "Advanced context-count "
        "validation failed."
    )


if gold_empty != 0:

    raise ValueError(
        "Gold-context validation failed."
    )


print(
    "PASS: Context assignment matches "
    "the frozen four-system design."
)


# 13. SAME-PROMPT VALIDATION

print(
    "\n[8] PROMPT CONSISTENCY"
)


prompt_hashes = set()


for dataset in datasets.values():

    for item in dataset:

        prompt_hashes.add(
            sha256_text(
                item[
                    "system_prompt"
                ]
            )
        )


print(
    f"Unique system-prompt hashes : "
    f"{len(prompt_hashes)}"
)


if len(prompt_hashes) != 1:

    raise ValueError(
        "Systems do not share the "
        "same frozen system prompt."
    )


print(
    "PASS: All 2,000 generation inputs "
    "use exactly the same system prompt."
)


# 14. BASELINE PROMPT CHECK

baseline_bad = []


for item in baseline_inputs:

    prompt = item[
        "user_prompt"
    ]


    if (
        "NO EXTERNAL EVIDENCE PROVIDED."
        not in
        prompt
    ):

        baseline_bad.append(
            item[
                "question_id"
            ]
        )


if baseline_bad:

    raise ValueError(
        "Baseline no-evidence prompt "
        "validation failed."
    )


print(
    "PASS: Every baseline prompt states "
    "NO EXTERNAL EVIDENCE PROVIDED."
)


# 15. LEAKAGE FIELD CHECK

print(
    "\n[9] GENERATION LEAKAGE CHECK"
)


forbidden_keys = {

    "source_record_id",

    "gold_context_ids",

    "gold_decision",

    "reference_answer",

    "final_decision",

    "LONG_ANSWER",

    "reasoning_required_pred",

    "reasoning_free_pred"
}


def find_forbidden_keys(
    value,
    path="root"
):

    problems = []


    if isinstance(
        value,
        dict
    ):

        for key, child in value.items():

            if key in forbidden_keys:

                problems.append(
                    f"{path}.{key}"
                )


            problems.extend(
                find_forbidden_keys(
                    child,
                    f"{path}.{key}"
                )
            )


    elif isinstance(
        value,
        list
    ):

        for index, child in enumerate(
            value
        ):

            problems.extend(
                find_forbidden_keys(
                    child,
                    f"{path}[{index}]"
                )
            )


    return problems


leakage_problems = []


for name, dataset in datasets.items():

    problems = find_forbidden_keys(
        dataset,
        path=name
    )


    leakage_problems.extend(
        problems
    )


print(
    f"Forbidden output fields found : "
    f"{len(leakage_problems)}"
)


if leakage_problems:

    for problem in leakage_problems[
        :20
    ]:

        print(
            problem
        )


    raise ValueError(
        "Evaluation-only field leakage "
        "detected in generation inputs."
    )


print(
    "PASS: No evaluation-only fields "
    "appear in generation inputs."
)


print(
    "PASS: Gold decision and reference "
    "answer were never accessed."
)


# 16. GOLD CONTEXT STATISTICS

gold_context_counts = [

    len(
        item[
            "contexts"
        ]
    )

    for item
    in gold_inputs
]


gold_context_mean = (
    sum(
        gold_context_counts
    )
    /
    len(
        gold_context_counts
    )
)


# 17. SAVE INPUT FILES

save_json(
    baseline_inputs,
    BASELINE_OUTPUT
)


save_json(
    basic_inputs,
    BASIC_OUTPUT
)


save_json(
    advanced_inputs,
    ADVANCED_OUTPUT
)


save_json(
    gold_inputs,
    GOLD_OUTPUT
)


# 18. SAVE PREPARATION SUMMARY

summary = {

    "split":
        "test",

    "number_of_questions":
        500,

    "number_of_systems":
        4,

    "total_generation_inputs":
        2000,

    "prompt_source":
        "src/12_prepare_llm_inputs.py",

    "prompt_sha256":
        sha256_text(
            SYSTEM_PROMPT
        ),

    "conditions": {

        "baseline_llm": {
            "questions": 500,
            "context_count": 0
        },

        "basic_rag": {
            "questions": 500,
            "context_count_per_question": 5
        },

        "advanced_rag": {
            "questions": 500,
            "context_count_per_question": 5
        },

        "gold_context_control": {
            "questions": 500,
            "minimum_contexts":
                min(
                    gold_context_counts
                ),
            "maximum_contexts":
                max(
                    gold_context_counts
                ),
            "mean_contexts":
                gold_context_mean
        }
    },

    "evaluation_access": {

        "gold_context_ids":
            True,

        "gold_decision":
            False,

        "reference_answer":
            False,

        "retrieval_metrics":
            False,

        "decision_metrics":
            False
    }
}


save_json(
    summary,
    SUMMARY_OUTPUT
)



# 19. FINAL OUTPUT


print(
    "\n[10] FINAL GENERATION INPUT SUMMARY"
)


print(
    f"Baseline : "
    f"{len(baseline_inputs)}"
)


print(
    f"Basic    : "
    f"{len(basic_inputs)}"
)


print(
    f"Advanced : "
    f"{len(advanced_inputs)}"
)


print(
    f"Gold     : "
    f"{len(gold_inputs)}"
)


print(
    f"Total    : "
    f"{sum(len(x) for x in datasets.values())}"
)


print(
    "\nGold context counts"
)


print(
    f"  Minimum : "
    f"{min(gold_context_counts)}"
)


print(
    f"  Maximum : "
    f"{max(gold_context_counts)}"
)


print(
    f"  Mean    : "
    f"{gold_context_mean:.3f}"
)


print(
    "\nEvaluation access"
)


print(
    "  gold_context_ids : YES"
)


print(
    "  gold_decision    : NO"
)


print(
    "  reference_answer : NO"
)


print(
    "  metrics computed : NO"
)


print(
    "\n[11] FILES SAVED"
)


print(
    BASELINE_OUTPUT
)


print(
    BASIC_OUTPUT
)


print(
    ADVANCED_OUTPUT
)


print(
    GOLD_OUTPUT
)


print(
    SUMMARY_OUTPUT
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 35 COMPLETE - "
    "FINAL GENERATION INPUTS READY"
)


print(
    "=" * 78
)
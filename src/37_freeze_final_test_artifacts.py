import hashlib
import json

from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "final_test_artifact_manifest.json"
)


# 2. ARTIFACTS TO FREEZE

ARTIFACT_PATHS = [

    # Section 34 - retrieval

    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "basic_dense_test_results.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "advanced_reranked_test_results.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "basic_rag_test_contexts.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "advanced_rag_test_contexts.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "test_retrieval_summary.json",


    # Section 35 - controlled generation inputs

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "baseline_test_inputs.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "basic_rag_test_inputs.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "advanced_rag_test_inputs.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "gold_context_test_inputs.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "test_generation_input_summary.json",


    # Section 36 - final generated answers

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "final_test_generation_results.json",

    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "final_test_generation_summary.json",


    # Scripts that created the final-test artifacts

    PROJECT_ROOT
    / "src"
    / "34_final_test_retrieval.py",

    PROJECT_ROOT
    / "src"
    / "35_prepare_final_test_llm_inputs.py",

    PROJECT_ROOT
    / "src"
    / "36_final_test_generation.py"
]


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

        return json.load(
            file
        )


def save_json(
    data,
    path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


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


def relative_path(path):

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/"
    )


# 4. START

print("=" * 78)

print(
    "SECTION 37 - FREEZE FINAL TEST ARTIFACTS"
)

print("=" * 78)



# 5. VERIFY ALL ARTIFACTS EXIST


print(
    "\n[1] VERIFYING FINAL TEST ARTIFACTS"
)


missing_files = []


for path in ARTIFACT_PATHS:

    if not path.exists():

        missing_files.append(
            path
        )


if missing_files:

    print(
        "Missing files:"
    )


    for path in missing_files:

        print(
            path
        )


    raise FileNotFoundError(
        "Cannot freeze final test artifacts "
        "because required files are missing."
    )


print(
    f"Required artifacts found : "
    f"{len(ARTIFACT_PATHS)}"
)


# 6. VERIFY GENERATION COMPLETENESS

print(
    "\n[2] VERIFYING FINAL GENERATION"
)


generation_path = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "generation"
    / "final_test_generation_results.json"
)


generation_data = load_json(
    generation_path
)


results = generation_data[
    "results"
]


print(
    f"Generated responses : "
    f"{len(results)}"
)


if len(results) != 2000:

    raise ValueError(
        "Expected exactly 2,000 "
        "final generated responses."
    )


systems = {

    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
}


system_counts = {}


for system in systems:

    count = sum(

        1

        for result
        in results

        if result[
            "system"
        ]
        ==
        system
    )


    system_counts[
        system
    ] = count


    print(
        f"{system:<22}: "
        f"{count}"
    )


    if count != 500:

        raise ValueError(
            f"{system} does not contain "
            f"exactly 500 responses."
        )


# 7. VERIFY UNIQUE PAIRED RESPONSES

keys = [

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
]


unique_keys = set(
    keys
)


print(
    f"Unique question-system pairs : "
    f"{len(unique_keys)}"
)


if len(unique_keys) != 2000:

    raise ValueError(
        "Duplicate or missing "
        "question-system pairs detected."
    )


print(
    "PASS: Final generation is complete "
    "and uniquely paired."
)


# 8. VERIFY NO GOLD EVALUATION FIELDS

print(
    "\n[3] FINAL LEAKAGE CHECK"
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


leakage_records = []


for result in results:

    overlap = (

        forbidden_fields

        &

        set(
            result.keys()
        )
    )


    if overlap:

        leakage_records.append(
            (
                result[
                    "question_id"
                ],
                result[
                    "system"
                ],
                sorted(
                    overlap
                )
            )
        )


print(
    f"Leakage records : "
    f"{len(leakage_records)}"
)


if leakage_records:

    raise ValueError(
        "Evaluation information detected "
        "inside final generated answers."
    )


print(
    "PASS: No evaluation-only fields "
    "appear in generated answers."
)


# 9. HASH EVERY FINAL TEST ARTIFACT

print(
    "\n[4] HASHING FINAL TEST ARTIFACTS"
)


artifact_hashes = {}


for path in ARTIFACT_PATHS:

    name = relative_path(
        path
    )


    digest = sha256_file(
        path
    )


    artifact_hashes[
        name
    ] = digest


    print(
        f"HASHED : {name}"
    )


print(
    f"\nFiles hashed : "
    f"{len(artifact_hashes)}"
)


# 10. CREATE FINAL TEST FREEZE MANIFEST

manifest = {

    "status":
        "FROZEN_BEFORE_TEST_EVALUATION",

    "stage":
        "final_test_generation_complete",

    "questions":
        500,

    "systems":
        [
            "baseline_llm",
            "basic_rag",
            "advanced_rag",
            "gold_context_control"
        ],

    "expected_generation_responses":
        2000,

    "completed_generation_responses":
        len(
            results
        ),

    "responses_per_system":
        system_counts,

    "test_gold_decisions_used_for_scoring":
        False,

    "test_reference_answers_used_for_scoring":
        False,

    "retrieval_metrics_calculated":
        False,

    "decision_metrics_calculated":
        False,

    "hallucination_metrics_calculated":
        False,

    "rules_after_freeze": [

        (
            "Do not modify retrieval outputs."
        ),

        (
            "Do not modify generation inputs."
        ),

        (
            "Do not regenerate individual answers "
            "based on their content."
        ),

        (
            "Do not change the generator model, "
            "prompt, retrieval configuration, "
            "judge, or metric definitions."
        ),

        (
            "All subsequent test analysis must "
            "use these frozen artifacts."
        )
    ],

    "sha256":
        artifact_hashes
}


save_json(
    manifest,
    OUTPUT_MANIFEST
)


# 11. VERIFY MANIFEST WRITTEN

saved_manifest = load_json(
    OUTPUT_MANIFEST
)


if (
    saved_manifest[
        "status"
    ]
    !=
    "FROZEN_BEFORE_TEST_EVALUATION"
):

    raise RuntimeError(
        "Final test freeze manifest "
        "was not written correctly."
    )


if (
    len(
        saved_manifest[
            "sha256"
        ]
    )
    !=
    len(
        ARTIFACT_PATHS
    )
):

    raise RuntimeError(
        "Artifact hash count mismatch."
    )


# 12. FINAL OUTPUT

print(
    "\n[5] FINAL TEST FREEZE SUMMARY"
)


print(
    "Status       : "
    "FROZEN_BEFORE_TEST_EVALUATION"
)


print(
    f"Questions    : 500"
)


print(
    f"Systems      : 4"
)


print(
    f"Responses    : "
    f"{len(results)}"
)


print(
    f"Files hashed : "
    f"{len(artifact_hashes)}"
)


print(
    "\nTest gold decisions used : NO"
)


print(
    "Decision metrics calculated: NO"
)


print(
    "Hallucination evaluated    : NO"
)


print(
    "\n[6] MANIFEST SAVED"
)


print(
    OUTPUT_MANIFEST
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 37 COMPLETE - "
    "FINAL TEST ARTIFACTS FROZEN"
)


print(
    "=" * 78
)
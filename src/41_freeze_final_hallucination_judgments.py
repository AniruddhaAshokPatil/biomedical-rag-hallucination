import hashlib
import json

from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_hallucination_v3_results.json"
)


SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_hallucination_v3_summary.json"
)


SECTION_40_PATH = (
    PROJECT_ROOT
    / "src"
    / "40_final_test_hallucination_evaluation.py"
)


FROZEN_JUDGE_PATH = (
    PROJECT_ROOT
    / "src"
    / "22_full_hallucination_evaluation.py"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_hallucination_judgment_manifest.json"
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


def relative_path(path):

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/"
    )


# 3. START

print("=" * 78)

print(
    "SECTION 41 - FREEZE FINAL HALLUCINATION JUDGMENTS"
)

print("=" * 78)


# 4. LOAD SECTION 40 OUTPUTS

print(
    "\n[1] VERIFYING SECTION 40 OUTPUTS"
)


results_data = load_json(
    RESULTS_PATH
)


summary_data = load_json(
    SUMMARY_PATH
)


judgments = (
    results_data[
        "judgments"
    ]
)


print(
    f"Question judgments : "
    f"{len(judgments)}"
)


if len(
    judgments
) != 500:

    raise ValueError(
        "Expected exactly 500 "
        "question-level judgments."
    )


system_judgments = sum(

    len(
        record[
            "systems"
        ]
    )

    for record
    in judgments
)


print(
    f"System judgments   : "
    f"{system_judgments}"
)


if system_judgments != 2000:

    raise ValueError(
        "Expected exactly 2,000 "
        "system judgments."
    )


# 5. VERIFY FROZEN JUDGE SETTINGS

print(
    "\n[2] VERIFYING JUDGE CONFIGURATION"
)


expected_settings = {

    "judge_version":
        "v3_final",

    "judge_model":
        "gpt-5.4-2026-03-05",

    "reasoning_effort":
        "low",

    "max_output_tokens":
        3500,

    "random_seed":
        42
}


for field, expected_value in (
    expected_settings.items()
):

    actual_value = (
        results_data.get(
            field
        )
    )


    print(
        f"{field:<20}: "
        f"{actual_value}"
    )


    if (
        actual_value
        !=
        expected_value
    ):

        raise ValueError(
            f"Unexpected {field}: "
            f"{actual_value}"
        )


print(
    "PASS: Frozen V3 judge "
    "configuration confirmed."
)


# 6. VERIFY ALL FOUR SYSTEMS

print(
    "\n[3] VERIFYING PAIRED SYSTEM COVERAGE"
)


SYSTEM_ORDER = [

    "baseline_llm",

    "basic_rag",

    "advanced_rag",

    "gold_context_control"
]


question_ids = set()


for judgment in judgments:

    question_id = str(
        judgment[
            "question_id"
        ]
    )


    if question_id in question_ids:

        raise ValueError(
            f"Duplicate question: "
            f"{question_id}"
        )


    question_ids.add(
        question_id
    )


    systems_present = set(
        judgment[
            "systems"
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
            f"Incomplete system judgments "
            f"for {question_id}."
        )


print(
    f"Unique questions : "
    f"{len(question_ids)}"
)


print(
    "PASS: Every question has all "
    "four paired system judgments."
)


# 7. VALIDATE CLAIM COUNTS

print(
    "\n[4] VALIDATING CLAIM COUNTS"
)


invalid_count_records = 0

invalid_evidence_records = 0


for judgment in judgments:

    for system_name in SYSTEM_ORDER:

        result = (
            judgment[
                "systems"
            ][
                system_name
            ]
        )


        counts = (
            result[
                "claim_counts"
            ]
        )


        calculated_total = (
            counts[
                "supported"
            ]
            +
            counts[
                "unsupported"
            ]
            +
            counts[
                "contradicted"
            ]
        )


        if (
            calculated_total
            !=
            counts[
                "total"
            ]
        ):

            invalid_count_records += 1


        if (
            result[
                "invalid_evidence_labels"
            ]
        ):

            invalid_evidence_records += 1


print(
    f"Invalid claim-count records : "
    f"{invalid_count_records}"
)


print(
    f"Invalid evidence records    : "
    f"{invalid_evidence_records}"
)


if invalid_count_records != 0:

    raise ValueError(
        "Claim-count validation failed."
    )


if invalid_evidence_records != 0:

    raise ValueError(
        "Evidence-label validation failed."
    )


print(
    "PASS: Claim counts and evidence "
    "labels are internally consistent."
)


# 8. HASH FINAL JUDGE ARTIFACTS

print(
    "\n[5] HASHING FINAL HALLUCINATION ARTIFACTS"
)


files_to_hash = [

    RESULTS_PATH,

    SUMMARY_PATH,

    SECTION_40_PATH,

    FROZEN_JUDGE_PATH
]


hashes = {}


for path in files_to_hash:

    digest = sha256_file(
        path
    )


    name = relative_path(
        path
    )


    hashes[
        name
    ] = digest


    print(
        f"HASHED : {name}"
    )


# 9. CREATE FREEZE MANIFEST

manifest = {

    "status":
        "FROZEN_BEFORE_HALLUCINATION_STATISTICS",

    "split":
        "test",

    "questions":
        500,

    "system_judgments":
        2000,

    "judge_version":
        "v3_final",

    "judge_model":
        "gpt-5.4-2026-03-05",

    "reasoning_effort":
        "low",

    "max_output_tokens":
        3500,

    "random_seed":
        42,

    "evidence_design":
        (
            "same gold evidence "
            "for all four systems"
        ),

    "blinding":
        (
            "deterministic anonymous "
            "A/B/C/D"
        ),

    "metric_definition": {

        "hallucination":
            (
                "(unsupported + contradicted) "
                "/ scored factual claims"
            ),

        "groundedness":
            (
                "supported / "
                "scored factual claims"
            )
    },

    "statistical_tests_run":
        False,

    "human_calibration_applied":
        False,

    "rules_after_freeze": [

        (
            "Do not rerun individual "
            "judge calls based on results."
        ),

        (
            "Do not alter claim labels "
            "or exclusions."
        ),

        (
            "Do not change the V3 prompt, "
            "schema, judge model, or "
            "metric definition."
        ),

        (
            "Subsequent statistical analysis "
            "must use these frozen judgments."
        )
    ],

    "sha256":
        hashes
}


save_json(
    manifest,
    OUTPUT_PATH
)


# 10. FINAL SUMMARY

print(
    "\n[6] HALLUCINATION FREEZE SUMMARY"
)


print(
    "Status : "
    "FROZEN_BEFORE_HALLUCINATION_STATISTICS"
)


print(
    "Questions         : 500"
)


print(
    "System judgments  : 2000"
)


print(
    f"Artifacts hashed  : "
    f"{len(hashes)}"
)


print(
    "Statistical tests : NO"
)


print(
    "Calibration       : NO"
)


print(
    "\n[7] MANIFEST SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 41 COMPLETE - "
    "HALLUCINATION JUDGMENTS FROZEN"
)


print(
    "=" * 78
)
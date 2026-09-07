import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_protocol_manifest.json"
)


# 2. HELPERS

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


# 3. FROZEN EXPERIMENTAL CONFIGURATION

protocol = {

    "status":
        "FROZEN_BEFORE_FINAL_TEST",

    "research_question":
        (
            "To what extent does RAG reduce hallucinations "
            "in biomedical/healthcare question answering "
            "compared with a standalone LLM, and does "
            "enhanced RAG provide additional improvements "
            "over basic RAG?"
        ),

    "systems": {

        "baseline_llm": {
            "retrieval":
                None
        },

        "basic_rag": {
            "retrieval":
                "Dense retrieval using all-MiniLM-L6-v2",

            "top_k":
                5
        },

        "advanced_rag": {
            "retrieval":
                (
                    "BM25 + dense weighted reciprocal rank "
                    "fusion followed by cross-encoder reranking"
                ),

            "bm25_weight":
                0.75,

            "dense_weight":
                1.0,

            "rrf_k":
                60,

            "reranker":
                "cross-encoder/ms-marco-MiniLM-L6-v2",

            "final_top_k":
                5
        },

        "gold_context_control": {
            "retrieval":
                "Gold attached source passages"
        }
    },

    "generator": {

        "model":
            "gpt-5.4-mini-2026-03-17",

        "reasoning":
            "none",

        "max_output_tokens":
            300,

        "verbosity":
            "low",

        "structured_output":
            True,

        "same_generator_for_all_systems":
            True
    },

    "answer_format": {

        "decision_labels": [
            "yes",
            "no",
            "maybe"
        ],

        "fields": [
            "decision",
            "answer",
            "citations"
        ]
    },

    "hallucination_evaluator": {

        "version":
            "V3",

        "model":
            "gpt-5.4-2026-03-05",

        "reasoning":
            "low",

        "max_output_tokens":
            3500,

        "grouped_system_evaluation":
            True,

        "system_identity_blinded":
            True,

        "common_gold_evidence":
            True,

        "definition":
            (
                "Unsupported or contradicted externally "
                "verifiable biomedical/research factual "
                "claims divided by total scored factual claims."
            ),

        "excluded_from_denominator": [
            "pure abstention",
            "uncertainty statements",
            "evidence-availability commentary",
            "citation/format commentary",
            "other non-biomedical meta statements"
        ]
    },

    "primary_metrics": [

        "decision accuracy",
        "decision macro-F1",
        "hallucination rate",
        "groundedness",
        "contradiction rate",
        "retrieval Source Hit@k",
        "retrieval MRR@10",
        "Gold-context Recall@k"
    ],

    "statistical_plan": {

        "paired_questions":
            True,

        "decision_test":
            "McNemar",

        "macro_f1":
            "paired question bootstrap",

        "hallucination":
            (
                "paired question-level inference with "
                "question-cluster bootstrap for claim rates"
            ),

        "multiple_comparison_correction":
            "Holm",

        "bootstrap_iterations":
            10000
    },

    "human_validation_summary": {

        "first_claim_validation": {
            "binary_accuracy":
                0.7000,

            "cohens_kappa":
                0.4375,

            "interpretation":
                (
                    "V3 is highly sensitive but produces "
                    "substantial false-positive hallucination calls."
                )
        },

        "system_balanced_claim_validation": {

            "binary_accuracy":
                0.7000,

            "cohens_kappa":
                0.4000,

            "hallucination_recall":
                0.9706,

            "false_positive_rate":
                0.3730,

            "system_fpr_range":
                0.0491,

            "interpretation":
                (
                    "False-positive behaviour is reasonably "
                    "consistent across systems."
                )
        },

        "extraction_validation": {

            "accuracy":
                0.7000,

            "macro_f1":
                0.6905,

            "cohens_kappa":
                0.4000,

            "interpretation":
                (
                    "Extraction/exclusion is imperfect. "
                    "Basic and Advanced show similar error "
                    "behaviour; Gold has greater exclusion error."
                )
        },

        "human_calibration":
            "Sensitivity analysis only"
    },

    "analysis_rules": {

        "automatic_v3_rates_are_primary":
            True,

        "human_calibrated_rates_are_sensitivity_analysis":
            True,

        "do_not_tune_after_freeze":
            True,

        "do_not_select_system_using_test_results":
            True,

        "gold_control_is_not_expected_to_be_perfect":
            True,

        "advanced_rag_not_assumed_superior":
            True
    }
}


# 4. FILES TO HASH

files_to_hash = []


# Core processed data
for filename in [
    "development_questions.json",
    "test_questions.json",
    "retrieval_corpus.json",
    "evaluation_ground_truth.json"
]:

    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / filename
    )

    if path.exists():

        files_to_hash.append(
            path
        )


# Important source scripts completed before freeze
script_numbers = set(
    range(
        4,
        32
    )
)


src_dir = (
    PROJECT_ROOT
    / "src"
)


for path in sorted(
    src_dir.glob(
        "*.py"
    )
):

    name = path.name

    prefix = (
        name.split(
            "_",
            1
        )[0]
    )


    try:

        number = int(
            prefix
        )

    except ValueError:

        continue


    if number in script_numbers:

        files_to_hash.append(
            path
        )


# 5. CREATE HASH SNAPSHOT

print("=" * 78)

print(
    "SECTION 32 - FREEZE FINAL EXPERIMENTAL PROTOCOL"
)

print("=" * 78)


print(
    "\n[1] HASHING FROZEN FILES"
)


file_hashes = {}


for path in files_to_hash:

    relative = relative_path(
        path
    )

    digest = sha256_file(
        path
    )


    file_hashes[
        relative
    ] = digest


    print(
        f"{relative}"
    )

    print(
        f"  {digest}"
    )


# 6. FINAL MANIFEST

manifest = {

    "freeze_timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "protocol":
        protocol,

    "file_hashes":
        file_hashes,

    "number_of_hashed_files":
        len(
            file_hashes
        )
}


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=2,
        ensure_ascii=False
    )


# 7. OUTPUT

print(
    "\n[2] FREEZE SUMMARY"
)


print(
    f"Files hashed : "
    f"{len(file_hashes)}"
)


print(
    "Generator    : "
    "gpt-5.4-mini-2026-03-17"
)


print(
    "Judge        : "
    "gpt-5.4-2026-03-05 / V3"
)


print(
    "Basic RAG    : "
    "Dense top-5"
)


print(
    "Advanced RAG : "
    "Weighted hybrid RRF + reranker top-5"
)


print(
    "Human validation completed : YES"
)


print(
    "Final test opened          : NO"
)


print(
    "\n[3] MANIFEST SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 32 COMPLETE - EXPERIMENT FROZEN"
)


print(
    "=" * 78
)
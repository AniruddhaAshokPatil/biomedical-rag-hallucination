import hashlib
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)

TEST_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_questions.json"
)

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_corpus.json"
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
        return json.load(file)


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


def normalise_relative(path):

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/"
    )


# 3. LOAD FREEZE MANIFEST

print("=" * 78)

print(
    "SECTION 33 - FINAL TEST READINESS CHECK"
)

print("=" * 78)


manifest = load_json(
    MANIFEST_PATH
)


print(
    "\n[1] FREEZE MANIFEST"
)


print(
    f"Status : "
    f"{manifest['protocol']['status']}"
)


print(
    f"Frozen files : "
    f"{manifest['number_of_hashed_files']}"
)


if (
    manifest[
        "protocol"
    ][
        "status"
    ]
    !=
    "FROZEN_BEFORE_FINAL_TEST"
):

    raise ValueError(
        "Manifest does not indicate "
        "FROZEN_BEFORE_FINAL_TEST."
    )


# 4. VERIFY ALL FROZEN FILE HASHES

print(
    "\n[2] FROZEN FILE INTEGRITY"
)


changed_files = []

missing_files = []

verified_files = 0


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

    else:

        verified_files += 1


print(
    f"Verified unchanged : "
    f"{verified_files}"
)


print(
    f"Changed files      : "
    f"{len(changed_files)}"
)


print(
    f"Missing files      : "
    f"{len(missing_files)}"
)


if changed_files:

    print(
        "\nCHANGED:"
    )

    for name in changed_files:
        print(
            f"  {name}"
        )


if missing_files:

    print(
        "\nMISSING:"
    )

    for name in missing_files:
        print(
            f"  {name}"
        )


if changed_files or missing_files:

    raise RuntimeError(
        "Frozen protocol integrity check failed. "
        "Do not run the final test yet."
    )


print(
    "PASS: Frozen files match the "
    "Section 32 manifest."
)


# 5. CHECK TEST QUESTION FILE

print(
    "\n[3] FINAL TEST QUESTION FILE"
)


test_questions = load_json(
    TEST_QUESTIONS_PATH
)


print(
    f"Test questions : "
    f"{len(test_questions)}"
)


if len(test_questions) != 500:

    raise ValueError(
        f"Expected 500 test questions, "
        f"found {len(test_questions)}."
    )


question_ids = [

    str(
        record[
            "question_id"
        ]
    )

    for record in test_questions
]


if len(
    set(
        question_ids
    )
) != 500:

    raise ValueError(
        "Duplicate test question IDs found."
    )


allowed_fields = {
    "question_id",
    "question"
}


unexpected_fields = set()


for record in test_questions:

    unexpected_fields.update(
        set(
            record.keys()
        )
        -
        allowed_fields
    )


print(
    f"Unique IDs     : "
    f"{len(set(question_ids))}"
)


print(
    f"Unexpected fields : "
    f"{sorted(unexpected_fields)}"
)


if unexpected_fields:

    raise ValueError(
        "Test question file contains fields "
        "beyond question_id/question. "
        "Stop before final evaluation."
    )


print(
    "PASS: Test file contains only IDs "
    "and questions."
)


# 6. CHECK RETRIEVAL CORPUS

print(
    "\n[4] RETRIEVAL CORPUS"
)


corpus = load_json(
    CORPUS_PATH
)


print(
    f"Corpus documents : "
    f"{len(corpus)}"
)


if len(corpus) != 3358:

    raise ValueError(
        f"Expected 3358 corpus documents, "
        f"found {len(corpus)}."
    )


document_ids = [

    str(
        record[
            "document_id"
        ]
    )

    for record in corpus
]


print(
    f"Unique document IDs : "
    f"{len(set(document_ids))}"
)


if (
    len(
        set(
            document_ids
        )
    )
    !=
    len(corpus)
):

    raise ValueError(
        "Duplicate corpus document IDs found."
    )


print(
    "PASS: Retrieval corpus is intact."
)


# 7. LOCATE SAVED RETRIEVAL CONFIGURATION FILES

print(
    "\n[5] RETRIEVAL CONFIGURATION CANDIDATES"
)


candidate_files = []


search_roots = [
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "data" / "processed"
]


for search_root in search_roots:

    if not search_root.exists():
        continue


    for path in search_root.rglob(
        "*.json"
    ):

        filename_lower = (
            path.name.lower()
        )


        if any(
            keyword in filename_lower
            for keyword in [
                "config",
                "retrieval",
                "freeze"
            ]
        ):

            # Do not list ground-truth files.
            if (
                "ground_truth"
                in filename_lower
            ):
                continue

            candidate_files.append(
                path
            )


candidate_files = sorted(
    set(
        candidate_files
    )
)


if not candidate_files:

    print(
        "No candidate configuration "
        "JSON files found."
    )


for path in candidate_files:

    print(
        normalise_relative(
            path
        )
    )


# 8. PRINT FROZEN RETRIEVAL SETTINGS FROM MANIFEST

print(
    "\n[6] FROZEN RETRIEVAL SETTINGS"
)


systems = (
    manifest[
        "protocol"
    ][
        "systems"
    ]
)


basic = systems[
    "basic_rag"
]

advanced = systems[
    "advanced_rag"
]


print(
    "Basic RAG"
)

print(
    f"  Retrieval : "
    f"{basic['retrieval']}"
)

print(
    f"  Top-k     : "
    f"{basic['top_k']}"
)


print(
    "\nAdvanced RAG"
)

print(
    f"  Retrieval    : "
    f"{advanced['retrieval']}"
)

print(
    f"  BM25 weight  : "
    f"{advanced['bm25_weight']}"
)

print(
    f"  Dense weight : "
    f"{advanced['dense_weight']}"
)

print(
    f"  RRF k        : "
    f"{advanced['rrf_k']}"
)

print(
    f"  Reranker     : "
    f"{advanced['reranker']}"
)

print(
    f"  Final top-k  : "
    f"{advanced['final_top_k']}"
)


# 9. FINAL READINESS RESULT

print(
    "\n[7] FINAL TEST READINESS"
)


print(
    "Frozen file integrity : PASS"
)

print(
    "500 test questions    : PASS"
)

print(
    "No test labels loaded : PASS"
)

print(
    "Retrieval corpus      : PASS"
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 33 COMPLETE - READY FOR FINAL TEST RETRIEVAL"
)

print(
    "=" * 78
)
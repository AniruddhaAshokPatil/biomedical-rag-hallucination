import json
from pathlib import Path
from collections import Counter


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "ori_pqal.json"
)

TEST_GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "test_ground_truth.json"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 2. OUTPUT PATHS

DEVELOPMENT_OUTPUT = (
    PROCESSED_DIR / "development_questions.json"
)

TEST_OUTPUT = (
    PROCESSED_DIR / "test_questions.json"
)

CORPUS_OUTPUT = (
    PROCESSED_DIR / "retrieval_corpus.json"
)

EVALUATION_OUTPUT = (
    PROCESSED_DIR / "evaluation_ground_truth.json"
)


# 3. LOAD / SAVE HELPERS

def load_json(file_path: Path):

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(
    data,
    file_path: Path
):

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# 4. LOAD ORIGINAL DATA

original_data = load_json(
    ORIGINAL_DATA_PATH
)

test_ground_truth = load_json(
    TEST_GROUND_TRUTH_PATH
)


all_ids = set(
    original_data.keys()
)

test_ids = set(
    test_ground_truth.keys()
)

development_ids = (
    all_ids - test_ids
)


print("=" * 75)
print("SECTION 4 - PREPARING EXPERIMENTAL DATA")
print("=" * 75)

print(
    f"\nOriginal records : "
    f"{len(original_data)}"
)

print(
    f"Development IDs  : "
    f"{len(development_ids)}"
)

print(
    f"Test IDs         : "
    f"{len(test_ids)}"
)


# 5. PREPARE DEVELOPMENT QUESTIONS

development_questions = []


for record_id in sorted(
    development_ids
):

    record = original_data[
        record_id
    ]

    development_questions.append(
        {
            "question_id": record_id,
            "question": record[
                "QUESTION"
            ]
        }
    )


# 6. PREPARE TEST QUESTIONS

test_questions = []


for record_id in sorted(
    test_ids
):

    record = original_data[
        record_id
    ]

    test_questions.append(
        {
            "question_id": record_id,
            "question": record[
                "QUESTION"
            ]
        }
    )



# 7. BUILD RETRIEVAL CORPUS

# Each CONTEXT becomes one retrievable document.
# Example:
# record 123
#     context 0
#     context 1
#     context 2
# becomes:
# 123_context_0
# 123_context_1
# 123_context_2
# NO reference answer or decision is included.

retrieval_corpus = []


for record_id in sorted(
    original_data.keys()
):

    record = original_data[
        record_id
    ]

    contexts = record.get(
        "CONTEXTS",
        []
    )

    labels = record.get(
        "LABELS",
        []
    )

    meshes = record.get(
        "MESHES",
        []
    )

    year = record.get(
        "YEAR"
    )


    for context_index, context in enumerate(
        contexts
    ):

        if context_index < len(labels):

            section = labels[
                context_index
            ]

        else:

            section = None


        document_id = (
            f"{record_id}"
            f"_context_"
            f"{context_index}"
        )


        document = {

            # Unique retrievable passage
            "document_id":
                document_id,

            # Used internally for
            # retrieval evaluation
            "source_record_id":
                record_id,

            # Actual biomedical text
            "text":
                context,

            # Metadata only
            "section":
                section,

            "year":
                year,

            "meshes":
                meshes
        }


        retrieval_corpus.append(
            document
        )


# 8. CREATE EVALUATION GROUND TRUTH

# THIS FILE MUST NEVER BE GIVEN TO THE LLM.
# It contains:
#  gold decision
#  reference answer
#  IDs of the known source passages

evaluation_ground_truth = {}


for record_id in sorted(
    original_data.keys()
):

    record = original_data[
        record_id
    ]


    gold_context_ids = []

    for context_index in range(
        len(record["CONTEXTS"])
    ):

        document_id = (
            f"{record_id}"
            f"_context_"
            f"{context_index}"
        )

        gold_context_ids.append(
            document_id
        )


    split = (
        "test"
        if record_id in test_ids
        else "development"
    )


    evaluation_ground_truth[
        record_id
    ] = {

        "split":
            split,

        "gold_decision":
            record[
                "final_decision"
            ],

        "reference_answer":
            record[
                "LONG_ANSWER"
            ],

        "gold_context_ids":
            gold_context_ids
    }

# 9. SAVE FILES

save_json(
    development_questions,
    DEVELOPMENT_OUTPUT
)

save_json(
    test_questions,
    TEST_OUTPUT
)

save_json(
    retrieval_corpus,
    CORPUS_OUTPUT
)

save_json(
    evaluation_ground_truth,
    EVALUATION_OUTPUT
)


# 10. VALIDATION CHECKS

print("\n[1] OUTPUT COUNTS")

print(
    f"Development questions : "
    f"{len(development_questions)}"
)

print(
    f"Test questions        : "
    f"{len(test_questions)}"
)

print(
    f"Retrieval documents   : "
    f"{len(retrieval_corpus)}"
)

print(
    f"Evaluation records    : "
    f"{len(evaluation_ground_truth)}"
)


# 11. CHECK QUESTION OVERLAP

development_question_ids = {
    item["question_id"]
    for item in development_questions
}

test_question_ids = {
    item["question_id"]
    for item in test_questions
}

overlap = (
    development_question_ids
    &
    test_question_ids
)


print("\n[2] SPLIT LEAKAGE CHECK")

print(
    f"Question ID overlap between "
    f"development/test: "
    f"{len(overlap)}"
)

if len(overlap) == 0:

    print(
        "PASS: Development and test "
        "question sets are separate."
    )

else:

    print(
        "WARNING: Split overlap detected."
    )


# 12. CHECK RETRIEVAL CORPUS FOR FORBIDDEN FIELDS

print("\n[3] RETRIEVAL CORPUS LEAKAGE CHECK")


forbidden_fields = {
    "LONG_ANSWER",
    "final_decision",
    "reasoning_required_pred",
    "reasoning_free_pred",
    "gold_decision",
    "reference_answer"
}


found_forbidden_fields = set()


for document in retrieval_corpus:

    for field in forbidden_fields:

        if field in document:

            found_forbidden_fields.add(
                field
            )


if len(found_forbidden_fields) == 0:

    print(
        "PASS: No answer/decision fields "
        "exist in retrieval corpus."
    )

else:

    print(
        "WARNING: Forbidden fields found:"
    )

    print(
        found_forbidden_fields
    )


# 13. SECTION DISTRIBUTION

print("\n[4] CORPUS SECTION DISTRIBUTION")


section_counts = Counter(

    document["section"]

    for document in retrieval_corpus
)


for section, count in section_counts.most_common():

    print(
        f"{str(section):<25}"
        f"{count}"
    )


# 14. SPLIT DISTRIBUTION IN CORPUS

development_documents = 0
test_documents = 0


for document in retrieval_corpus:

    source_id = document[
        "source_record_id"
    ]

    if source_id in test_ids:

        test_documents += 1

    else:

        development_documents += 1


print("\n[5] CORPUS SOURCE DISTRIBUTION")

print(
    f"Development-source passages : "
    f"{development_documents}"
)

print(
    f"Test-source passages        : "
    f"{test_documents}"
)


# 15. DISPLAY SAMPLE CORPUS DOCUMENT

print("\n[6] SAMPLE RETRIEVAL DOCUMENT")

sample_document = retrieval_corpus[0]


print(
    json.dumps(
        sample_document,
        indent=2,
        ensure_ascii=False
    )
)


# 16. DISPLAY SAMPLE QUESTION

print("\n[7] SAMPLE DEVELOPMENT QUESTION")

print(
    json.dumps(
        development_questions[0],
        indent=2,
        ensure_ascii=False
    )
)

# 17. FINAL SUMMARY

print("\n" + "=" * 75)
print("SECTION 4 COMPLETE")
print("=" * 75)

print(f"""
Files created:

{DEVELOPMENT_OUTPUT.name}
{TEST_OUTPUT.name}
{CORPUS_OUTPUT.name}
{EVALUATION_OUTPUT.name}

""")
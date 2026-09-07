import json
import random
import re
from pathlib import Path
from collections import defaultdict


# 1. FILE PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATA_PATH = PROJECT_ROOT / "data" / "ori_pqal.json"
TEST_GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "test_ground_truth.json"


# 2. LOAD JSON

def load_json(file_path: Path):
    """Load JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


original_data = load_json(ORIGINAL_DATA_PATH)
test_ground_truth = load_json(TEST_GROUND_TRUTH_PATH)


# 3. CREATE DEVELOPMENT SET

all_ids = set(original_data.keys())
test_ids = set(test_ground_truth.keys())

development_ids = sorted(all_ids - test_ids)

development_data = {
    record_id: original_data[record_id]
    for record_id in development_ids
}


print("=" * 75)
print("SECTION 3 - RECORD STRUCTURE AND EVIDENCE INSPECTION")
print("=" * 75)

print(f"\nDevelopment records available: {len(development_data)}")


# 4. DEFINE INPUT VS GROUND-TRUTH FIELDS

print("\n[1] FIELD ROLES")

retrieval_fields = [
    "CONTEXTS",
    "LABELS",
    "MESHES",
    "YEAR"
]

evaluation_fields = [
    "LONG_ANSWER",
    "final_decision"
]

excluded_prediction_fields = [
    "reasoning_required_pred",
    "reasoning_free_pred"
]

print("\nPotential evidence / metadata:")
for field in retrieval_fields:
    print(f"  - {field}")

print("\nGround truth - DO NOT expose to model:")
for field in evaluation_fields:
    print(f"  - {field}")

print("\nExisting prediction fields - DO NOT expose to model:")
for field in excluded_prediction_fields:
    print(f"  - {field}")


# 5. CHECK CONTEXT <-> LABEL ALIGNMENT

print("\n[2] CONTEXT / SECTION-LABEL ALIGNMENT")

alignment_errors = []

for record_id, record in original_data.items():

    contexts = record.get("CONTEXTS", [])
    labels = record.get("LABELS", [])

    if len(contexts) != len(labels):
        alignment_errors.append(
            {
                "id": record_id,
                "contexts": len(contexts),
                "labels": len(labels)
            }
        )


print(
    f"Records where number of CONTEXTS != number of LABELS: "
    f"{len(alignment_errors)}"
)

if len(alignment_errors) == 0:
    print("PASS: Every context has a corresponding section label.")
else:
    print("\nFirst five alignment errors:")

    for error in alignment_errors[:5]:
        print(error)


# 6. BASIC TEXT LENGTH STATISTICS

print("\n[3] TEXT LENGTH STATISTICS")

question_word_counts = []
context_word_counts = []
answer_word_counts = []

for record in original_data.values():

    question_word_counts.append(
        len(record["QUESTION"].split())
    )

    for context in record["CONTEXTS"]:
        context_word_counts.append(
            len(context.split())
        )

    answer_word_counts.append(
        len(record["LONG_ANSWER"].split())
    )


def average(values):
    return sum(values) / len(values)


print(
    f"Average question length : "
    f"{average(question_word_counts):.1f} words"
)

print(
    f"Average context length  : "
    f"{average(context_word_counts):.1f} words"
)

print(
    f"Average reference answer: "
    f"{average(answer_word_counts):.1f} words"
)

print(
    f"Total evidence passages : "
    f"{len(context_word_counts)}"
)


# 7. CHECK WHETHER LONG ANSWER IS ALREADY INSIDE CONTEXT

print("\n[4] LONG ANSWER / CONTEXT OVERLAP")

exact_answer_in_context = 0


def normalize_text(text):
    """
    Lowercase and remove extra whitespace/punctuation
    for basic text comparison.
    """

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


for record in original_data.values():

    long_answer = normalize_text(record["LONG_ANSWER"])

    combined_context = normalize_text(
        " ".join(record["CONTEXTS"])
    )

    if long_answer and long_answer in combined_context:
        exact_answer_in_context += 1


percentage = (
    exact_answer_in_context / len(original_data)
) * 100

print(
    f"Reference answer appears verbatim inside contexts: "
    f"{exact_answer_in_context}/{len(original_data)} "
    f"({percentage:.1f}%)"
)

print(
    "\nInterpretation:"
)

print(
    "A low percentage suggests LONG_ANSWER is generally a "
    "separate conclusion/reference answer rather than simply "
    "copied context."
)


# 8. ORGANISE DEVELOPMENT RECORDS BY DECISION

print("\n[5] DEVELOPMENT RECORDS BY DECISION")

records_by_decision = defaultdict(list)

for record_id, record in development_data.items():

    decision = record["final_decision"]

    records_by_decision[decision].append(record_id)


for decision in sorted(records_by_decision):

    print(
        f"{decision:>10}: "
        f"{len(records_by_decision[decision])} records"
    )


# 9. SAMPLE DEVELOPMENT RECORDS

print("\n" + "=" * 75)
print("MANUAL RECORD INSPECTION")
print("=" * 75)

random.seed(42)

for decision in ["yes", "no", "maybe"]:

    available_ids = records_by_decision.get(decision, [])

    sample_size = min(2, len(available_ids))

    sampled_ids = random.sample(
        available_ids,
        sample_size
    )

    for record_id in sampled_ids:

        record = development_data[record_id]

        print("\n" + "-" * 75)

        print(
            f"ID: {record_id} | "
            f"Decision: {record['final_decision']} | "
            f"Year: {record.get('YEAR')}"
        )

        print(
            f"MESH terms: "
            f"{', '.join(record.get('MESHES', []))}"
        )

        print("\nQUESTION:")
        print(record["QUESTION"])

        print("\nEVIDENCE CONTEXTS:")

        for index, context in enumerate(
            record["CONTEXTS"]
        ):

            if index < len(record.get("LABELS", [])):
                section_label = record["LABELS"][index]
            else:
                section_label = "UNKNOWN"

            print(
                f"\nContext {index + 1} "
                f"[{section_label}]"
            )

            print(context)

        print("\nREFERENCE LONG ANSWER:")
        print(record["LONG_ANSWER"])

        print("\nREFERENCE DECISION:")
        print(record["final_decision"])

# 10. FINAL MESSAGE

print("\n" + "=" * 75)
print("SECTION 3 COMPLETE")
print("=" * 75)
import json
from pathlib import Path
from collections import Counter


# 1. FILE PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATA_PATH = PROJECT_ROOT / "data" / "ori_pqal.json"
TEST_GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "test_ground_truth.json"


# 2. LOAD JSON

def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


original_data = load_json(ORIGINAL_DATA_PATH)
test_ground_truth = load_json(TEST_GROUND_TRUTH_PATH)


print("=" * 70)
print("SECTION 2 - DATASET VALIDATION AND ANALYSIS")
print("=" * 70)


# 3. BASIC DATASET SIZE

print("\n[1] DATASET SIZE")

print(f"Total records in original dataset : {len(original_data)}")
print(f"Records in test ground truth      : {len(test_ground_truth)}")


# 4. CHECK TEST IDS

print("\n[2] TEST ID VALIDATION")

original_ids = set(original_data.keys())
test_ids = set(test_ground_truth.keys())

missing_test_ids = test_ids - original_ids

print(f"Test IDs found in original dataset : {len(test_ids & original_ids)}")
print(f"Test IDs missing from original data: {len(missing_test_ids)}")

if len(missing_test_ids) == 0:
    print("PASS: Every test ID exists in the original dataset.")
else:
    print("WARNING: Some test IDs are missing.")


# 5. CREATE DEVELOPMENT AND TEST IDS

development_ids = original_ids - test_ids

print("\n[3] PROPOSED DATA SPLIT")

print(f"Development records : {len(development_ids)}")
print(f"Test records        : {len(test_ids)}")

print("\nImportant:")
print("Development set will be used for system development/tuning.")
print("Test set will be reserved for final evaluation.")


# 6. FINAL DECISION DISTRIBUTION

print("\n[4] LABEL DISTRIBUTION - FULL DATASET")

full_labels = [
    record.get("final_decision")
    for record in original_data.values()
]

full_label_counts = Counter(full_labels)

for label, count in sorted(full_label_counts.items(), key=lambda x: str(x[0])):
    percentage = (count / len(original_data)) * 100
    print(f"{str(label):>10}: {count:4d} ({percentage:5.1f}%)")


# 7. DEVELOPMENT LABEL DISTRIBUTION

print("\n[5] LABEL DISTRIBUTION - DEVELOPMENT SET")

development_labels = [
    original_data[record_id].get("final_decision")
    for record_id in development_ids
]

development_label_counts = Counter(development_labels)

for label, count in sorted(
    development_label_counts.items(),
    key=lambda x: str(x[0])
):
    percentage = (count / len(development_ids)) * 100
    print(f"{str(label):>10}: {count:4d} ({percentage:5.1f}%)")


# 8. TEST LABEL DISTRIBUTION

print("\n[6] LABEL DISTRIBUTION - TEST SET")

test_label_counts = Counter(test_ground_truth.values())

for label, count in sorted(test_label_counts.items(), key=lambda x: str(x[0])):
    percentage = (count / len(test_ground_truth)) * 100
    print(f"{str(label):>10}: {count:4d} ({percentage:5.1f}%)")


# 9. VERIFY TEST LABELS

print("\n[7] VERIFY TEST LABELS")

label_mismatches = []

for record_id, ground_truth_label in test_ground_truth.items():

    dataset_label = original_data[record_id].get("final_decision")

    if dataset_label != ground_truth_label:
        label_mismatches.append(
            (
                record_id,
                dataset_label,
                ground_truth_label
            )
        )


print(f"Number of label mismatches: {len(label_mismatches)}")

if len(label_mismatches) == 0:
    print("PASS: Test ground-truth labels match final_decision.")
else:
    print("WARNING: Label mismatches detected.")

    for mismatch in label_mismatches[:5]:
        print(mismatch)


# 10. CHECK REQUIRED FIELDS

print("\n[8] REQUIRED FIELD CHECK")

required_fields = [
    "QUESTION",
    "CONTEXTS",
    "final_decision",
    "LONG_ANSWER"
]

missing_field_counts = Counter()

for record in original_data.values():

    for field in required_fields:

        if field not in record:
            missing_field_counts[field] += 1


for field in required_fields:
    print(
        f"{field:<20}: "
        f"{missing_field_counts[field]} records missing field"
    )


# 11. CHECK EMPTY VALUES

print("\n[9] EMPTY VALUE CHECK")

empty_questions = 0
empty_contexts = 0
empty_answers = 0
empty_decisions = 0

for record in original_data.values():

    if not record.get("QUESTION"):
        empty_questions += 1

    if not record.get("CONTEXTS"):
        empty_contexts += 1

    if not record.get("LONG_ANSWER"):
        empty_answers += 1

    if not record.get("final_decision"):
        empty_decisions += 1


print(f"Empty questions       : {empty_questions}")
print(f"Empty contexts        : {empty_contexts}")
print(f"Empty long answers    : {empty_answers}")
print(f"Empty final decisions : {empty_decisions}")


# 12. CONTEXT STATISTICS

print("\n[10] CONTEXT STATISTICS")

context_counts = [
    len(record.get("CONTEXTS", []))
    for record in original_data.values()
]

print(f"Minimum contexts per question : {min(context_counts)}")
print(f"Maximum contexts per question : {max(context_counts)}")
print(
    f"Average contexts per question : "
    f"{sum(context_counts) / len(context_counts):.2f}"
)


context_distribution = Counter(context_counts)

print("\nContext-count distribution:")

for number_of_contexts in sorted(context_distribution):

    print(
        f"{number_of_contexts} contexts : "
        f"{context_distribution[number_of_contexts]} questions"
    )


# 13. DUPLICATE QUESTION CHECK

print("\n[11] DUPLICATE QUESTION CHECK")

questions = [
    record.get("QUESTION", "").strip().lower()
    for record in original_data.values()
]

question_counts = Counter(questions)

duplicate_questions = {
    question: count
    for question, count in question_counts.items()
    if count > 1
}

print(f"Number of duplicated question texts: {len(duplicate_questions)}")

if duplicate_questions:

    print("\nExample duplicates:")

    for question, count in list(duplicate_questions.items())[:5]:
        print(f"{count}x -> {question}")

# 14. YEAR ANALYSIS

print("\n[12] YEAR INFORMATION")

years = [
    record.get("YEAR")
    for record in original_data.values()
    if record.get("YEAR")
]

missing_years = sum(
    1
    for record in original_data.values()
    if not record.get("YEAR")
)

print(f"Records with year    : {len(years)}")
print(f"Records missing year : {missing_years}")

if years:

    numeric_years = []

    for year in years:
        try:
            numeric_years.append(int(year))
        except (ValueError, TypeError):
            pass

    if numeric_years:
        print(f"Earliest year        : {min(numeric_years)}")
        print(f"Latest year          : {max(numeric_years)}")


# 15. SUMMARY

print("\n" + "=" * 70)
print("DATASET VALIDATION COMPLETE")
print("=" * 70)

print(f"""
Full dataset : {len(original_data)}
Development  : {len(development_ids)}
Test         : {len(test_ids)}

Missing test IDs       : {len(missing_test_ids)}
Label mismatches       : {len(label_mismatches)}
Duplicate questions    : {len(duplicate_questions)}
""")
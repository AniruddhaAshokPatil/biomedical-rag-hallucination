import json
from pathlib import Path

# 1. Define file paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATA_PATH = PROJECT_ROOT / "data" / "ori_pqal.json"
TEST_GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "test_ground_truth.json"


# 2. Load JSON files

def load_json(file_path: Path):
    """Load a JSON file and return its contents."""
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


original_data = load_json(ORIGINAL_DATA_PATH)
test_ground_truth = load_json(TEST_GROUND_TRUTH_PATH)


# 3. Basic inspection

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

print(f"\nOriginal dataset type: {type(original_data)}")
print(f"Original dataset size: {len(original_data)}")

print(f"\nGround-truth type: {type(test_ground_truth)}")
print(f"Ground-truth size: {len(test_ground_truth)}")


# 4. Display one example

first_id = next(iter(original_data))
first_record = original_data[first_id]

print("\n" + "=" * 60)
print("FIRST RECORD")
print("=" * 60)

print(f"\nRecord ID: {first_id}")

print("\nAvailable fields:")
for field in first_record.keys():
    print(f"  - {field}")


 

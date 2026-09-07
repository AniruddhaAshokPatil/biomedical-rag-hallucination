import json
from pathlib import Path
from collections import Counter


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "retrieval"
)

BM25_RESULTS_PATH = (
    RESULTS_DIR / "bm25_dev_results.json"
)

DENSE_RESULTS_PATH = (
    RESULTS_DIR / "dense_dev_results.json"
)

OUTPUT_PATH = (
    RESULTS_DIR / "retrieval_comparison_dev.json"
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


def source_hit(result, k):
    """
    Returns True if at least one correct-source
    passage appears within top-k.
    """

    return any(

        document["is_gold_source"]

        for document
        in result["retrieved_documents"][:k]
    )


# 3. LOAD RESULTS

print("=" * 75)
print("SECTION 7 - RETRIEVAL COMPLEMENTARITY ANALYSIS")
print("=" * 75)


bm25_results = load_json(
    BM25_RESULTS_PATH
)

dense_results = load_json(
    DENSE_RESULTS_PATH
)


print("\n[1] FILES LOADED")

print(
    f"BM25 results  : "
    f"{len(bm25_results)}"
)

print(
    f"Dense results : "
    f"{len(dense_results)}"
)


# 4. INDEX RESULTS BY QUESTION ID

bm25_by_id = {

    result["question_id"]: result

    for result
    in bm25_results
}


dense_by_id = {

    result["question_id"]: result

    for result
    in dense_results
}


bm25_ids = set(
    bm25_by_id.keys()
)

dense_ids = set(
    dense_by_id.keys()
)


if bm25_ids != dense_ids:

    raise ValueError(
        "BM25 and Dense result sets "
        "do not contain identical questions."
    )


question_ids = sorted(
    bm25_ids
)


print(
    f"Common questions: "
    f"{len(question_ids)}"
)


# 5. HIT COMPLEMENTARITY

K_VALUES = [
    1,
    3,
    5,
    10
]


comparison = {

    "number_of_questions":
        len(question_ids),

    "hit_comparison":
        {},

    "rank_comparison":
        {},

    "question_details":
        []
}


print(
    "\n[2] BM25 / DENSE HIT COMPLEMENTARITY"
)


for k in K_VALUES:

    categories = Counter()


    for question_id in question_ids:

        bm25_hit = source_hit(
            bm25_by_id[question_id],
            k
        )

        dense_hit = source_hit(
            dense_by_id[question_id],
            k
        )


        if bm25_hit and dense_hit:

            categories["both"] += 1

        elif bm25_hit and not dense_hit:

            categories["bm25_only"] += 1

        elif dense_hit and not bm25_hit:

            categories["dense_only"] += 1

        else:

            categories["neither"] += 1


    comparison[
        "hit_comparison"
    ][str(k)] = dict(
        categories
    )


    print(
        f"\nTop {k}:"
    )

    print(
        f"  Both succeed : "
        f"{categories['both']}"
    )

    print(
        f"  BM25 only    : "
        f"{categories['bm25_only']}"
    )

    print(
        f"  Dense only   : "
        f"{categories['dense_only']}"
    )

    print(
        f"  Neither      : "
        f"{categories['neither']}"
    )


# 6. FIRST GOLD RANK COMPARISON

print(
    "\n[3] FIRST CORRECT SOURCE RANK"
)


rank_categories = Counter()


for question_id in question_ids:

    bm25_rank = (
        bm25_by_id[
            question_id
        ]["first_gold_rank"]
    )

    dense_rank = (
        dense_by_id[
            question_id
        ]["first_gold_rank"]
    )


    # None means no gold source appeared
    # inside top 10.

    if bm25_rank is None:
        bm25_value = 999
    else:
        bm25_value = bm25_rank


    if dense_rank is None:
        dense_value = 999
    else:
        dense_value = dense_rank


    if dense_value < bm25_value:

        rank_categories[
            "dense_better"
        ] += 1

    elif bm25_value < dense_value:

        rank_categories[
            "bm25_better"
        ] += 1

    else:

        rank_categories[
            "same_rank"
        ] += 1


comparison[
    "rank_comparison"
] = dict(
    rank_categories
)


print(
    f"Dense ranks gold earlier : "
    f"{rank_categories['dense_better']}"
)

print(
    f"BM25 ranks gold earlier  : "
    f"{rank_categories['bm25_better']}"
)

print(
    f"Same first-gold rank     : "
    f"{rank_categories['same_rank']}"
)


# 7. QUESTION-LEVEL DETAILS

for question_id in question_ids:

    bm25_result = (
        bm25_by_id[
            question_id
        ]
    )

    dense_result = (
        dense_by_id[
            question_id
        ]
    )


    detail = {

        "question_id":
            question_id,

        "question":
            bm25_result[
                "question"
            ],

        "bm25_first_gold_rank":
            bm25_result[
                "first_gold_rank"
            ],

        "dense_first_gold_rank":
            dense_result[
                "first_gold_rank"
            ],

        "bm25_hit_at_5":
            source_hit(
                bm25_result,
                5
            ),

        "dense_hit_at_5":
            source_hit(
                dense_result,
                5
            )
    }


    comparison[
        "question_details"
    ].append(
        detail
    )


# 8. DENSE RESCUES BM25

print(
    "\n[4] DENSE RESCUES BM25 AT TOP 5"
)


dense_rescues = [

    detail

    for detail
    in comparison["question_details"]

    if (
        not detail[
            "bm25_hit_at_5"
        ]
        and
        detail[
            "dense_hit_at_5"
        ]
    )
]


print(
    f"Questions where BM25 fails "
    f"but Dense succeeds: "
    f"{len(dense_rescues)}"
)


for detail in dense_rescues[:10]:

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"ID: "
        f"{detail['question_id']}"
    )

    print(
        detail[
            "question"
        ]
    )

    print(
        f"BM25 first gold rank : "
        f"{detail['bm25_first_gold_rank']}"
    )

    print(
        f"Dense first gold rank: "
        f"{detail['dense_first_gold_rank']}"
    )


# 9. BM25 RESCUES DENSE

print(
    "\n[5] BM25 RESCUES DENSE AT TOP 5"
)


bm25_rescues = [

    detail

    for detail
    in comparison["question_details"]

    if (
        detail[
            "bm25_hit_at_5"
        ]
        and
        not detail[
            "dense_hit_at_5"
        ]
    )
]


print(
    f"Questions where Dense fails "
    f"but BM25 succeeds: "
    f"{len(bm25_rescues)}"
)


for detail in bm25_rescues[:10]:

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"ID: "
        f"{detail['question_id']}"
    )

    print(
        detail[
            "question"
        ]
    )

    print(
        f"BM25 first gold rank : "
        f"{detail['bm25_first_gold_rank']}"
    )

    print(
        f"Dense first gold rank: "
        f"{detail['dense_first_gold_rank']}"
    )


# 10. BOTH FAIL TOP 5

print(
    "\n[6] BOTH RETRIEVERS FAIL AT TOP 5"
)


both_fail = [

    detail

    for detail
    in comparison["question_details"]

    if (
        not detail[
            "bm25_hit_at_5"
        ]
        and
        not detail[
            "dense_hit_at_5"
        ]
    )
]


print(
    f"Questions where both fail: "
    f"{len(both_fail)}"
)


for detail in both_fail[:10]:

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"ID: "
        f"{detail['question_id']}"
    )

    print(
        detail[
            "question"
        ]
    )

    print(
        f"BM25 first gold rank : "
        f"{detail['bm25_first_gold_rank']}"
    )

    print(
        f"Dense first gold rank: "
        f"{detail['dense_first_gold_rank']}"
    )


# 11. WEEKEND QUESTION

print(
    "\n[7] WEEKEND QUESTION COMPARISON"
)


TARGET_ID = "17610439"


if TARGET_ID in bm25_by_id:

    bm25_target = (
        bm25_by_id[
            TARGET_ID
        ]
    )

    dense_target = (
        dense_by_id[
            TARGET_ID
        ]
    )


    print(
        bm25_target[
            "question"
        ]
    )

    print(
        f"BM25 first gold rank : "
        f"{bm25_target['first_gold_rank']}"
    )

    print(
        f"Dense first gold rank: "
        f"{dense_target['first_gold_rank']}"
    )


# 12. SAVE COMPARISON

save_json(
    comparison,
    OUTPUT_PATH
)


print(
    "\n[8] RESULTS SAVED"
)

print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 7 COMPLETE"
)

print(
    "=" * 75
)
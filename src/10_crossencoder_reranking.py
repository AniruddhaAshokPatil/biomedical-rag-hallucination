import json
from pathlib import Path

import numpy as np

from sentence_transformers import CrossEncoder


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


WEIGHTED_RESULTS_PATH = (
    RESULTS_DIR
    / "weighted_rrf_dev_results.json"
)

WEIGHTED_SUMMARY_PATH = (
    RESULTS_DIR
    / "weighted_rrf_dev_summary.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR
    / "evaluation_ground_truth.json"
)


RERANKED_RESULTS_PATH = (
    RESULTS_DIR
    / "reranked_dev_results.json"
)

RERANKED_SUMMARY_PATH = (
    RESULTS_DIR
    / "reranked_dev_summary.json"
)


# 2. CONFIGURATION

RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)


K_VALUES = [
    1,
    3,
    5,
    10
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


# 4. LOAD DATA

print("=" * 75)

print(
    "SECTION 10 - CROSS-ENCODER RERANKING"
)

print("=" * 75)


weighted_results = load_json(
    WEIGHTED_RESULTS_PATH
)

weighted_summary = load_json(
    WEIGHTED_SUMMARY_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)


print("\n[1] DATA LOADED")

print(
    f"Weighted-RRF questions : "
    f"{len(weighted_results)}"
)


# 5. DEVELOPMENT SAFETY CHECK

print(
    "\n[2] DEVELOPMENT SAFETY CHECK"
)


wrong_split = []


for result in weighted_results:

    question_id = result[
        "question_id"
    ]

    split = ground_truth[
        question_id
    ]["split"]


    if split != "development":

        wrong_split.append(
            question_id
        )


print(
    f"Non-development questions: "
    f"{len(wrong_split)}"
)


if wrong_split:

    raise ValueError(
        "Test-set leakage detected."
    )


print(
    "PASS: Reranker evaluation uses "
    "development questions only."
)


# 6. LOAD CROSS-ENCODER

print(
    "\n[3] LOADING CROSS-ENCODER"
)

print(
    f"Model: {RERANKER_MODEL}"
)


reranker = CrossEncoder(
    RERANKER_MODEL
)


print(
    f"Device: "
    f"{reranker.device}"
)

# 7. METRIC STORAGE

source_hits = {

    k: 0

    for k in K_VALUES
}


context_recalls = {

    k: []

    for k in K_VALUES
}


reciprocal_ranks = []

no_gold_top_10 = 0

reranked_results = []


# 8. RUN RERANKING
# The Weighted RRF result already contains the union of BM25 and Dense candidates.
# Usually this will be between 10 and 20 unique passages.
# We score each:
# (question, passage)
# pair jointly using the Cross-Encoder.


print(
    "\n[4] RUNNING CROSS-ENCODER RERANKING"
)


total_questions = len(
    weighted_results
)


for index, result in enumerate(
    weighted_results,
    start=1
):

    question_id = result[
        "question_id"
    ]

    question_text = result[
        "question"
    ]


    candidates = result[
        "retrieved_documents"
    ]


    # Construct question-passage pairs

    pairs = [

        (
            question_text,
            document["text"]
        )

        for document
        in candidates
    ]


    # Cross-Encoder relevance scoring


    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )


    scores = np.asarray(
        scores
    ).reshape(-1)


    # Copy documents and attach reranker score
    

    scored_candidates = []


    for document, score in zip(
        candidates,
        scores
    ):

        updated_document = (
            document.copy()
        )


        # Preserve previous Weighted-RRF rank
        updated_document[
            "pre_rerank_rank"
        ] = document[
            "rank"
        ]


        updated_document[
            "reranker_score"
        ] = float(
            score
        )


        scored_candidates.append(
            updated_document
        )


    # Sort highest reranker relevance first

    scored_candidates.sort(

        key=lambda document:
            document[
                "reranker_score"
            ],

        reverse=True
    )


    # Assign new ranks
   

    for new_rank, document in enumerate(
        scored_candidates,
        start=1
    ):

        document[
            "rank"
        ] = new_rank


    # Gold evidence

    gold_context_ids = set(

        ground_truth[
            question_id
        ][
            "gold_context_ids"
        ]
    )


    # Find first gold rank

    first_gold_rank = None


    for document in scored_candidates:

        if (
            document[
                "document_id"
            ]
            in gold_context_ids
        ):

            first_gold_rank = (
                document[
                    "rank"
                ]
            )

            break


    # METRICS @ K

    for k in K_VALUES:

        top_k = (
            scored_candidates[:k]
        )


        retrieved_gold = [

            document

            for document in top_k

            if (
                document[
                    "document_id"
                ]
                in gold_context_ids
            )
        ]


        if retrieved_gold:

            source_hits[
                k
            ] += 1


        recall = (

            len(
                retrieved_gold
            )

            /
            len(
                gold_context_ids
            )
        )


        context_recalls[
            k
        ].append(
            recall
        )


    # MRR@10

    if (
        first_gold_rank
        is not None

        and

        first_gold_rank <= 10
    ):

        reciprocal_ranks.append(

            1
            /
            first_gold_rank
        )

    else:

        reciprocal_ranks.append(
            0
        )

        no_gold_top_10 += 1

    # STORE RESULT

    reranked_results.append(
        {

            "question_id":
                question_id,

            "question":
                question_text,

            "first_gold_rank":
                first_gold_rank,

            "retrieved_documents":
                scored_candidates
        }
    )


    if index % 100 == 0:

        print(
            f"Processed "
            f"{index}/"
            f"{total_questions} "
            f"questions"
        )


# 9. CALCULATE SUMMARY

summary = {

    "retriever":
        "Weighted RRF + Cross-Encoder",

    "reranker_model":
        RERANKER_MODEL,

    "candidate_source":
        "Weighted RRF union of BM25 top-10 "
        "and Dense top-10",

    "number_of_questions":
        total_questions,

    "source_hit_at_k":
        {},

    "gold_context_recall_at_k":
        {},

    "mrr_at_10":
        None,

    "queries_without_gold_in_top_10":
        no_gold_top_10
}


for k in K_VALUES:

    summary[
        "source_hit_at_k"
    ][str(k)] = (

        source_hits[k]

        /
        total_questions
    )


    summary[
        "gold_context_recall_at_k"
    ][str(k)] = (

        sum(
            context_recalls[k]
        )

        /
        len(
            context_recalls[k]
        )
    )


summary[
    "mrr_at_10"
] = (

    sum(
        reciprocal_ranks
    )

    /
    len(
        reciprocal_ranks
    )
)


# 10. PRINT RESULTS

print(
    "\n[5] RERANKED RETRIEVAL RESULTS"
)


print("\nSource Hit@K")

print("-" * 40)


for k in K_VALUES:

    value = summary[
        "source_hit_at_k"
    ][str(k)]


    print(
        f"Hit@{k:<2} : "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


print(
    "\nMean Gold-Context Recall@K"
)

print("-" * 40)


for k in K_VALUES:

    value = summary[
        "gold_context_recall_at_k"
    ][str(k)]


    print(
        f"Recall@{k:<2} : "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


print(
    "\nRanking Quality"
)

print("-" * 40)


print(
    f"MRR@10 : "
    f"{summary['mrr_at_10']:.4f}"
)


print(
    f"Queries with no correct source "
    f"in top 10: "
    f"{no_gold_top_10}"
)


# 11. WEIGHTED RRF VS RERANKER

print(
    "\n[6] WEIGHTED RRF VS RERANKED"
)


print(
    f"\n"
    f"{'Metric':<15}"
    f"{'Weighted RRF':>15}"
    f"{'Reranked':>15}"
)

print(
    "-" * 45
)


for k in K_VALUES:

    old_value = (

        weighted_summary[
            "source_hit_at_k"
        ][str(k)]
    )

    new_value = (

        summary[
            "source_hit_at_k"
        ][str(k)]
    )


    print(
        f"Hit@{k:<10}"
        f"{old_value:>15.4f}"
        f"{new_value:>15.4f}"
    )


print(
    "-" * 45
)


print(
    f"{'MRR@10':<15}"
    f"{weighted_summary['mrr_at_10']:>15.4f}"
    f"{summary['mrr_at_10']:>15.4f}"
)


# 12. CONTEXT RECALL COMPARISON

print(
    "\n[7] CONTEXT RECALL COMPARISON"
)


print(
    f"\n"
    f"{'Metric':<15}"
    f"{'Weighted RRF':>15}"
    f"{'Reranked':>15}"
)

print(
    "-" * 45
)


for k in K_VALUES:

    old_value = (

        weighted_summary[
            "gold_context_recall_at_k"
        ][str(k)]
    )

    new_value = (

        summary[
            "gold_context_recall_at_k"
        ][str(k)]
    )


    print(
        f"Recall@{k:<7}"
        f"{old_value:>15.4f}"
        f"{new_value:>15.4f}"
    )


# 13. WEEKEND QUESTION

print(
    "\n[8] WEEKEND QUESTION AFTER RERANKING"
)


TARGET_ID = "17610439"


for result in reranked_results:

    if (
        result[
            "question_id"
        ]
        ==
        TARGET_ID
    ):

        gold_ids = set(

            ground_truth[
                TARGET_ID
            ][
                "gold_context_ids"
            ]
        )


        print(
            result[
                "question"
            ]
        )


        print(
            f"\nFirst gold rank: "
            f"{result['first_gold_rank']}"
        )


        print(
            "\nTop 5:"
        )


        for document in (
            result[
                "retrieved_documents"
            ][:5]
        ):

            marker = (

                "CORRECT"

                if (
                    document[
                        "document_id"
                    ]
                    in gold_ids
                )

                else ""
            )


            print(
                "\n"
                + "-"
                * 60
            )


            print(
                f"New rank: "
                f"{document['rank']} "
                f"{marker}"
            )


            print(
                f"Previous rank: "
                f"{document['pre_rerank_rank']}"
            )


            print(
                f"Source: "
                f"{document['source_record_id']}"
            )


            print(
                f"Reranker score: "
                f"{document['reranker_score']:.6f}"
            )


            print(
                f"Section: "
                f"{document['section']}"
            )


            print(
                f"Text: "
                f"{document['text']}"
            )


        break


# 14. FIND BIGGEST IMPROVEMENTS

print(
    "\n[9] EXAMPLES WHERE RERANKING "
    "IMPROVED GOLD RANK"
)


improvements = []


weighted_by_id = {

    result[
        "question_id"
    ]: result

    for result
    in weighted_results
}


for result in reranked_results:

    question_id = result[
        "question_id"
    ]


    old_rank = weighted_by_id[
        question_id
    ][
        "first_gold_rank"
    ]


    new_rank = result[
        "first_gold_rank"
    ]


    if old_rank is None:

        old_value = 999

    else:

        old_value = old_rank


    if new_rank is None:

        new_value = 999

    else:

        new_value = new_rank


    if new_value < old_value:

        improvements.append(
            {
                "question_id":
                    question_id,

                "question":
                    result[
                        "question"
                    ],

                "old_rank":
                    old_rank,

                "new_rank":
                    new_rank,

                "improvement":
                    old_value
                    -
                    new_value
            }
        )


improvements.sort(

    key=lambda item:
        item[
            "improvement"
        ],

    reverse=True
)


print(
    f"Questions with improved "
    f"first-gold rank: "
    f"{len(improvements)}"
)


for item in improvements[:10]:

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"ID: "
        f"{item['question_id']}"
    )

    print(
        item[
            "question"
        ]
    )

    print(
        f"Before reranking: "
        f"{item['old_rank']}"
    )

    print(
        f"After reranking : "
        f"{item['new_rank']}"
    )


# 15. FIND DEGRADATIONS

print(
    "\n[10] EXAMPLES WHERE RERANKING "
    "WORSENED GOLD RANK"
)


degradations = []


for result in reranked_results:

    question_id = result[
        "question_id"
    ]


    old_rank = weighted_by_id[
        question_id
    ][
        "first_gold_rank"
    ]


    new_rank = result[
        "first_gold_rank"
    ]


    old_value = (
        old_rank
        if old_rank is not None
        else 999
    )


    new_value = (
        new_rank
        if new_rank is not None
        else 999
    )


    if new_value > old_value:

        degradations.append(
            {
                "question_id":
                    question_id,

                "question":
                    result[
                        "question"
                    ],

                "old_rank":
                    old_rank,

                "new_rank":
                    new_rank,

                "change":
                    new_value
                    -
                    old_value
            }
        )


degradations.sort(

    key=lambda item:
        item[
            "change"
        ],

    reverse=True
)


print(
    f"Questions with worsened "
    f"first-gold rank: "
    f"{len(degradations)}"
)


for item in degradations[:10]:

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"ID: "
        f"{item['question_id']}"
    )

    print(
        item[
            "question"
        ]
    )

    print(
        f"Before reranking: "
        f"{item['old_rank']}"
    )

    print(
        f"After reranking : "
        f"{item['new_rank']}"
    )


# 16. SAVE

save_json(
    reranked_results,
    RERANKED_RESULTS_PATH
)

save_json(
    summary,
    RERANKED_SUMMARY_PATH
)


print(
    "\n[11] RESULTS SAVED"
)

print(
    RERANKED_RESULTS_PATH
)

print(
    RERANKED_SUMMARY_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 10 COMPLETE"
)

print(
    "=" * 75
)
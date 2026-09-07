import json
from pathlib import Path
from collections import defaultdict


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
)

BM25_RESULTS_PATH = (
    RESULTS_DIR
    / "bm25_dev_results.json"
)

DENSE_RESULTS_PATH = (
    RESULTS_DIR
    / "dense_dev_results.json"
)

BM25_SUMMARY_PATH = (
    RESULTS_DIR
    / "bm25_dev_summary.json"
)

DENSE_SUMMARY_PATH = (
    RESULTS_DIR
    / "dense_dev_summary.json"
)

HYBRID_RESULTS_PATH = (
    RESULTS_DIR
    / "hybrid_rrf_dev_results.json"
)

HYBRID_SUMMARY_PATH = (
    RESULTS_DIR
    / "hybrid_rrf_dev_summary.json"
)


# 2. SETTINGS

K_VALUES = [
    1,
    3,
    5,
    10
]

MAX_K = max(K_VALUES)


# Standard RRF constant
RRF_K = 60


# 3. JSON HELPERS

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
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


# 4. LOAD PREVIOUS RETRIEVAL RESULTS

print("=" * 75)
print("SECTION 8 - HYBRID RETRIEVAL USING RRF")
print("=" * 75)


bm25_results = load_json(
    BM25_RESULTS_PATH
)

dense_results = load_json(
    DENSE_RESULTS_PATH
)

bm25_summary = load_json(
    BM25_SUMMARY_PATH
)

dense_summary = load_json(
    DENSE_SUMMARY_PATH
)


print("\n[1] RESULTS LOADED")

print(
    f"BM25 questions  : "
    f"{len(bm25_results)}"
)

print(
    f"Dense questions : "
    f"{len(dense_results)}"
)


# 5. INDEX RESULTS BY QUESTION

bm25_by_id = {

    result["question_id"]:
        result

    for result
    in bm25_results
}


dense_by_id = {

    result["question_id"]:
        result

    for result
    in dense_results
}


if set(bm25_by_id) != set(dense_by_id):

    raise ValueError(
        "BM25 and Dense question sets differ."
    )


question_ids = sorted(
    bm25_by_id.keys()
)


print(
    f"Common questions: "
    f"{len(question_ids)}"
)


# 6. RRF FUNCTION

def reciprocal_rank_fusion(
    bm25_documents,
    dense_documents,
    rrf_k=60
):
    """
    Combine BM25 and Dense rankings using
    Reciprocal Rank Fusion.

    A document receives:

        1 / (rrf_k + rank)

    from every retriever in which it appears.

    Documents appearing highly in both rankings
    accumulate a larger score.
    """

    fusion_scores = defaultdict(float)

    document_data = {}


    
    # -------BM25 contribution----------
    

    for document in bm25_documents:

        document_id = document[
            "document_id"
        ]

        rank = document[
            "rank"
        ]


        fusion_scores[
            document_id
        ] += (

            1
            /
            (
                rrf_k
                +
                rank
            )
        )


        document_data[
            document_id
        ] = document.copy()



    # ----------Dense contribution---------
    

    for document in dense_documents:

        document_id = document[
            "document_id"
        ]

        rank = document[
            "rank"
        ]


        fusion_scores[
            document_id
        ] += (

            1
            /
            (
                rrf_k
                +
                rank
            )
        )


        if document_id not in document_data:

            document_data[
                document_id
            ] = document.copy()


  
    # ---------Sort by RRF score----------
    
    ranked_ids = sorted(

        fusion_scores.keys(),

        key=lambda document_id:
            fusion_scores[
                document_id
            ],

        reverse=True
    )


    fused_results = []


    for new_rank, document_id in enumerate(
        ranked_ids,
        start=1
    ):

        document = (
            document_data[
                document_id
            ].copy()
        )


        document[
            "rank"
        ] = new_rank


        document[
            "rrf_score"
        ] = round(
            fusion_scores[
                document_id
            ],
            8
        )


        # Find original BM25 rank
        bm25_rank = None

        for bm25_doc in bm25_documents:

            if (
                bm25_doc[
                    "document_id"
                ]
                ==
                document_id
            ):

                bm25_rank = (
                    bm25_doc[
                        "rank"
                    ]
                )

                break


        # Find original Dense rank
        dense_rank = None

        for dense_doc in dense_documents:

            if (
                dense_doc[
                    "document_id"
                ]
                ==
                document_id
            ):

                dense_rank = (
                    dense_doc[
                        "rank"
                    ]
                )

                break


        document[
            "bm25_rank"
        ] = bm25_rank

        document[
            "dense_rank"
        ] = dense_rank


        fused_results.append(
            document
        )


    return fused_results


# 7. METRIC STORAGE

source_hits = {

    k: 0

    for k in K_VALUES
}


gold_context_recalls = {

    k: []

    for k in K_VALUES
}


reciprocal_ranks = []

no_gold_top_10 = 0

all_hybrid_results = []


# 8. RUN HYBRID FUSION

print(
    "\n[2] RUNNING RECIPROCAL RANK FUSION"
)


for index, question_id in enumerate(
    question_ids,
    start=1
):

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


    question_text = (
        bm25_result[
            "question"
        ]
    )


    
    # ------Combine top-10 BM25 + top-10 Dense------
    

    hybrid_documents = (
        reciprocal_rank_fusion(

            bm25_result[
                "retrieved_documents"
            ],

            dense_result[
                "retrieved_documents"
            ],

            rrf_k=RRF_K
        )
    )


    # Determine gold context IDs
    # We can infer them safely from the boolean
    # labels already produced by previous retrieval
    # evaluation runs.

    gold_context_ids = set()


    for document in (
        bm25_result[
            "retrieved_documents"
        ]
        +
        dense_result[
            "retrieved_documents"
        ]
    ):

        if document[
            "is_gold_context"
        ]:

            gold_context_ids.add(
                document[
                    "document_id"
                ]
            )


    # --------------------------------------------------------
    # For context recall we need the TOTAL number of gold contexts, not only those retrieved.
    # We recover that from source documents appearing in the saved results only for ranking metrics.
    # Therefore context recall will be calculated separately below from known per-question source passage count where available.
    # --------------------------------------------------------


    first_gold_rank = None


    for document in hybrid_documents:

        if (
            document[
                "is_gold_source"
            ]
            and
            first_gold_rank is None
        ):

            first_gold_rank = (
                document[
                    "rank"
                ]
            )



    #---------SOURCE HIT@K-------

    for k in K_VALUES:

        top_k = (
            hybrid_documents[:k]
        )


        hit = any(

            document[
                "is_gold_source"
            ]

            for document
            in top_k
        )


        if hit:

            source_hits[
                k
            ] += 1

    
    # -------MRR-------

    if (
        first_gold_rank is not None
        and
        first_gold_rank <= 10
    ):

        reciprocal_rank = (

            1
            /
            first_gold_rank
        )

    else:

        reciprocal_rank = 0

        no_gold_top_10 += 1


    reciprocal_ranks.append(
        reciprocal_rank
    )


    all_hybrid_results.append(
        {

            "question_id":
                question_id,

            "question":
                question_text,

            "first_gold_rank":
                first_gold_rank,

            "retrieved_documents":
                hybrid_documents
        }
    )


    if index % 100 == 0:

        print(
            f"Processed "
            f"{index}/"
            f"{len(question_ids)} "
            f"questions"
        )


# 9. SOURCE HIT METRICS

summary = {

    "retriever":
        "Hybrid RRF",

    "rrf_k":
        RRF_K,

    "number_of_questions":
        len(question_ids),

    "source_hit_at_k":
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
        len(question_ids)
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


# 10. PRINT HYBRID RESULTS

print(
    "\n[3] HYBRID RETRIEVAL RESULTS"
)


print("\nSource Hit@K")

print("-" * 40)


for k in K_VALUES:

    value = (

        summary[
            "source_hit_at_k"
        ][str(k)]
    )


    print(
        f"Hit@{k:<2} : "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


print("\nRanking Quality")

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


# 11. THREE-WAY COMPARISON

print(
    "\n[4] BM25 VS DENSE VS HYBRID"
)


print(
    f"\n"
    f"{'Metric':<15}"
    f"{'BM25':>12}"
    f"{'Dense':>12}"
    f"{'Hybrid':>12}"
)


print(
    "-" * 51
)


for k in K_VALUES:

    bm25_value = (
        bm25_summary[
            "source_hit_at_k"
        ][str(k)]
    )

    dense_value = (
        dense_summary[
            "source_hit_at_k"
        ][str(k)]
    )

    hybrid_value = (
        summary[
            "source_hit_at_k"
        ][str(k)]
    )


    print(
        f"Hit@{k:<10}"
        f"{bm25_value:>12.4f}"
        f"{dense_value:>12.4f}"
        f"{hybrid_value:>12.4f}"
    )


print(
    "-" * 51
)


print(
    f"{'MRR@10':<15}"
    f"{bm25_summary['mrr_at_10']:>12.4f}"
    f"{dense_summary['mrr_at_10']:>12.4f}"
    f"{summary['mrr_at_10']:>12.4f}"
)


# 12. WEEKEND QUESTION

print(
    "\n[5] WEEKEND QUESTION - HYBRID"
)


TARGET_ID = "17610439"


target = None


for result in all_hybrid_results:

    if (
        result[
            "question_id"
        ]
        ==
        TARGET_ID
    ):

        target = result

        break


if target:

    print(
        target[
            "question"
        ]
    )


    print(
        f"\nHybrid first gold rank: "
        f"{target['first_gold_rank']}"
    )


    print(
        "\nHybrid Top 5:"
    )


    for document in (
        target[
            "retrieved_documents"
        ][:5]
    ):

        marker = (

            "CORRECT SOURCE"

            if document[
                "is_gold_source"
            ]

            else ""
        )


        print(
            "\n"
            + "-"
            * 60
        )


        print(
            f"Rank: "
            f"{document['rank']} "
            f"{marker}"
        )


        print(
            f"Source ID: "
            f"{document['source_record_id']}"
        )


        print(
            f"BM25 rank: "
            f"{document['bm25_rank']}"
        )


        print(
            f"Dense rank: "
            f"{document['dense_rank']}"
        )


        print(
            f"RRF score: "
            f"{document['rrf_score']}"
        )


        print(
            f"Section: "
            f"{document['section']}"
        )


        print(
            f"Text: "
            f"{document['text']}"
        )


# 13. SAVE RESULTS

save_json(
    all_hybrid_results,
    HYBRID_RESULTS_PATH
)

save_json(
    summary,
    HYBRID_SUMMARY_PATH
)


print(
    "\n[6] RESULTS SAVED"
)


print(
    HYBRID_RESULTS_PATH
)

print(
    HYBRID_SUMMARY_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 8 COMPLETE"
)

print(
    "=" * 75
)
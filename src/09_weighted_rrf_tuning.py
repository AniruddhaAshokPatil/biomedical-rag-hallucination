import json
from pathlib import Path
from collections import defaultdict


# 1. PATHS

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


BM25_RESULTS_PATH = (
    RESULTS_DIR
    / "bm25_dev_results.json"
)

DENSE_RESULTS_PATH = (
    RESULTS_DIR
    / "dense_dev_results.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR
    / "evaluation_ground_truth.json"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "weighted_rrf_tuning.json"
)

BEST_RESULTS_PATH = (
    RESULTS_DIR
    / "weighted_rrf_dev_results.json"
)

BEST_SUMMARY_PATH = (
    RESULTS_DIR
    / "weighted_rrf_dev_summary.json"
)


# 2. SETTINGS

RRF_K = 60

K_VALUES = [
    1,
    3,
    5,
    10
]


# ------------------------------------------------------------
# Dense is held at 1.0.
# We vary how much influence BM25 receives.
# These values are defined BEFORE looking at
# weighted-RRF performance.
# ------------------------------------------------------------

WEIGHT_CONFIGURATIONS = [

    {
        "bm25_weight": 0.00,
        "dense_weight": 1.00
    },

    {
        "bm25_weight": 0.25,
        "dense_weight": 1.00
    },

    {
        "bm25_weight": 0.50,
        "dense_weight": 1.00
    },

    {
        "bm25_weight": 0.75,
        "dense_weight": 1.00
    },

    {
        "bm25_weight": 1.00,
        "dense_weight": 1.00
    }
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


# 4. WEIGHTED RRF

def weighted_rrf(
    bm25_documents,
    dense_documents,
    bm25_weight,
    dense_weight,
    rrf_k=60
):

    scores = defaultdict(float)

    document_data = {}


    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    for document in bm25_documents:

        document_id = document[
            "document_id"
        ]

        rank = document[
            "rank"
        ]


        scores[
            document_id
        ] += (

            bm25_weight
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



    # ------Dense contribution------------
   

    for document in dense_documents:

        document_id = document[
            "document_id"
        ]

        rank = document[
            "rank"
        ]


        scores[
            document_id
        ] += (

            dense_weight
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


    # -----Rank fused candidates-------

    ranked_ids = sorted(

        scores,

        key=lambda document_id:
            scores[document_id],

        reverse=True
    )


    fused = []


    for rank, document_id in enumerate(
        ranked_ids,
        start=1
    ):

        document = (
            document_data[
                document_id
            ].copy()
        )


        document["rank"] = rank

        document["weighted_rrf_score"] = round(
            scores[document_id],
            8
        )


        fused.append(
            document
        )


    return fused


# 5. LOAD DATA

print("=" * 75)

print(
    "SECTION 9 - WEIGHTED RRF TUNING"
)

print("=" * 75)


bm25_results = load_json(
    BM25_RESULTS_PATH
)

dense_results = load_json(
    DENSE_RESULTS_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)


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


question_ids = sorted(
    bm25_by_id.keys()
)


if set(question_ids) != set(dense_by_id):

    raise ValueError(
        "BM25 and Dense question IDs differ."
    )


# 6. DEVELOPMENT SAFETY CHECK

print(
    "\n[1] DEVELOPMENT SAFETY CHECK"
)


test_questions_found = 0


for question_id in question_ids:

    if (
        ground_truth[
            question_id
        ]["split"]
        !=
        "development"
    ):

        test_questions_found += 1


print(
    f"Test questions found: "
    f"{test_questions_found}"
)


if test_questions_found:

    raise ValueError(
        "Test leakage detected."
    )


print(
    "PASS: Weight tuning uses development "
    "questions only."
)


# 7. EVALUATION FUNCTION

def evaluate_configuration(
    bm25_weight,
    dense_weight
):

    hit_counts = {

        k: 0

        for k in K_VALUES
    }


    context_recalls = {

        k: []

        for k in K_VALUES
    }


    reciprocal_ranks = []

    no_gold_top10 = 0

    detailed_results = []


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


        fused = weighted_rrf(

            bm25_result[
                "retrieved_documents"
            ],

            dense_result[
                "retrieved_documents"
            ],

            bm25_weight,
            dense_weight,
            RRF_K
        )


        gold_context_ids = set(

            ground_truth[
                question_id
            ]["gold_context_ids"]
        )


        first_gold_rank = None


        for document in fused:

            if (
                document[
                    "document_id"
                ]
                in gold_context_ids
            ):

                if first_gold_rank is None:

                    first_gold_rank = (
                        document["rank"]
                    )

       
        # Metrics @ K

        for k in K_VALUES:

            top_k = fused[:k]


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

                hit_counts[k] += 1


            recall = (

                len(retrieved_gold)
                /
                len(gold_context_ids)
            )


            context_recalls[
                k
            ].append(
                recall
            )


        #------- MRR @ 10---------

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

            no_gold_top10 += 1


        detailed_results.append(
            {

                "question_id":
                    question_id,

                "question":
                    bm25_result[
                        "question"
                    ],

                "first_gold_rank":
                    first_gold_rank,

                "retrieved_documents":
                    fused
            }
        )


    # ------Summary----------------

    summary = {

        "bm25_weight":
            bm25_weight,

        "dense_weight":
            dense_weight,

        "source_hit_at_k":
            {},

        "gold_context_recall_at_k":
            {},

        "mrr_at_10":
            sum(
                reciprocal_ranks
            )
            /
            len(
                reciprocal_ranks
            ),

        "queries_without_gold_in_top_10":
            no_gold_top10
    }


    for k in K_VALUES:

        summary[
            "source_hit_at_k"
        ][str(k)] = (

            hit_counts[k]

            /
            len(question_ids)
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


    return (
        summary,
        detailed_results
    )


# 8. RUN ALL WEIGHT CONFIGURATIONS

print(
    "\n[2] TESTING WEIGHT CONFIGURATIONS"
)


all_summaries = []

all_detailed_results = {}


for config in WEIGHT_CONFIGURATIONS:

    bm25_weight = config[
        "bm25_weight"
    ]

    dense_weight = config[
        "dense_weight"
    ]


    print(
        f"\nTesting:"
        f" BM25={bm25_weight:.2f},"
        f" Dense={dense_weight:.2f}"
    )


    (
        summary,
        detailed_results

    ) = evaluate_configuration(

        bm25_weight,
        dense_weight
    )


    all_summaries.append(
        summary
    )


    config_key = (
        f"bm25_{bm25_weight:.2f}"
        f"_dense_{dense_weight:.2f}"
    )


    all_detailed_results[
        config_key
    ] = detailed_results


    print(
        f"  Hit@1       : "
        f"{summary['source_hit_at_k']['1']:.4f}"
    )

    print(
        f"  Hit@5       : "
        f"{summary['source_hit_at_k']['5']:.4f}"
    )

    print(
        f"  Recall@5    : "
        f"{summary['gold_context_recall_at_k']['5']:.4f}"
    )

    print(
        f"  MRR@10      : "
        f"{summary['mrr_at_10']:.4f}"
    )


# 9. SELECT BEST CONFIGURATION

# PRIMARY selection:
# Gold Context Recall@5
# TIEBREAKERS:
# 1. Hit@5
# 2. MRR@10
# Why Recall@5?
# Our downstream generator will receive a small evidence set. We want that evidence set to contain as much gold evidence as possible.


best_summary = max(

    all_summaries,

    key=lambda result: (

        result[
            "gold_context_recall_at_k"
        ]["5"],

        result[
            "source_hit_at_k"
        ]["5"],

        result[
            "mrr_at_10"
        ]
    )
)


best_bm25_weight = (
    best_summary[
        "bm25_weight"
    ]
)

best_dense_weight = (
    best_summary[
        "dense_weight"
    ]
)


best_key = (
    f"bm25_{best_bm25_weight:.2f}"
    f"_dense_{best_dense_weight:.2f}"
)


best_detailed_results = (
    all_detailed_results[
        best_key
    ]
)



# 10. PRINT COMPARISON TABLE


print(
    "\n[3] WEIGHT COMPARISON"
)


header = (
    f"{'BM25':>6}"
    f"{'Dense':>8}"
    f"{'Hit@1':>10}"
    f"{'Hit@5':>10}"
    f"{'Recall@5':>12}"
    f"{'MRR@10':>10}"
)


print(header)

print(
    "-" * len(header)
)


for summary in all_summaries:

    print(

        f"{summary['bm25_weight']:>6.2f}"

        f"{summary['dense_weight']:>8.2f}"

        f"{summary['source_hit_at_k']['1']:>10.4f}"

        f"{summary['source_hit_at_k']['5']:>10.4f}"

        f"{summary['gold_context_recall_at_k']['5']:>12.4f}"

        f"{summary['mrr_at_10']:>10.4f}"
    )



# 11. BEST CONFIGURATION


print(
    "\n[4] SELECTED CONFIGURATION"
)


print(
    f"BM25 weight  : "
    f"{best_bm25_weight}"
)

print(
    f"Dense weight : "
    f"{best_dense_weight}"
)


print(
    f"\nHit@1    : "
    f"{best_summary['source_hit_at_k']['1']:.4f}"
)

print(
    f"Hit@3    : "
    f"{best_summary['source_hit_at_k']['3']:.4f}"
)

print(
    f"Hit@5    : "
    f"{best_summary['source_hit_at_k']['5']:.4f}"
)

print(
    f"Hit@10   : "
    f"{best_summary['source_hit_at_k']['10']:.4f}"
)


print(
    f"\nRecall@1 : "
    f"{best_summary['gold_context_recall_at_k']['1']:.4f}"
)

print(
    f"Recall@3 : "
    f"{best_summary['gold_context_recall_at_k']['3']:.4f}"
)

print(
    f"Recall@5 : "
    f"{best_summary['gold_context_recall_at_k']['5']:.4f}"
)

print(
    f"Recall@10: "
    f"{best_summary['gold_context_recall_at_k']['10']:.4f}"
)


print(
    f"\nMRR@10   : "
    f"{best_summary['mrr_at_10']:.4f}"
)



# 12. WEEKEND QUESTION


print(
    "\n[5] WEEKEND QUESTION"
)


TARGET_ID = "17610439"


for result in best_detailed_results:

    if (
        result[
            "question_id"
        ]
        ==
        TARGET_ID
    ):

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
                    in
                    ground_truth[
                        TARGET_ID
                    ][
                        "gold_context_ids"
                    ]
                )

                else ""
            )


            print(
                "\n"
                + "-"
                * 60
            )

            print(
                f"Rank "
                f"{document['rank']} "
                f"{marker}"
            )

            print(
                f"Source: "
                f"{document['source_record_id']}"
            )

            print(
                f"Score: "
                f"{document['weighted_rrf_score']}"
            )

            print(
                f"Text: "
                f"{document['text']}"
            )


        break



# 13. SAVE


save_json(
    {
        "selection_metric":
            "gold_context_recall_at_5",

        "tie_breakers": [
            "source_hit_at_5",
            "mrr_at_10"
        ],

        "all_configurations":
            all_summaries,

        "selected_configuration":
            best_summary
    },
    OUTPUT_PATH
)


save_json(
    best_detailed_results,
    BEST_RESULTS_PATH
)


save_json(
    best_summary,
    BEST_SUMMARY_PATH
)


print(
    "\n[6] RESULTS SAVED"
)

print(
    OUTPUT_PATH
)

print(
    BEST_RESULTS_PATH
)

print(
    BEST_SUMMARY_PATH
)


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 9 COMPLETE"
)

print(
    "=" * 75
)
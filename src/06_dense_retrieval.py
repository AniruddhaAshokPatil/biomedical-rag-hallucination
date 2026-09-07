import json
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "retrieval"
)

EMBEDDINGS_DIR = (
    PROJECT_ROOT / "data" / "embeddings"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EMBEDDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DEVELOPMENT_QUESTIONS_PATH = (
    PROCESSED_DIR / "development_questions.json"
)

CORPUS_PATH = (
    PROCESSED_DIR / "retrieval_corpus.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR / "evaluation_ground_truth.json"
)


DENSE_RESULTS_PATH = (
    RESULTS_DIR / "dense_dev_results.json"
)

DENSE_SUMMARY_PATH = (
    RESULTS_DIR / "dense_dev_summary.json"
)

BM25_SUMMARY_PATH = (
    RESULTS_DIR / "bm25_dev_summary.json"
)


CORPUS_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR
    / "minilm_corpus_embeddings.npy"
)


# 2. MODEL CONFIGURATION

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

BATCH_SIZE = 32


# 3. RETRIEVAL SETTINGS

K_VALUES = [
    1,
    3,
    5,
    10
]

MAX_K = max(
    K_VALUES
)


# 4. JSON HELPERS

def load_json(file_path: Path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

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


# 5. LOAD DATA

print("=" * 75)

print(
    "SECTION 6 - DENSE SEMANTIC RETRIEVAL"
)

print("=" * 75)


development_questions = load_json(
    DEVELOPMENT_QUESTIONS_PATH
)

retrieval_corpus = load_json(
    CORPUS_PATH
)

evaluation_ground_truth = load_json(
    GROUND_TRUTH_PATH
)


print("\n[1] DATA LOADED")

print(
    f"Development questions : "
    f"{len(development_questions)}"
)

print(
    f"Retrieval documents   : "
    f"{len(retrieval_corpus)}"
)


# 6. TEST-SET SAFETY CHECK

print(
    "\n[2] DEVELOPMENT-ONLY "
    "EVALUATION CHECK"
)


wrong_split = []


for question in development_questions:

    question_id = question[
        "question_id"
    ]

    split = (
        evaluation_ground_truth[
            question_id
        ]["split"]
    )

    if split != "development":

        wrong_split.append(
            question_id
        )


print(
    f"Non-development questions found: "
    f"{len(wrong_split)}"
)


if wrong_split:

    raise ValueError(
        "Test-set leakage detected."
    )

else:

    print(
        "PASS: Dense retrieval evaluation "
        "uses development questions only."
    )


# 7. LOAD EMBEDDING MODEL

print("\n[3] LOADING EMBEDDING MODEL")

print(
    f"Model: {MODEL_NAME}"
)


model = SentenceTransformer(
    MODEL_NAME
)


print(
    f"Model device: "
    f"{model.device}"
)

print(
    f"Embedding dimension: "
    f"{model.get_embedding_dimension()}"
)

# 8. PREPARE CORPUS TEXT

corpus_texts = [

    document["text"]

    for document
    in retrieval_corpus
]


print(
    f"\nCorpus passages prepared: "
    f"{len(corpus_texts)}"
)


# 9. CREATE OR LOAD CORPUS EMBEDDINGS

# Each biomedical passage becomes a vector.

# Example:
# "Acute appendicitis..."
#        embedding model
# [0.012, -0.041, 0.093, ...]
# normalize_embeddings=True
# means every vector has length 1.
# Therefore:
# query_embedding @ document_embedding
# is equivalent to cosine similarity.

print(
    "\n[4] CORPUS EMBEDDINGS"
)


if CORPUS_EMBEDDINGS_PATH.exists():

    print(
        "Existing corpus embeddings found."
    )

    corpus_embeddings = np.load(
        CORPUS_EMBEDDINGS_PATH
    )


    # ---------Validate cached embeddings--------

    if (
        corpus_embeddings.shape[0]
        != len(retrieval_corpus)
    ):

        raise ValueError(
            "Cached embedding count does not "
            "match retrieval corpus."
        )


    print(
        "Loaded cached embeddings:"
    )

    print(
        corpus_embeddings.shape
    )


else:

    print(
        "No cached embeddings found."
    )

    print(
        "Encoding biomedical corpus..."
    )


    corpus_embeddings = model.encode(

        corpus_texts,

        batch_size=BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )


    np.save(
        CORPUS_EMBEDDINGS_PATH,
        corpus_embeddings
    )


    print(
        "Corpus embeddings saved:"
    )

    print(
        CORPUS_EMBEDDINGS_PATH
    )


print(
    f"Corpus embedding matrix shape: "
    f"{corpus_embeddings.shape}"
)


# 10. ENCODE DEVELOPMENT QUESTIONS

print(
    "\n[5] ENCODING DEVELOPMENT QUESTIONS"
)


question_texts = [

    question["question"]

    for question
    in development_questions
]


question_embeddings = model.encode(

    question_texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


print(
    f"Question embedding matrix shape: "
    f"{question_embeddings.shape}"
)


# 11. METRIC STORAGE

source_hits = {

    k: 0

    for k in K_VALUES
}


gold_context_recalls = {

    k: []

    for k in K_VALUES
}


reciprocal_ranks = []

no_match_queries = 0

all_results = []


# 12. RUN DENSE RETRIEVAL

# Because vectors are normalized:
# cosine similarity = dot product
# scores = corpus_embeddings @ query_embedding


print(
    "\n[6] RUNNING DENSE RETRIEVAL"
)


total_questions = len(
    development_questions
)


for question_index, question_record in enumerate(
    development_questions
):

    question_id = question_record[
        "question_id"
    ]

    question_text = question_record[
        "question"
    ]


    query_embedding = (
        question_embeddings[
            question_index
        ]
    )


    
    # ------------Calculate semantic similarity against----------
    # every document
    

    similarity_scores = (
        corpus_embeddings
        @ query_embedding
    )

    
    # Get top MAX_K indices
   
    top_indices = np.argpartition(

        similarity_scores,

        -MAX_K

    )[-MAX_K:]


    # Sort those candidates from highest
    # similarity to lowest

    top_indices = top_indices[

        np.argsort(
            similarity_scores[
                top_indices
            ]
        )[::-1]

    ]


    
    # ---------Gold evidence----------
    

    gold_info = (
        evaluation_ground_truth[
            question_id
        ]
    )


    gold_context_ids = set(

        gold_info[
            "gold_context_ids"
        ]
    )


    ranked_documents = []

    first_gold_rank = None


   
    # -------Build ranked result-------
   

    for rank, doc_index in enumerate(
        top_indices,
        start=1
    ):

        document = retrieval_corpus[
            int(doc_index)
        ]


        score = float(
            similarity_scores[
                doc_index
            ]
        )


        is_gold_source = (

            document[
                "source_record_id"
            ]

            ==
            question_id
        )


        is_gold_context = (

            document[
                "document_id"
            ]

            in gold_context_ids
        )


        if (
            is_gold_source
            and first_gold_rank is None
        ):

            first_gold_rank = rank


        ranked_documents.append(
            {

                "rank":
                    rank,

                "document_id":
                    document[
                        "document_id"
                    ],

                "source_record_id":
                    document[
                        "source_record_id"
                    ],

                "score":
                    round(
                        score,
                        6
                    ),

                "section":
                    document[
                        "section"
                    ],

                "is_gold_source":
                    is_gold_source,

                "is_gold_context":
                    is_gold_context,

                "text":
                    document[
                        "text"
                    ]
            }
        )


    # SOURCE HIT@K

    for k in K_VALUES:

        top_k_documents = (
            ranked_documents[:k]
        )


        hit = any(

            document[
                "is_gold_source"
            ]

            for document
            in top_k_documents
        )


        if hit:

            source_hits[k] += 1


        # GOLD CONTEXT RECALL@K

        retrieved_gold_count = sum(

            1

            for document
            in top_k_documents

            if document[
                "is_gold_context"
            ]
        )


        if gold_context_ids:

            recall = (

                retrieved_gold_count

                /
                len(
                    gold_context_ids
                )
            )

        else:

            recall = 0


        gold_context_recalls[
            k
        ].append(
            recall
        )


    # MRR

    if first_gold_rank is not None:

        reciprocal_rank = (

            1

            /
            first_gold_rank
        )

    else:

        reciprocal_rank = 0

        no_match_queries += 1


    reciprocal_ranks.append(
        reciprocal_rank
    )


    # STORE RESULT

    all_results.append(
        {

            "question_id":
                question_id,

            "question":
                question_text,

            "first_gold_rank":
                first_gold_rank,

            "retrieved_documents":
                ranked_documents
        }
    )


    # Progress

    processed = (
        question_index
        + 1
    )


    if processed % 100 == 0:

        print(
            f"Processed "
            f"{processed}"
            f"/{total_questions} "
            f"questions"
        )


# 13. CALCULATE FINAL METRICS

print(
    "\n[7] DENSE RETRIEVAL RESULTS"
)


summary = {

    "retriever":
        "Dense",

    "model":
        MODEL_NAME,

    "number_of_questions":
        total_questions,

    "number_of_documents":
        len(
            retrieval_corpus
        ),

    "source_hit_at_k":
        {},

    "gold_context_recall_at_k":
        {},

    "mrr_at_10":
        None,

    "queries_without_gold_in_top_10":
        no_match_queries
}


for k in K_VALUES:

    hit_rate = (

        source_hits[k]

        /
        total_questions
    )


    mean_context_recall = (

        sum(
            gold_context_recalls[k]
        )

        /
        len(
            gold_context_recalls[k]
        )
    )


    summary[
        "source_hit_at_k"
    ][str(k)] = hit_rate


    summary[
        "gold_context_recall_at_k"
    ][str(k)] = (
        mean_context_recall
    )


mean_reciprocal_rank = (

    sum(
        reciprocal_ranks
    )

    /
    len(
        reciprocal_ranks
    )
)


summary[
    "mrr_at_10"
] = mean_reciprocal_rank



# 14. PRINT RESULTS


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


print("\nRanking Quality")

print("-" * 40)


print(
    f"MRR@10 : "
    f"{mean_reciprocal_rank:.4f}"
)


print(
    f"Queries with no correct source "
    f"in top 10: "
    f"{no_match_queries}"
)

# 15. SAVE RESULT

save_json(
    all_results,
    DENSE_RESULTS_PATH
)

save_json(
    summary,
    DENSE_SUMMARY_PATH
)


print(
    "\n[8] RESULTS SAVED"
)

print(
    DENSE_RESULTS_PATH
)

print(
    DENSE_SUMMARY_PATH
)


# 16. COMPARE WITH BM25

print(
    "\n[9] BM25 VS DENSE COMPARISON"
)


if BM25_SUMMARY_PATH.exists():

    bm25_summary = load_json(
        BM25_SUMMARY_PATH
    )


    print(
        "\n"
        f"{'Metric':<15}"
        f"{'BM25':>12}"
        f"{'Dense':>12}"
    )

    print(
        "-" * 39
    )


    for k in K_VALUES:

        bm25_value = (
            bm25_summary[
                "source_hit_at_k"
            ][str(k)]
        )

        dense_value = (
            summary[
                "source_hit_at_k"
            ][str(k)]
        )


        print(
            f"Hit@{k:<10}"
            f"{bm25_value:>12.4f}"
            f"{dense_value:>12.4f}"
        )


    print(
        "-" * 39
    )


    print(
        f"{'MRR@10':<15}"
        f"{bm25_summary['mrr_at_10']:>12.4f}"
        f"{summary['mrr_at_10']:>12.4f}"
    )


else:

    print(
        "BM25 summary not found."
    )


# 17. CHECK OUR WEEKEND EXAMPLE

# This is the BM25 failure that we discussed.
# We specifically inspect whether semantic retrieval handles it better.


TARGET_QUESTION_ID = (
    "17610439"
)


print(
    "\n[10] WEEKEND QUESTION CHECK"
)


target_result = None


for result in all_results:

    if (
        result[
            "question_id"
        ]
        ==
        TARGET_QUESTION_ID
    ):

        target_result = result

        break


if target_result is not None:

    print(
        "\nQUESTION:"
    )

    print(
        target_result[
            "question"
        ]
    )


    print(
        "\nDense Top 5:"
    )


    for document in (
        target_result[
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
            f"Section: "
            f"{document['section']}"
        )


        print(
            f"Similarity: "
            f"{document['score']}"
        )


        print(
            f"Text: "
            f"{document['text']}"
        )


else:

    print(
        "Target question not found."
    )



# 18. SHOW DENSE RETRIEVAL FAILURES


print(
    "\n[11] EXAMPLE DENSE RETRIEVAL FAILURES"
)


failure_count = 0


for result in all_results:

    top_five = (
        result[
            "retrieved_documents"
        ][:5]
    )


    has_gold_in_top_five = any(

        document[
            "is_gold_source"
        ]

        for document
        in top_five
    )


    if not has_gold_in_top_five:

        failure_count += 1


        print(
            "\n"
            + "="
            * 75
        )


        print(
            f"Question ID: "
            f"{result['question_id']}"
        )


        print(
            f"QUESTION: "
            f"{result['question']}"
        )


        print(
            "\nTop 3 dense results:"
        )


        for document in (
            top_five[:3]
        ):

            print(
                "\n"
                + "-"
                * 60
            )


            print(
                f"Rank "
                f"{document['rank']}"
            )


            print(
                f"Source ID: "
                f"{document['source_record_id']}"
            )


            print(
                f"Similarity: "
                f"{document['score']}"
            )


            print(
                f"Section: "
                f"{document['section']}"
            )


            print(
                f"Text: "
                f"{document['text']}"
            )


        if failure_count == 3:

            break


# 19. COMPLETE

print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 6 COMPLETE"
)

print(
    "=" * 75
)
import ast
import hashlib
import heapq
import json
import math
import re

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer
)


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


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


MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)


BM25_SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "05_bm25_retrieval.py"
)


WEIGHTED_RRF_SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "09_weighted_rrf_tuning.py"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BASIC_RESULTS_PATH = (
    OUTPUT_DIR
    / "basic_dense_test_results.json"
)


ADVANCED_RESULTS_PATH = (
    OUTPUT_DIR
    / "advanced_reranked_test_results.json"
)


BASIC_CONTEXTS_PATH = (
    OUTPUT_DIR
    / "basic_rag_test_contexts.json"
)


ADVANCED_CONTEXTS_PATH = (
    OUTPUT_DIR
    / "advanced_rag_test_contexts.json"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "test_retrieval_summary.json"
)


# 2. FROZEN SETTINGS

DENSE_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


DENSE_BATCH_SIZE = 32


BM25_K1 = 1.5

BM25_B = 0.75


BM25_CANDIDATES = 10

DENSE_CANDIDATES = 10


BM25_WEIGHT = 0.75

DENSE_WEIGHT = 1.0

RRF_K = 60


RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)


FINAL_TOP_K = 5


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

        return json.load(
            file
        )


def save_json(
    data,
    path
):

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


# 4. START

print("=" * 78)

print(
    "SECTION 34 - FINAL TEST RETRIEVAL"
)

print("=" * 78)


# 5. VERIFY FROZEN PROTOCOL

manifest = load_json(
    MANIFEST_PATH
)


print(
    "\n[1] VERIFYING FROZEN PROTOCOL"
)


changed_files = []

missing_files = []


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


    actual_hash = sha256_file(
        path
    )


    if actual_hash != expected_hash:

        changed_files.append(
            relative_path
        )


if missing_files:

    print(
        "\nMissing frozen files:"
    )


    for filename in missing_files:

        print(
            f"  {filename}"
        )


if changed_files:

    print(
        "\nChanged frozen files:"
    )


    for filename in changed_files:

        print(
            f"  {filename}"
        )


if (
    missing_files
    or
    changed_files
):

    raise RuntimeError(
        "Frozen protocol integrity "
        "check failed. "
        "Do not continue with the "
        "final test."
    )


print(
    "PASS: All frozen files remain unchanged."
)


# 6. EXTRACT EXACT FROZEN DEFINITIONS
# We load ONLY selected class/function definitions from the frozen development scripts.
# We do NOT execute the original scripts themselves.


def extract_definitions(
    source_path,
    required_names,
    namespace
):

    source = source_path.read_text(
        encoding="utf-8"
    )


    syntax_tree = ast.parse(
        source,
        filename=str(
            source_path
        )
    )


    selected_nodes = []


    for node in syntax_tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.ClassDef
            )
        ):

            if node.name in required_names:

                selected_nodes.append(
                    node
                )


    found_names = {

        node.name

        for node
        in selected_nodes
    }


    missing_names = (
        set(
            required_names
        )
        -
        found_names
    )


    if missing_names:

        raise RuntimeError(
            f"Could not extract "
            f"{sorted(missing_names)} "
            f"from "
            f"{source_path.name}"
        )


    module = ast.Module(
        body=selected_nodes,
        type_ignores=[]
    )


    ast.fix_missing_locations(
        module
    )


    compiled = compile(
        module,
        filename=str(
            source_path
        ),
        mode="exec"
    )


    exec(
        compiled,
        namespace
    )

# Frozen BM25 definitions

bm25_namespace = {

    "re":
        re,

    "math":
        math,

    "heapq":
        heapq,

    "Counter":
        Counter,

    "defaultdict":
        defaultdict
}


extract_definitions(
    BM25_SOURCE_PATH,
    [
        "tokenize",
        "BM25Retriever"
    ],
    bm25_namespace
)


tokenize = (
    bm25_namespace[
        "tokenize"
    ]
)


BM25Retriever = (
    bm25_namespace[
        "BM25Retriever"
    ]
)

# Frozen Weighted RRF definition
# IMPORTANT FIX:
# weighted_rrf() uses defaultdict internally,
# so defaultdict must exist in this namespace.

rrf_namespace = {

    "defaultdict":
        defaultdict
}


extract_definitions(
    WEIGHTED_RRF_SOURCE_PATH,
    [
        "weighted_rrf"
    ],
    rrf_namespace
)


weighted_rrf = (
    rrf_namespace[
        "weighted_rrf"
    ]
)


print(
    "PASS: Exact frozen BM25 and "
    "weighted-RRF definitions loaded."
)


# 7. LOAD TEST QUESTIONS + RETRIEVAL CORPUS

print(
    "\n[2] LOADING FINAL TEST INPUTS"
)


test_questions = load_json(
    TEST_QUESTIONS_PATH
)


retrieval_corpus = load_json(
    CORPUS_PATH
)


print(
    f"Test questions : "
    f"{len(test_questions)}"
)


print(
    f"Corpus docs    : "
    f"{len(retrieval_corpus)}"
)


if len(test_questions) != 500:

    raise ValueError(
        "Expected exactly 500 "
        "test questions."
    )


if len(retrieval_corpus) != 3358:

    raise ValueError(
        "Expected exactly 3358 "
        "retrieval corpus documents."
    )


# 8. TEST LEAKAGE GUARD

print(
    "\n[3] TEST LEAKAGE GUARD"
)


allowed_question_fields = {
    "question_id",
    "question"
}


for question in test_questions:

    unexpected_fields = (

        set(
            question.keys()
        )

        -

        allowed_question_fields
    )


    if unexpected_fields:

        raise ValueError(
            "Unexpected test-question "
            f"fields found: "
            f"{unexpected_fields}"
        )


print(
    "PASS: Test inputs contain only "
    "question_id and question."
)


print(
    "PASS: evaluation_ground_truth.json "
    "has NOT been loaded."
)


# 9. BUILD FROZEN BM25 INDEX

print(
    "\n[4] BUILDING BM25 INDEX"
)


bm25 = BM25Retriever(
    retrieval_corpus,
    k1=BM25_K1,
    b=BM25_B
)


print(
    f"BM25 k1 : "
    f"{BM25_K1}"
)


print(
    f"BM25 b  : "
    f"{BM25_B}"
)



# 10. LOAD DENSE EMBEDDING MODEL


print(
    "\n[5] LOADING DENSE MODEL"
)


print(
    f"Model : "
    f"{DENSE_MODEL}"
)


dense_model = SentenceTransformer(
    DENSE_MODEL
)


print(
    f"Device : "
    f"{dense_model.device}"
)


print(
    f"Dimension : "
    f"{dense_model.get_embedding_dimension()}"
)


# 11. ENCODE RETRIEVAL CORPUS

print(
    "\n[6] ENCODING RETRIEVAL CORPUS"
)


corpus_texts = [

    document[
        "text"
    ]

    for document
    in retrieval_corpus
]


corpus_embeddings = (
    dense_model.encode(

        corpus_texts,

        batch_size=
            DENSE_BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )
)


print(
    f"Corpus embedding shape : "
    f"{corpus_embeddings.shape}"
)


# 12. ENCODE TEST QUESTIONS

print(
    "\n[7] ENCODING TEST QUESTIONS"
)


question_texts = [

    question[
        "question"
    ]

    for question
    in test_questions
]


question_embeddings = (
    dense_model.encode(

        question_texts,

        batch_size=
            DENSE_BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )
)


print(
    f"Question embedding shape : "
    f"{question_embeddings.shape}"
)


# 13. LOAD CROSS-ENCODER

print(
    "\n[8] LOADING CROSS-ENCODER"
)


print(
    f"Model : "
    f"{RERANKER_MODEL}"
)


reranker = CrossEncoder(
    RERANKER_MODEL
)


print(
    f"Device : "
    f"{reranker.device}"
)


# 14. SERIALISE A RETRIEVED DOCUMENT

def make_retrieval_document(
    document,
    rank,
    score
):

    return {

        "document_id":
            document[
                "document_id"
            ],

        "source_record_id":
            document[
                "source_record_id"
            ],

        "rank":
            rank,

        "score":
            round(
                float(
                    score
                ),
                6
            ),

        "text":
            document[
                "text"
            ],

        "section":
            document.get(
                "section"
            ),

        "year":
            document.get(
                "year"
            ),

        "meshes":
            document.get(
                "meshes"
            )
    }


# 15. DENSE TOP-10
# Exact Section 6 retrieval pattern:
# normalized corpus vectors
# normalized question vector
# dot product = cosine similarity
# np.argpartition top 10
# np.argsort descending


def dense_top_10(
    query_embedding
):

    similarity_scores = (
        corpus_embeddings
        @
        query_embedding
    )


    top_indices = (
        np.argpartition(

            similarity_scores,

            -DENSE_CANDIDATES

        )[
            -DENSE_CANDIDATES:
        ]
    )


    top_indices = top_indices[

        np.argsort(

            similarity_scores[
                top_indices
            ]

        )[
            ::-1
        ]

    ]


    ranked_documents = []


    for rank, doc_index in enumerate(
        top_indices,
        start=1
    ):

        doc_index = int(
            doc_index
        )


        document = (
            retrieval_corpus[
                doc_index
            ]
        )


        score = float(
            similarity_scores[
                doc_index
            ]
        )


        ranked_documents.append(
            make_retrieval_document(

                document=
                    document,

                rank=
                    rank,

                score=
                    score
            )
        )


    return ranked_documents


# 16. BM25 TOP-10

def bm25_top_10(
    question
):

    retrieved = bm25.search(
        question,
        top_k=BM25_CANDIDATES
    )


    ranked_documents = []


    for rank, (
        doc_index,
        score
    ) in enumerate(
        retrieved,
        start=1
    ):

        document = (
            retrieval_corpus[
                doc_index
            ]
        )


        ranked_documents.append(
            make_retrieval_document(

                document=
                    document,

                rank=
                    rank,

                score=
                    score
            )
        )


    return ranked_documents


# 17. GENERATION CONTEXT FORMAT
# source_record_id is deliberately removed here.


def make_generation_contexts(
    ranked_documents
):

    contexts = []


    top_documents = (
        ranked_documents[
            :FINAL_TOP_K
        ]
    )


    for context_number, document in enumerate(
        top_documents,
        start=1
    ):

        contexts.append(
            {

                "label":
                    f"C{context_number}",

                "document_id":
                    document[
                        "document_id"
                    ],

                "text":
                    document[
                        "text"
                    ],

                "section":
                    document.get(
                        "section"
                    ),

                "year":
                    document.get(
                        "year"
                    )
            }
        )


    return contexts


# 18. RUN FINAL TEST RETRIEVAL

print(
    "\n[9] RUNNING FINAL TEST RETRIEVAL"
)


basic_results = []

advanced_results = []


basic_context_records = []

advanced_context_records = []


candidate_counts = []


for index, (
    question_record,
    query_embedding
) in enumerate(

    zip(
        test_questions,
        question_embeddings
    ),

    start=1

):

    question_id = str(
        question_record[
            "question_id"
        ]
    )


    question = (
        question_record[
            "question"
        ]
    )


    # BASIC RAG
    # Dense retrieval top-10.
    # Top-5 becomes generation context.

    dense_documents = dense_top_10(
        query_embedding
    )


    basic_results.append(
        {

            "question_id":
                question_id,

            "question":
                question,

            "retriever":
                "Dense",

            "model":
                DENSE_MODEL,

            "retrieved_documents":
                dense_documents
        }
    )


    basic_context_records.append(
        {

            "question_id":
                question_id,

            "question":
                question,

            "retrieval_system":
                "basic_rag",

            "top_k":
                FINAL_TOP_K,

            "contexts":
                make_generation_contexts(
                    dense_documents
                )
        }
    )


    # ========================================================
    # ADVANCED RAG
    # BM25 top-10
    # +
    # Dense top-10
    # ->
    # weighted RRF
    # ->
    # rerank complete unique union
    # ->
    # final top-5 generation contexts

    bm25_documents = bm25_top_10(
        question
    )


    fused_documents = weighted_rrf(

        bm25_documents,

        dense_documents,

        BM25_WEIGHT,

        DENSE_WEIGHT,

        RRF_K
    )


    candidate_counts.append(
        len(
            fused_documents
        )
    )


    if not fused_documents:

        raise RuntimeError(
            f"No advanced retrieval "
            f"candidates for "
            f"question {question_id}."
        )


    pairs = [

        [
            question,
            document[
                "text"
            ]
        ]

        for document
        in fused_documents
    ]


    reranker_scores = (
        reranker.predict(
            pairs,
            show_progress_bar=False
        )
    )


    scored_candidates = []


    for document, score in zip(
        fused_documents,
        reranker_scores
    ):

        updated_document = dict(
            document
        )


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


    scored_candidates.sort(

        key=lambda document:

            document[
                "reranker_score"
            ],

        reverse=True
    )


    for new_rank, document in enumerate(
        scored_candidates,
        start=1
    ):

        document[
            "rank"
        ] = new_rank


    advanced_results.append(
        {

            "question_id":
                question_id,

            "question":
                question,

            "retriever":
                (
                    "Weighted RRF + "
                    "Cross-Encoder"
                ),

            "candidate_source":
                (
                    "Weighted RRF union "
                    "of BM25 top-10 and "
                    "Dense top-10"
                ),

            "bm25_weight":
                BM25_WEIGHT,

            "dense_weight":
                DENSE_WEIGHT,

            "rrf_k":
                RRF_K,

            "reranker_model":
                RERANKER_MODEL,

            "retrieved_documents":
                scored_candidates
        }
    )


    advanced_context_records.append(
        {

            "question_id":
                question_id,

            "question":
                question,

            "retrieval_system":
                "advanced_rag",

            "top_k":
                FINAL_TOP_K,

            "contexts":
                make_generation_contexts(
                    scored_candidates
                )
        }
    )


    if index % 50 == 0:

        print(
            f"Processed "
            f"{index}/500"
        )


# 19. STRUCTURAL VALIDATION

print(
    "\n[10] STRUCTURAL VALIDATION"
)


if len(
    basic_results
) != 500:

    raise ValueError(
        "Basic retrieval does not "
        "contain 500 questions."
    )


if len(
    advanced_results
) != 500:

    raise ValueError(
        "Advanced retrieval does not "
        "contain 500 questions."
    )


if len(
    basic_context_records
) != 500:

    raise ValueError(
        "Basic generation contexts "
        "do not contain 500 questions."
    )


if len(
    advanced_context_records
) != 500:

    raise ValueError(
        "Advanced generation contexts "
        "do not contain 500 questions."
    )


basic_ids = {

    record[
        "question_id"
    ]

    for record
    in basic_results
}


advanced_ids = {

    record[
        "question_id"
    ]

    for record
    in advanced_results
}


test_ids = {

    str(
        record[
            "question_id"
        ]
    )

    for record
    in test_questions
}


if basic_ids != test_ids:

    raise ValueError(
        "Basic retrieval IDs do not "
        "match test-question IDs."
    )


if advanced_ids != test_ids:

    raise ValueError(
        "Advanced retrieval IDs do not "
        "match test-question IDs."
    )


for record in basic_context_records:

    if len(
        record[
            "contexts"
        ]
    ) != FINAL_TOP_K:

        raise ValueError(
            "Basic RAG record does not "
            "contain exactly 5 contexts."
        )


for record in advanced_context_records:

    if len(
        record[
            "contexts"
        ]
    ) != FINAL_TOP_K:

        raise ValueError(
            "Advanced RAG record does not "
            "contain exactly 5 contexts."
        )


print(
    "PASS: Basic retrieval = "
    "500 questions."
)


print(
    "PASS: Advanced retrieval = "
    "500 questions."
)


print(
    "PASS: Basic/Advanced IDs "
    "match the frozen test IDs."
)


print(
    "PASS: Every generation record "
    "contains exactly 5 contexts."
)


# 20. GENERATION-CONTEXT LEAKAGE CHECK

print(
    "\n[11] GENERATION-CONTEXT LEAKAGE CHECK"
)


forbidden_fields = {

    "source_record_id",

    "gold_decision",

    "reference_answer",

    "gold_context_ids",

    "final_decision",

    "LONG_ANSWER",

    "reasoning_required_pred",

    "reasoning_free_pred"
}


for collection in [

    basic_context_records,

    advanced_context_records

]:

    for record in collection:

        for context in record[
            "contexts"
        ]:

            overlap = (

                forbidden_fields

                &

                set(
                    context.keys()
                )
            )


            if overlap:

                raise ValueError(
                    "Forbidden fields found "
                    "inside generation "
                    f"context: {overlap}"
                )


print(
    "PASS: No source_record_id or "
    "evaluation-only fields appear "
    "in generation contexts."
)


# 21. ADVANCED CANDIDATE VALIDATION

candidate_counts_array = np.array(
    candidate_counts,
    dtype=float
)


if candidate_counts_array.size != 500:

    raise ValueError(
        "Expected 500 advanced "
        "candidate counts."
    )


if (
    candidate_counts_array.min()
    <
    10
):

    raise ValueError(
        "Advanced candidate union "
        "contains fewer than 10 "
        "documents for at least "
        "one question."
    )


if (
    candidate_counts_array.max()
    >
    20
):

    raise ValueError(
        "Advanced candidate union "
        "contains more than 20 "
        "documents."
    )


# 22. SUMMARY

summary = {

    "split":
        "test",

    "number_of_questions":
        500,

    "evaluation_ground_truth_loaded":
        False,

    "retrieval_metrics_calculated":
        False,

    "basic_rag": {

        "retriever":
            "Dense",

        "model":
            DENSE_MODEL,

        "batch_size":
            DENSE_BATCH_SIZE,

        "normalized_embeddings":
            True,

        "similarity":
            "dot product / cosine",

        "candidate_depth_saved":
            DENSE_CANDIDATES,

        "generation_top_k":
            FINAL_TOP_K
    },

    "advanced_rag": {

        "bm25_k1":
            BM25_K1,

        "bm25_b":
            BM25_B,

        "bm25_candidate_depth":
            BM25_CANDIDATES,

        "dense_candidate_depth":
            DENSE_CANDIDATES,

        "bm25_weight":
            BM25_WEIGHT,

        "dense_weight":
            DENSE_WEIGHT,

        "rrf_k":
            RRF_K,

        "reranker_model":
            RERANKER_MODEL,

        "generation_top_k":
            FINAL_TOP_K,

        "candidate_union_min":
            int(
                candidate_counts_array.min()
            ),

        "candidate_union_max":
            int(
                candidate_counts_array.max()
            ),

        "candidate_union_mean":
            float(
                candidate_counts_array.mean()
            )
    }
}



# 23. SAVE FINAL TEST RETRIEVAL


save_json(
    basic_results,
    BASIC_RESULTS_PATH
)


save_json(
    advanced_results,
    ADVANCED_RESULTS_PATH
)


save_json(
    basic_context_records,
    BASIC_CONTEXTS_PATH
)


save_json(
    advanced_context_records,
    ADVANCED_CONTEXTS_PATH
)


save_json(
    summary,
    SUMMARY_PATH
)



# 24. FINAL TEST RETRIEVAL SUMMARY


print(
    "\n[12] FINAL TEST RETRIEVAL SUMMARY"
)


print(
    "Basic RAG"
)


print(
    f"  Questions       : "
    f"{len(basic_results)}"
)


print(
    f"  Dense saved     : "
    f"top-{DENSE_CANDIDATES}"
)


print(
    f"  Generation      : "
    f"top-{FINAL_TOP_K}"
)


print(
    "\nAdvanced RAG"
)


print(
    f"  Questions       : "
    f"{len(advanced_results)}"
)


print(
    f"  BM25 candidates : "
    f"top-{BM25_CANDIDATES}"
)


print(
    f"  Dense candidates: "
    f"top-{DENSE_CANDIDATES}"
)


print(
    f"  Union minimum   : "
    f"{int(candidate_counts_array.min())}"
)


print(
    f"  Union maximum   : "
    f"{int(candidate_counts_array.max())}"
)


print(
    f"  Union mean      : "
    f"{candidate_counts_array.mean():.2f}"
)


print(
    f"  Final contexts  : "
    f"top-{FINAL_TOP_K}"
)


print(
    "\nRetrieval metrics calculated : NO"
)


print(
    "Test ground truth loaded     : NO"
)


# 25. FILES SAVED

print(
    "\n[13] FILES SAVED"
)


print(
    BASIC_RESULTS_PATH
)


print(
    ADVANCED_RESULTS_PATH
)


print(
    BASIC_CONTEXTS_PATH
)


print(
    ADVANCED_CONTEXTS_PATH
)


print(
    SUMMARY_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 34 COMPLETE - "
    "FINAL TEST RETRIEVAL FROZEN"
)


print(
    "=" * 78
)
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "retrieval"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

GENERATION_DIR = (
    PROJECT_ROOT / "data" / "generation"
)

GENERATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DENSE_RESULTS_PATH = (
    RESULTS_DIR / "dense_dev_results.json"
)

ADVANCED_RESULTS_PATH = (
    RESULTS_DIR / "reranked_dev_results.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR / "evaluation_ground_truth.json"
)

DEVELOPMENT_QUESTIONS_PATH = (
    PROCESSED_DIR / "development_questions.json"
)


BASIC_CONTEXT_OUTPUT = (
    GENERATION_DIR / "basic_rag_dev_contexts.json"
)

ADVANCED_CONTEXT_OUTPUT = (
    GENERATION_DIR / "advanced_rag_dev_contexts.json"
)

EXPERIMENT_CONFIG_OUTPUT = (
    GENERATION_DIR / "experiment_config.json"
)


# 2. SETTINGS - NOW FROZEN

TOP_K = 5

BASIC_RETRIEVER = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BM25_WEIGHT = 0.75
DENSE_WEIGHT = 1.0
RRF_K = 60

RERANKER = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


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
print("SECTION 11 - FREEZE RETRIEVAL FOR GENERATION")
print("=" * 75)


dense_results = load_json(
    DENSE_RESULTS_PATH
)

advanced_results = load_json(
    ADVANCED_RESULTS_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)

development_questions = load_json(
    DEVELOPMENT_QUESTIONS_PATH
)


print("\n[1] DATA LOADED")

print(
    f"Development questions : "
    f"{len(development_questions)}"
)

print(
    f"Dense results         : "
    f"{len(dense_results)}"
)

print(
    f"Advanced results      : "
    f"{len(advanced_results)}"
)


# 5. INDEX BY QUESTION ID

dense_by_id = {
    item["question_id"]: item
    for item in dense_results
}

advanced_by_id = {
    item["question_id"]: item
    for item in advanced_results
}


# 6. SAFETY CHECK

print(
    "\n[2] DEVELOPMENT SAFETY CHECK"
)


wrong_split = []


for question in development_questions:

    question_id = question[
        "question_id"
    ]

    if (
        ground_truth[
            question_id
        ]["split"]
        != "development"
    ):

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
    "PASS: Only development questions "
    "are being prepared."
)


# 7. PREPARE BASIC RAG CONTEXTS

basic_rag_contexts = []


for question in development_questions:

    question_id = question[
        "question_id"
    ]

    result = dense_by_id[
        question_id
    ]


    top_documents = (
        result[
            "retrieved_documents"
        ][:TOP_K]
    )


    contexts = []


    for document in top_documents:

        contexts.append(
            {
                "rank":
                    document["rank"],

                "document_id":
                    document["document_id"],

                "section":
                    document["section"],

                "text":
                    document["text"]
            }
        )


    basic_rag_contexts.append(
        {
            "question_id":
                question_id,

            "question":
                question["question"],

            "retrieval_system":
                "dense",

            "top_k":
                TOP_K,

            "contexts":
                contexts
        }
    )


# 8. PREPARE ADVANCED RAG CONTEXTS

advanced_rag_contexts = []


for question in development_questions:

    question_id = question[
        "question_id"
    ]

    result = advanced_by_id[
        question_id
    ]


    top_documents = (
        result[
            "retrieved_documents"
        ][:TOP_K]
    )


    contexts = []


    for document in top_documents:

        contexts.append(
            {
                "rank":
                    document["rank"],

                "document_id":
                    document["document_id"],

                "section":
                    document["section"],

                "text":
                    document["text"]
            }
        )


    advanced_rag_contexts.append(
        {
            "question_id":
                question_id,

            "question":
                question["question"],

            "retrieval_system":
                "weighted_rrf_crossencoder",

            "top_k":
                TOP_K,

            "contexts":
                contexts
        }
    )


# 9. EXPERIMENT CONFIGURATION

experiment_config = {

    "task":
        "Biomedical question answering "
        "with hallucination reduction",

    "development_questions":
        500,

    "test_questions":
        500,

    "generation_top_k":
        TOP_K,

    "systems": {

        "baseline_llm": {
            "retrieval":
                None
        },

        "basic_rag": {

            "retrieval":
                "dense",

            "embedding_model":
                BASIC_RETRIEVER,

            "top_k":
                TOP_K
        },

        "advanced_rag": {

            "retrieval":
                "BM25 + Dense + Weighted RRF "
                "+ Cross-Encoder",

            "bm25_weight":
                BM25_WEIGHT,

            "dense_weight":
                DENSE_WEIGHT,

            "rrf_k":
                RRF_K,

            "reranker":
                RERANKER,

            "top_k":
                TOP_K
        },

        "gold_context_control": {

            "retrieval":
                "gold contexts",

            "purpose":
                "Estimate generation performance "
                "when correct evidence is available."
        }
    },

    "frozen": True
}


# 10. SAVE

save_json(
    basic_rag_contexts,
    BASIC_CONTEXT_OUTPUT
)

save_json(
    advanced_rag_contexts,
    ADVANCED_CONTEXT_OUTPUT
)

save_json(
    experiment_config,
    EXPERIMENT_CONFIG_OUTPUT
)


# 11. VALIDATION

print(
    "\n[3] OUTPUT COUNTS"
)

print(
    f"Basic RAG records    : "
    f"{len(basic_rag_contexts)}"
)

print(
    f"Advanced RAG records : "
    f"{len(advanced_rag_contexts)}"
)


basic_bad = sum(
    1
    for item in basic_rag_contexts
    if len(item["contexts"]) != TOP_K
)


advanced_bad = sum(
    1
    for item in advanced_rag_contexts
    if len(item["contexts"]) != TOP_K
)


print(
    f"Basic records without {TOP_K} contexts    : "
    f"{basic_bad}"
)

print(
    f"Advanced records without {TOP_K} contexts : "
    f"{advanced_bad}"
)


# 12. LEAKAGE CHECK

print(
    "\n[4] GENERATION INPUT LEAKAGE CHECK"
)


forbidden_fields = {
    "gold_decision",
    "reference_answer",
    "LONG_ANSWER",
    "final_decision"
}


violations = []


for system_name, dataset in [

    (
        "basic",
        basic_rag_contexts
    ),

    (
        "advanced",
        advanced_rag_contexts
    )
]:

    for item in dataset:

        if (
            forbidden_fields
            &
            set(item.keys())
        ):

            violations.append(
                (
                    system_name,
                    item["question_id"]
                )
            )


print(
    f"Leakage violations: "
    f"{len(violations)}"
)


if violations:

    raise ValueError(
        "Ground-truth leakage detected."
    )


print(
    "PASS: Generation inputs contain "
    "no reference answers or decisions."
)


# 13. SHOW ONE COMPARISON

TARGET_ID = "17610439"


print(
    "\n[5] WEEKEND QUESTION - BASIC VS ADVANCED"
)


basic_target = next(

    item

    for item in basic_rag_contexts

    if item[
        "question_id"
    ] == TARGET_ID
)


advanced_target = next(

    item

    for item in advanced_rag_contexts

    if item[
        "question_id"
    ] == TARGET_ID
)


print(
    "\nQUESTION:"
)

print(
    basic_target[
        "question"
    ]
)


print(
    "\nBASIC RAG TOP 5:"
)


for context in basic_target[
    "contexts"
]:

    print(
        "\n"
        + "-"
        * 60
    )

    print(
        f"Rank {context['rank']} "
        f"[{context['section']}]"
    )

    print(
        context["text"]
    )


print(
    "\nADVANCED RAG TOP 5:"
)


for context in advanced_target[
    "contexts"
]:

    print(
        "\n"
        + "-"
        * 60
    )

    print(
        f"Rank {context['rank']} "
        f"[{context['section']}]"
    )

    print(
        context["text"]
    )


# 14. COMPLETE

print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 11 COMPLETE"
)

print(
    "=" * 75
)
import hashlib
import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


ORIGINAL_FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "frozen_protocol"
    / "final_protocol_manifest.json"
)


FINAL_TEST_FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "final_test_artifact_manifest.json"
)


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


GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_ground_truth.json"
)


BASIC_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "basic_dense_test_results.json"
)


ADVANCED_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "retrieval"
    / "advanced_reranked_test_results.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DETAIL_PATH = (
    OUTPUT_DIR
    / "final_test_retrieval_results.json"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_test_retrieval_summary.json"
)


# 2. SETTINGS

K_VALUES = [
    1,
    3,
    5,
    10
]


SYSTEM_FILES = {

    "basic_rag":
        BASIC_RESULTS_PATH,

    "advanced_rag":
        ADVANCED_RESULTS_PATH
}


SYSTEM_DISPLAY_NAMES = {

    "basic_rag":
        "Basic RAG (Dense)",

    "advanced_rag":
        "Advanced RAG (Hybrid + Reranker)"
}


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


def relative_path(path):

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/"
    )


def find_manifest_hash(
    manifest,
    suffix
):

    candidate_sections = [
        "sha256",
        "file_hashes"
    ]


    for section_name in candidate_sections:

        hashes = manifest.get(
            section_name,
            {}
        )


        for relative_name, digest in (
            hashes.items()
        ):

            normalized = (
                relative_name
                .replace(
                    "\\",
                    "/"
                )
            )


            if normalized.endswith(
                suffix
            ):

                return (
                    relative_name,
                    digest
                )


    return (
        None,
        None
    )


def verify_frozen_file(
    path,
    manifests
):

    suffix = relative_path(
        path
    )


    for manifest in manifests:

        key, expected_hash = (
            find_manifest_hash(
                manifest,
                suffix
            )
        )


        if expected_hash is None:

            continue


        actual_hash = sha256_file(
            path
        )


        if actual_hash != expected_hash:

            raise RuntimeError(
                "Frozen file changed: "
                f"{suffix}"
            )


        return True


    return False


# 4. START

print("=" * 78)

print(
    "SECTION 44 - FINAL TEST RETRIEVAL EVALUATION"
)

print("=" * 78)


# 5. VERIFY FROZEN RETRIEVAL OUTPUTS

print(
    "\n[1] VERIFYING FROZEN RETRIEVAL ARTIFACTS"
)


original_manifest = load_json(
    ORIGINAL_FREEZE_MANIFEST
)


final_manifest = load_json(
    FINAL_TEST_FREEZE_MANIFEST
)


manifests = [
    final_manifest,
    original_manifest
]


required_frozen_files = [

    BASIC_RESULTS_PATH,
    ADVANCED_RESULTS_PATH,
    CORPUS_PATH
]


for path in required_frozen_files:

    verified = verify_frozen_file(
        path,
        manifests
    )


    if not verified:

        raise RuntimeError(
            "Could not locate a frozen hash "
            f"for {relative_path(path)}"
        )


    print(
        "PASS: "
        f"{relative_path(path)}"
    )


print(
    "PASS: Frozen retrieval artifacts "
    "remain unchanged."
)


# 6. LOAD DATA

print(
    "\n[2] LOADING FINAL TEST RETRIEVAL DATA"
)


test_questions = load_json(
    TEST_QUESTIONS_PATH
)


corpus = load_json(
    CORPUS_PATH
)


ground_truth = load_json(
    GROUND_TRUTH_PATH
)


basic_results = load_json(
    BASIC_RESULTS_PATH
)


advanced_results = load_json(
    ADVANCED_RESULTS_PATH
)


print(
    f"Test questions     : "
    f"{len(test_questions)}"
)


print(
    f"Corpus documents   : "
    f"{len(corpus)}"
)


print(
    f"Basic results      : "
    f"{len(basic_results)}"
)


print(
    f"Advanced results   : "
    f"{len(advanced_results)}"
)


if len(
    test_questions
) != 500:

    raise ValueError(
        "Expected 500 test questions."
    )


if len(
    basic_results
) != 500:

    raise ValueError(
        "Expected 500 Basic retrieval results."
    )


if len(
    advanced_results
) != 500:

    raise ValueError(
        "Expected 500 Advanced retrieval results."
    )


# 7. INDEX CORPUS

print(
    "\n[3] INDEXING CORPUS AND TEST GROUND TRUTH"
)


corpus_by_id = {}


for document in corpus:

    document_id = str(
        document[
            "document_id"
        ]
    )


    if document_id in corpus_by_id:

        raise ValueError(
            f"Duplicate corpus document ID: "
            f"{document_id}"
        )


    corpus_by_id[
        document_id
    ] = document


print(
    f"Unique corpus IDs : "
    f"{len(corpus_by_id)}"
)


test_question_ids = [

    str(
        item[
            "question_id"
        ]
    )

    for item
    in test_questions
]


if len(
    set(
        test_question_ids
    )
) != 500:

    raise ValueError(
        "Test question IDs are not unique."
    )


test_ground_truth = {}


for question_id in test_question_ids:

    if question_id not in ground_truth:

        raise KeyError(
            f"Missing evaluation ground truth "
            f"for question {question_id}."
        )


    record = (
        ground_truth[
            question_id
        ]
    )


    if (
        record[
            "split"
        ]
        !=
        "test"
    ):

        raise ValueError(
            f"Question {question_id} "
            "is not labelled as test."
        )


    gold_context_ids = [

        str(
            context_id
        )

        for context_id
        in record[
            "gold_context_ids"
        ]
    ]


    if not gold_context_ids:

        raise ValueError(
            f"Question {question_id} "
            "has no gold contexts."
        )


    missing_gold_contexts = [

        context_id

        for context_id
        in gold_context_ids

        if context_id not in corpus_by_id
    ]


    if missing_gold_contexts:

        raise KeyError(
            f"Gold contexts missing from corpus "
            f"for {question_id}: "
            f"{missing_gold_contexts}"
        )


    gold_source_ids = {

        str(
            corpus_by_id[
                context_id
            ][
                "source_record_id"
            ]
        )

        for context_id
        in gold_context_ids
    }


    test_ground_truth[
        question_id
    ] = {

        "gold_context_ids":
            gold_context_ids,

        "gold_source_ids":
            sorted(
                gold_source_ids
            )
    }


print(
    "PASS: Test ground truth and "
    "gold contexts are valid."
)


# 8. INDEX RETRIEVAL RESULTS

def index_retrieval_results(
    records,
    system_name
):

    indexed = {}


    for record in records:

        question_id = str(
            record[
                "question_id"
            ]
        )


        if question_id in indexed:

            raise ValueError(
                f"Duplicate {system_name} "
                f"question ID: {question_id}"
            )


        documents = (
            record[
                "retrieved_documents"
            ]
        )


        if len(
            documents
        ) < 10:

            raise ValueError(
                f"{system_name} question "
                f"{question_id} has fewer "
                "than 10 ranked documents."
            )


        # Sort explicitly by frozen rank.
        ranked_documents = sorted(
            documents,
            key=lambda item:
                int(
                    item[
                        "rank"
                    ]
                )
        )


        ranks = [

            int(
                item[
                    "rank"
                ]
            )

            for item
            in ranked_documents
        ]


        expected_ranks = list(
            range(
                1,
                len(
                    ranked_documents
                )
                +
                1
            )
        )


        if ranks != expected_ranks:

            raise ValueError(
                f"Non-contiguous ranking in "
                f"{system_name} / {question_id}."
            )


        seen_document_ids = set()


        for document in ranked_documents:

            document_id = str(
                document[
                    "document_id"
                ]
            )


            if document_id in seen_document_ids:

                raise ValueError(
                    f"Duplicate retrieved document "
                    f"in {system_name} / "
                    f"{question_id}: "
                    f"{document_id}"
                )


            seen_document_ids.add(
                document_id
            )


            if document_id not in corpus_by_id:

                raise KeyError(
                    f"Retrieved document "
                    f"{document_id} missing "
                    "from corpus."
                )


            stored_source = str(
                document[
                    "source_record_id"
                ]
            )


            corpus_source = str(
                corpus_by_id[
                    document_id
                ][
                    "source_record_id"
                ]
            )


            if stored_source != corpus_source:

                raise ValueError(
                    f"source_record_id mismatch "
                    f"for {document_id}."
                )


        indexed[
            question_id
        ] = {

            "metadata":
                {
                    key:
                        value

                    for key, value
                    in record.items()

                    if key
                    not in {
                        "retrieved_documents",
                        "question"
                    }
                },

            "retrieved_documents":
                ranked_documents
        }


    return indexed


basic_by_id = index_retrieval_results(
    basic_results,
    "basic_rag"
)


advanced_by_id = index_retrieval_results(
    advanced_results,
    "advanced_rag"
)


for system_name, index in [

    (
        "basic_rag",
        basic_by_id
    ),

    (
        "advanced_rag",
        advanced_by_id
    )

]:

    if (
        set(
            index.keys()
        )
        !=
        set(
            test_question_ids
        )
    ):

        raise ValueError(
            f"{system_name} question IDs "
            "do not exactly match the "
            "500-question test set."
        )


print(
    "\n[4] RETRIEVAL RESULT VALIDATION"
)


basic_lengths = [

    len(
        basic_by_id[
            question_id
        ][
            "retrieved_documents"
        ]
    )

    for question_id
    in test_question_ids
]


advanced_lengths = [

    len(
        advanced_by_id[
            question_id
        ][
            "retrieved_documents"
        ]
    )

    for question_id
    in test_question_ids
]


print(
    "Basic candidate counts    : "
    f"min={min(basic_lengths)} "
    f"max={max(basic_lengths)} "
    f"mean={sum(basic_lengths) / len(basic_lengths):.2f}"
)


print(
    "Advanced candidate counts : "
    f"min={min(advanced_lengths)} "
    f"max={max(advanced_lengths)} "
    f"mean={sum(advanced_lengths) / len(advanced_lengths):.2f}"
)


print(
    "PASS: Both systems contain "
    "valid frozen top-10 rankings."
)


# 9. METRIC DEFINITIONS


# PRIMARY RETRIEVAL METRIC

# Source Hit@k:
#   1 if at least one of the first k retrieved passages belongs to the same source record as the benchmark question.

# MRR@10: reciprocal rank of the first correct source passage within the top 10; otherwise 0.

# SUPPLEMENTARY METRIC
# Gold-context Recall@k: number of attached gold context passages retrieved in top k / total attached gold context passages.

# 10. EVALUATE ONE SYSTEM

def evaluate_system(
    system_name,
    retrieval_index
):

    per_question = []


    hit_counts = {

        str(k):
            0

        for k
        in K_VALUES
    }


    recall_sums = {

        str(k):
            0.0

        for k
        in K_VALUES
    }


    reciprocal_rank_sum = 0.0


    source_misses_at_10 = []


    for question_id in test_question_ids:

        ground = (
            test_ground_truth[
                question_id
            ]
        )


        gold_context_ids = set(
            ground[
                "gold_context_ids"
            ]
        )


        gold_source_ids = set(
            ground[
                "gold_source_ids"
            ]
        )


        ranked_documents = (
            retrieval_index[
                question_id
            ][
                "retrieved_documents"
            ]
        )


        first_correct_source_rank = None


        for document in (
            ranked_documents[
                :10
            ]
        ):

            if (
                str(
                    document[
                        "source_record_id"
                    ]
                )
                in
                gold_source_ids
            ):

                first_correct_source_rank = int(
                    document[
                        "rank"
                    ]
                )

                break


        if first_correct_source_rank is None:

            reciprocal_rank = 0.0

            source_misses_at_10.append(
                question_id
            )


        else:

            reciprocal_rank = (
                1.0
                /
                first_correct_source_rank
            )


        reciprocal_rank_sum += (
            reciprocal_rank
        )


        question_hits = {}

        question_recalls = {}


        for k in K_VALUES:

            top_k = (
                ranked_documents[
                    :k
                ]
            )


            top_k_source_ids = {

                str(
                    document[
                        "source_record_id"
                    ]
                )

                for document
                in top_k
            }


            source_hit = int(
                bool(
                    top_k_source_ids
                    &
                    gold_source_ids
                )
            )


            top_k_document_ids = {

                str(
                    document[
                        "document_id"
                    ]
                )

                for document
                in top_k
            }


            retrieved_gold_contexts = (
                top_k_document_ids
                &
                gold_context_ids
            )


            gold_context_recall = (
                len(
                    retrieved_gold_contexts
                )
                /
                len(
                    gold_context_ids
                )
            )


            question_hits[
                f"hit_at_{k}"
            ] = source_hit


            question_recalls[
                f"recall_at_{k}"
            ] = gold_context_recall


            hit_counts[
                str(k)
            ] += source_hit


            recall_sums[
                str(k)
            ] += (
                gold_context_recall
            )


        per_question.append(
            {

                "question_id":
                    question_id,

                "gold_context_count":
                    len(
                        gold_context_ids
                    ),

                "gold_source_ids":
                    sorted(
                        gold_source_ids
                    ),

                "first_correct_source_rank_at_10":
                    first_correct_source_rank,

                "reciprocal_rank_at_10":
                    reciprocal_rank,

                "source_hits":
                    question_hits,

                "gold_context_recall":
                    question_recalls
            }
        )


    number_questions = len(
        test_question_ids
    )


    source_hit_rates = {

        f"hit_at_{k}":
            (
                hit_counts[
                    str(k)
                ]
                /
                number_questions
            )

        for k
        in K_VALUES
    }


    gold_context_recall_rates = {

        f"recall_at_{k}":
            (
                recall_sums[
                    str(k)
                ]
                /
                number_questions
            )

        for k
        in K_VALUES
    }


    mrr_at_10 = (
        reciprocal_rank_sum
        /
        number_questions
    )


    candidate_lengths = [

        len(
            retrieval_index[
                question_id
            ][
                "retrieved_documents"
            ]
        )

        for question_id
        in test_question_ids
    ]


    summary = {

        "system":
            system_name,

        "questions":
            number_questions,

        "candidate_count": {

            "min":
                min(
                    candidate_lengths
                ),

            "max":
                max(
                    candidate_lengths
                ),

            "mean":
                (
                    sum(
                        candidate_lengths
                    )
                    /
                    len(
                        candidate_lengths
                    )
                )
        },

        "source_hit_counts": {

            f"hit_at_{k}":
                hit_counts[
                    str(k)
                ]

            for k
            in K_VALUES
        },

        "source_hit_rates":
            source_hit_rates,

        "mrr_at_10":
            mrr_at_10,

        "source_misses_at_10":
            len(
                source_misses_at_10
            ),

        "source_miss_question_ids_at_10":
            source_misses_at_10,

        "gold_context_recall":
            gold_context_recall_rates
    }


    return (
        summary,
        per_question
    )


# 11. RUN FINAL RETRIEVAL EVALUATION

print(
    "\n[5] CALCULATING FINAL RETRIEVAL METRICS"
)


basic_summary, basic_question_results = (
    evaluate_system(
        "basic_rag",
        basic_by_id
    )
)


advanced_summary, advanced_question_results = (
    evaluate_system(
        "advanced_rag",
        advanced_by_id
    )
)


system_summaries = {

    "basic_rag":
        basic_summary,

    "advanced_rag":
        advanced_summary
}


# 12. PAIRED SOURCE-HIT COMPLEMENTARITY

print(
    "\n[6] PAIRED SOURCE-HIT COMPLEMENTARITY"
)


basic_question_index = {

    record[
        "question_id"
    ]:
        record

    for record
    in basic_question_results
}


advanced_question_index = {

    record[
        "question_id"
    ]:
        record

    for record
    in advanced_question_results
}


paired_hit_analysis = {}


for k in K_VALUES:

    both_hit = 0

    basic_only = 0

    advanced_only = 0

    neither = 0


    for question_id in test_question_ids:

        basic_hit = bool(

            basic_question_index[
                question_id
            ][
                "source_hits"
            ][
                f"hit_at_{k}"
            ]
        )


        advanced_hit = bool(

            advanced_question_index[
                question_id
            ][
                "source_hits"
            ][
                f"hit_at_{k}"
            ]
        )


        if (
            basic_hit
            and
            advanced_hit
        ):

            both_hit += 1


        elif (
            basic_hit
            and
            not advanced_hit
        ):

            basic_only += 1


        elif (
            advanced_hit
            and
            not basic_hit
        ):

            advanced_only += 1


        else:

            neither += 1


    paired_hit_analysis[
        f"hit_at_{k}"
    ] = {

        "both_hit":
            both_hit,

        "basic_only":
            basic_only,

        "advanced_only":
            advanced_only,

        "neither":
            neither
    }


    print(
        f"@{k:<2} "
        f"both={both_hit:<3} "
        f"basic_only={basic_only:<3} "
        f"advanced_only={advanced_only:<3} "
        f"neither={neither:<3}"
    )


# 13. ADVANCED - BASIC DESCRIPTIVE DIFFERENCES

differences = {

    "source_hit_rates":
        {},

    "gold_context_recall":
        {},

    "mrr_at_10":
        (
            advanced_summary[
                "mrr_at_10"
            ]
            -
            basic_summary[
                "mrr_at_10"
            ]
        )
}


for k in K_VALUES:

    differences[
        "source_hit_rates"
    ][
        f"hit_at_{k}"
    ] = (

        advanced_summary[
            "source_hit_rates"
        ][
            f"hit_at_{k}"
        ]

        -

        basic_summary[
            "source_hit_rates"
        ][
            f"hit_at_{k}"
        ]
    )


    differences[
        "gold_context_recall"
    ][
        f"recall_at_{k}"
    ] = (

        advanced_summary[
            "gold_context_recall"
        ][
            f"recall_at_{k}"
        ]

        -

        basic_summary[
            "gold_context_recall"
        ][
            f"recall_at_{k}"
        ]
    )


# 14. PRINT SYSTEM RESULTS

print(
    "\n[7] FINAL TEST RETRIEVAL RESULTS"
)


for system_name in [

    "basic_rag",
    "advanced_rag"

]:

    summary = (
        system_summaries[
            system_name
        ]
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        SYSTEM_DISPLAY_NAMES[
            system_name
        ]
    )


    print(
        "=" * 72
    )


    print(
        "\nSource Hit:"
    )


    for k in K_VALUES:

        count = (
            summary[
                "source_hit_counts"
            ][
                f"hit_at_{k}"
            ]
        )


        rate = (
            summary[
                "source_hit_rates"
            ][
                f"hit_at_{k}"
            ]
        )


        print(
            f"  Hit@{k:<2} : "
            f"{count:>3}/500 "
            f"({rate:.4f})"
        )


    print(
        f"\n  MRR@10 : "
        f"{summary['mrr_at_10']:.4f}"
    )


    print(
        f"  Misses@10 : "
        f"{summary['source_misses_at_10']}"
    )


    print(
        "\nGold-context Recall:"
    )


    for k in K_VALUES:

        rate = (
            summary[
                "gold_context_recall"
            ][
                f"recall_at_{k}"
            ]
        )


        print(
            f"  Recall@{k:<2} : "
            f"{rate:.4f}"
        )


# 15. PRINT BASIC VS ADVANCED DIFFERENCE

print(
    "\n[8] ADVANCED - BASIC RETRIEVAL DIFFERENCE"
)


print(
    "\nSource Hit difference:"
)


for k in K_VALUES:

    difference = (
        differences[
            "source_hit_rates"
        ][
            f"hit_at_{k}"
        ]
    )


    print(
        f"  Hit@{k:<2} : "
        f"{difference:+.4f} "
        f"({difference * 100:+.2f} pp)"
    )


print(
    "\nGold-context Recall difference:"
)


for k in K_VALUES:

    difference = (
        differences[
            "gold_context_recall"
        ][
            f"recall_at_{k}"
        ]
    )


    print(
        f"  Recall@{k:<2} : "
        f"{difference:+.4f} "
        f"({difference * 100:+.2f} pp)"
    )


print(
    f"\nMRR@10 difference : "
    f"{differences['mrr_at_10']:+.4f}"
)


# 16. COMPACT SCORECARD

print(
    "\n[9] RETRIEVAL SCORECARD"
)


print(
    f"{'Metric':<20}"
    f"{'Basic':>12}"
    f"{'Advanced':>12}"
    f"{'Adv-Basic':>14}"
)


print(
    "-" * 58
)


for k in K_VALUES:

    basic_value = (
        basic_summary[
            "source_hit_rates"
        ][
            f"hit_at_{k}"
        ]
    )


    advanced_value = (
        advanced_summary[
            "source_hit_rates"
        ][
            f"hit_at_{k}"
        ]
    )


    print(
        f"{f'Source Hit@{k}':<20}"
        f"{basic_value:>12.4f}"
        f"{advanced_value:>12.4f}"
        f"{advanced_value - basic_value:>+14.4f}"
    )


print(
    f"{'MRR@10':<20}"
    f"{basic_summary['mrr_at_10']:>12.4f}"
    f"{advanced_summary['mrr_at_10']:>12.4f}"
    f"{differences['mrr_at_10']:>+14.4f}"
)


for k in K_VALUES:

    basic_value = (
        basic_summary[
            "gold_context_recall"
        ][
            f"recall_at_{k}"
        ]
    )


    advanced_value = (
        advanced_summary[
            "gold_context_recall"
        ][
            f"recall_at_{k}"
        ]
    )


    print(
        f"{f'Gold Recall@{k}':<20}"
        f"{basic_value:>12.4f}"
        f"{advanced_value:>12.4f}"
        f"{advanced_value - basic_value:>+14.4f}"
    )


# 17. SAVE DETAILED RESULTS

detail_output = {

    "split":
        "test",

    "questions":
        500,

    "metric_definitions": {

        "source_hit_at_k":
            (
                "Whether at least one of the "
                "top-k retrieved passages has "
                "the same source_record_id as "
                "the benchmark gold source."
            ),

        "mrr_at_10":
            (
                "Mean reciprocal rank of the "
                "first correct-source passage "
                "within the top 10; zero if no "
                "correct-source passage occurs."
            ),

        "gold_context_recall_at_k":
            (
                "Number of benchmark-attached "
                "gold context passages appearing "
                "in the top k divided by the "
                "number of attached gold "
                "context passages."
            )
    },

    "systems": {

        "basic_rag":
            basic_question_results,

        "advanced_rag":
            advanced_question_results
    }
}


save_json(
    detail_output,
    DETAIL_PATH
)


# 18. SAVE SUMMARY

summary_output = {

    "split":
        "test",

    "analysis":
        "final_frozen_retrieval_evaluation",

    "questions":
        500,

    "evaluated_k":
        K_VALUES,

    "primary_retrieval_metrics": [
        "source_hit_at_k",
        "mrr_at_10"
    ],

    "supplementary_retrieval_metric":
        "gold_context_recall_at_k",

    "system_summaries":
        system_summaries,

    "paired_source_hit_complementarity":
        paired_hit_analysis,

    "advanced_minus_basic":
        differences,

    "frozen_input_hashes": {

        relative_path(
            BASIC_RESULTS_PATH
        ):
            sha256_file(
                BASIC_RESULTS_PATH
            ),

        relative_path(
            ADVANCED_RESULTS_PATH
        ):
            sha256_file(
                ADVANCED_RESULTS_PATH
            ),

        relative_path(
            CORPUS_PATH
        ):
            sha256_file(
                CORPUS_PATH
            ),

        relative_path(
            GROUND_TRUTH_PATH
        ):
            sha256_file(
                GROUND_TRUTH_PATH
            )
    },

    "notes": [

        (
            "Basic RAG generation used the "
            "top five passages from the frozen "
            "Dense ranking."
        ),

        (
            "Advanced RAG generation used the "
            "top five passages after frozen "
            "cross-encoder reranking."
        ),

        (
            "Retrieval evaluation additionally "
            "reports the frozen ranking through "
            "rank 10 for both systems."
        ),

        (
            "No retrieval model, weighting, "
            "reranker, or hyperparameter was "
            "changed after opening test labels."
        )
    ]
}


save_json(
    summary_output,
    SUMMARY_PATH
)



# 19. FINAL OUTPUT


print(
    "\n[10] FILES SAVED"
)


print(
    DETAIL_PATH
)


print(
    SUMMARY_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 44 COMPLETE - "
    "FINAL TEST RETRIEVAL EVALUATION"
)


print(
    "=" * 78
)

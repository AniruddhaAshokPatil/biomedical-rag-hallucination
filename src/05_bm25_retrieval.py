import json
import re
import math
import heapq

from pathlib import Path
from collections import Counter, defaultdict


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "retrieval"
)

RESULTS_DIR.mkdir(
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


RESULTS_OUTPUT_PATH = (
    RESULTS_DIR / "bm25_dev_results.json"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR / "bm25_dev_summary.json"
)

# 2. HELPER FUNCTIONS

def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(data, file_path: Path):
    """Save data as formatted JSON."""

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


# 3. TOKENIZATION

def tokenize(text):
    """
    Simple lexical tokenizer.

    Example:

    "D-dimer levels (P<0.05)"
            ↓
    ["d-dimer", "levels", "p", "0", "05"]

    We:
    - convert text to lowercase
    - extract alphanumeric tokens
    - preserve simple hyphenated words
    """

    if not text:
        return []

    text = text.lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text
    )

    return tokens


# 4. BM25 IMPLEMENTATION

class BM25Retriever:
    """
    Minimal BM25 Okapi implementation.

    Parameters
    ----------
    k1:
        Controls term-frequency saturation.

    b:
        Controls document-length normalization.

    Common BM25 defaults:
        k1 = 1.5
        b  = 0.75
    """

    def __init__(
        self,
        documents,
        k1=1.5,
        b=0.75
    ):

        self.documents = documents

        self.k1 = k1
        self.b = b

        self.num_documents = len(
            documents
        )

        self.doc_lengths = []

        self.postings = defaultdict(list)

        self.document_frequency = Counter()

        self.idf = {}

        self._build_index()


    def _build_index(self):
        """
        Create the lexical search index.
        """

        print(
            "\nBuilding BM25 index..."
        )

        total_document_length = 0


        
        # --------Tokenize every document------------
       

        for doc_index, document in enumerate(
            self.documents
        ):

            # IMPORTANT:
            # BM25 searches ONLY the evidence text.
            # We are NOT indexing:
            # source_record_id
            # document_id
            # section
            # year
            # meshes
            # This prevents metadata from creating

            tokens = tokenize(
                document["text"]
            )

            document_length = len(
                tokens
            )

            self.doc_lengths.append(
                document_length
            )

            total_document_length += (
                document_length
            )


            # Count terms inside this document

            term_frequencies = Counter(
                tokens
            )


            for term, frequency in (
                term_frequencies.items()
            ):

                self.postings[
                    term
                ].append(
                    (
                        doc_index,
                        frequency
                    )
                )

                self.document_frequency[
                    term
                ] += 1


     
        # ----------Average document length------------
       

        if self.num_documents > 0:

            self.average_document_length = (
                total_document_length
                / self.num_documents
            )

        else:

            self.average_document_length = 0


     
        # -----------Calculate IDF-----------
        # IDF:
        # Rare terms receive higher importance.
        # Example:
        # "appendicitis" -> relatively informative
        # "the" -> not very informative
        # ----------------------------------------------------

        for term, df in (
            self.document_frequency.items()
        ):

            numerator = (
                self.num_documents
                - df
                + 0.5
            )

            denominator = (
                df
                + 0.5
            )

            self.idf[term] = math.log(
                1
                +
                (
                    numerator
                    / denominator
                )
            )


        print(
            f"Documents indexed : "
            f"{self.num_documents}"
        )

        print(
            f"Unique terms      : "
            f"{len(self.postings)}"
        )

        print(
            f"Average doc length: "
            f"{self.average_document_length:.2f} tokens"
        )


    def search(
        self,
        query,
        top_k=10
    ):
        """
        Search the BM25 index.

        Returns:
            list of:
            (document_index, score)
        """

        query_tokens = tokenize(
            query
        )

        # We use unique query terms so repeated
        # query words do not artificially inflate
        # the result.

        query_terms = set(
            query_tokens
        )


        scores = defaultdict(float)


        
        #-------------- Calculate BM25 score-------------
       

        for term in query_terms:

            if term not in self.postings:
                continue


            term_idf = self.idf[
                term
            ]


            for doc_index, term_frequency in (
                self.postings[term]
            ):

                document_length = (
                    self.doc_lengths[
                        doc_index
                    ]
                )


                length_normalization = (

                    self.k1
                    *
                    (
                        1
                        - self.b
                        +
                        self.b
                        *
                        (
                            document_length
                            /
                            self.average_document_length
                        )
                    )
                )


                numerator = (
                    term_frequency
                    *
                    (
                        self.k1
                        + 1
                    )
                )


                denominator = (
                    term_frequency
                    +
                    length_normalization
                )


                score = (
                    term_idf
                    *
                    (
                        numerator
                        /
                        denominator
                    )
                )


                scores[
                    doc_index
                ] += score


       
        # ---------- Retrieve highest scoring documents------------
        

        ranked_results = heapq.nlargest(
            top_k,
            scores.items(),
            key=lambda item: item[1]
        )


        return ranked_results


# 5. LOAD DATA

print("=" * 75)
print("SECTION 5 - BM25 RETRIEVAL BASELINE")
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


# 6. SAFETY CHECK

print("\n[2] DEVELOPMENT-ONLY EVALUATION CHECK")


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


if len(wrong_split) == 0:

    print(
        "PASS: BM25 tuning/evaluation "
        "uses development questions only."
    )

else:

    raise ValueError(
        "Test-set leakage detected."
    )


# 7. BUILD BM25

bm25 = BM25Retriever(
    retrieval_corpus,
    k1=1.5,
    b=0.75
)

# 8. RETRIEVAL SETTINGS

K_VALUES = [
    1,
    3,
    5,
    10
]

MAX_K = max(
    K_VALUES
)


# 9. METRIC STORAGE

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


# 10. RUN RETRIEVAL

print("\n[3] RUNNING BM25 RETRIEVAL")


total_questions = len(
    development_questions
)


for question_number, question_record in enumerate(
    development_questions,
    start=1
):

    question_id = question_record[
        "question_id"
    ]

    question_text = question_record[
        "question"
    ]


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


    
    # -------------Search corpus-------------
   

    retrieved = bm25.search(
        question_text,
        top_k=MAX_K
    )


    ranked_documents = []


    first_gold_rank = None


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


        is_gold_context = (
            document[
                "document_id"
            ]
            in gold_context_ids
        )


        is_gold_source = (
            document[
                "source_record_id"
            ]
            ==
            question_id
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


    
    # -----------Source Hit@K-----------

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

            source_hits[
                k
            ] += 1


       
        # ----------Gold Context Recall@K-----------
        

        retrieved_gold_count = sum(

            1

            for document
            in top_k_documents

            if document[
                "is_gold_context"
            ]
        )


        if len(
            gold_context_ids
        ) > 0:

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

 
    # ----------Reciprocal Rank---------

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


   
    # ----------Save query result----------
   
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


    
    # ---------Progress indicator---------
   

    if question_number % 100 == 0:

        print(
            f"Processed "
            f"{question_number}"
            f"/{total_questions} "
            f"questions"
        )


# 11. CALCULATE FINAL METRICS

print("\n[4] RETRIEVAL RESULTS")


summary = {
    "retriever": "BM25",
    "number_of_questions":
        total_questions,

    "number_of_documents":
        len(
            retrieval_corpus
        ),

    "parameters": {
        "k1": 1.5,
        "b": 0.75
    },

    "source_hit_at_k": {},

    "gold_context_recall_at_k": {},

    "mrr_at_10": None,

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


# 12. PRINT METRIC TABLE

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


print("\nMean Gold-Context Recall@K")

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


# 13. SAVE RESULTS

save_json(
    all_results,
    RESULTS_OUTPUT_PATH
)

save_json(
    summary,
    SUMMARY_OUTPUT_PATH
)


print("\n[5] RESULTS SAVED")

print(
    RESULTS_OUTPUT_PATH
)

print(
    SUMMARY_OUTPUT_PATH
)


# 14. SHOW SOME FAILURE EXAMPLES

print("\n[6] EXAMPLE RETRIEVAL FAILURES")

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
            + "-" * 75
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
            "\nTop 3 BM25 results:"
        )


        for document in top_five[:3]:

            print(
                f"\nRank "
                f"{document['rank']}"
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
                f"Score: "
                f"{document['score']}"
            )

            print(
                f"Text: "
                f"{document['text'][:300]}"
            )


        if failure_count == 3:
            break


# 15. COMPLETE

print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 5 COMPLETE"
)

print(
    "=" * 75
)

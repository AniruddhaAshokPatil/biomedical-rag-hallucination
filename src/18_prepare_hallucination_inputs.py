import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

GENERATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "generation"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


GENERATIONS_PATH = (
    RESULTS_DIR
    / "development_generations.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR
    / "evaluation_ground_truth.json"
)

CORPUS_PATH = (
    PROCESSED_DIR
    / "retrieval_corpus.json"
)


INPUT_FILES = {

    "baseline_llm":
        GENERATION_DIR
        / "baseline_dev_inputs.json",

    "basic_rag":
        GENERATION_DIR
        / "basic_rag_dev_inputs.json",

    "advanced_rag":
        GENERATION_DIR
        / "advanced_rag_dev_inputs.json",

    "gold_context_control":
        GENERATION_DIR
        / "gold_context_dev_inputs.json"
}


OUTPUT_PATH = (
    RESULTS_DIR
    / "hallucination_evaluation_inputs.json"
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


# 3. LOAD DATA

print("=" * 75)

print(
    "SECTION 18 - PREPARE HALLUCINATION EVALUATION INPUTS"
)

print("=" * 75)


generation_data = load_json(
    GENERATIONS_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)

retrieval_corpus = load_json(
    CORPUS_PATH
)


generation_results = (
    generation_data[
        "results"
    ]
)


print(
    "\n[1] DATA LOADED"
)

print(
    f"Generated answers : "
    f"{len(generation_results)}"
)

print(
    f"Corpus passages   : "
    f"{len(retrieval_corpus)}"
)


# 4. LOAD ORIGINAL GENERATION INPUTS

generation_inputs = {}


for system_name, path in INPUT_FILES.items():

    records = load_json(
        path
    )


    generation_inputs[
        system_name
    ] = {

        record[
            "question_id"
        ]: record

        for record in records
    }


# 5. INDEX CORPUS

corpus_by_id = {

    document[
        "document_id"
    ]:
        document

    for document in retrieval_corpus
}


# 6. DEVELOPMENT SAFETY CHECK

print(
    "\n[2] DEVELOPMENT SAFETY CHECK"
)


wrong_split = []


for result in generation_results:

    question_id = result[
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
    f"Non-development answers: "
    f"{len(wrong_split)}"
)


if wrong_split:

    raise ValueError(
        "Test-set leakage detected."
    )


print(
    "PASS: Hallucination preparation "
    "uses development data only."
)


# 7. BUILD GOLD EVIDENCE

def get_gold_evidence(
    question_id
):

    gold_context_ids = (

        ground_truth[
            question_id
        ][
            "gold_context_ids"
        ]
    )


    evidence = []


    for index, document_id in enumerate(
        gold_context_ids,
        start=1
    ):

        document = (
            corpus_by_id[
                document_id
            ]
        )


        evidence.append(
            {
                "label":
                    f"G{index}",

                "document_id":
                    document_id,

                "section":
                    document[
                        "section"
                    ],

                "text":
                    document[
                        "text"
                    ]
            }
        )


    return evidence


# 8. BUILD PROVIDED EVIDENCE

def get_provided_evidence(
    question_id,
    system_name
):

    item = (

        generation_inputs[
            system_name
        ][
            question_id
        ]
    )


    evidence = []


    for context in item[
        "contexts"
    ]:

        evidence.append(
            {
                "label":
                    context[
                        "context_label"
                    ],

                "section":
                    context[
                        "section"
                    ],

                "text":
                    context[
                        "text"
                    ]
            }
        )


    return evidence


# 9. BUILD EVALUATION RECORDS

print(
    "\n[3] BUILDING EVALUATION RECORDS"
)


evaluation_records = []


for result in generation_results:

    question_id = result[
        "question_id"
    ]

    system_name = result[
        "system"
    ]


    gold_evidence = get_gold_evidence(
        question_id
    )


    provided_evidence = (
        get_provided_evidence(
            question_id,
            system_name
        )
    )


    record = {

        "question_id":
            question_id,

        "system":
            system_name,

        "question":
            result[
                "question"
            ],

        "generated_decision":
            result[
                "decision"
            ],

        "generated_answer":
            result[
                "answer"
            ],

        "generated_citations":
            result[
                "citations"
            ],

        # Common benchmark evidence.
        # Used for fair hallucination comparison across ALL four systems.

        "gold_evidence":
            gold_evidence,

        # Evidence actually provided to generation.
        # Baseline -> []
        # Basic -> retrieved top 5
        # Advanced -> reranked top 5
        # Gold control -> gold contexts
        # Used later for RAG faithfulness.

        "provided_evidence":
            provided_evidence,

        # Evaluation-only targets
        "gold_decision":
            ground_truth[
                question_id
            ][
                "gold_decision"
            ],

        "reference_answer":
            ground_truth[
                question_id
            ][
                "reference_answer"
            ]
    }


    evaluation_records.append(
        record
    )


# 10. VALIDATION

print(
    "\n[4] VALIDATION"
)


print(
    f"Evaluation records: "
    f"{len(evaluation_records)}"
)


if len(evaluation_records) != 2000:

    raise ValueError(
        "Expected exactly 2,000 "
        "evaluation records."
    )


systems = {}


for record in evaluation_records:

    system = record[
        "system"
    ]

    systems[
        system
    ] = (
        systems.get(
            system,
            0
        )
        +
        1
    )


for system, count in systems.items():

    print(
        f"{system:<22}: "
        f"{count}"
    )


# 11. PROVIDED CONTEXT CHECK

print(
    "\n[5] PROVIDED EVIDENCE CHECK"
)


baseline_with_context = 0

rag_without_context = 0


for record in evaluation_records:

    system = record[
        "system"
    ]

    context_count = len(
        record[
            "provided_evidence"
        ]
    )


    if (
        system
        ==
        "baseline_llm"

        and
        context_count != 0
    ):

        baseline_with_context += 1


    if (
        system
        in {
            "basic_rag",
            "advanced_rag",
            "gold_context_control"
        }

        and

        context_count == 0
    ):

        rag_without_context += 1


print(
    f"Baseline records with context : "
    f"{baseline_with_context}"
)


print(
    f"Evidence systems with no context: "
    f"{rag_without_context}"
)


if (
    baseline_with_context
    or
    rag_without_context
):

    raise ValueError(
        "Unexpected evidence configuration."
    )


print(
    "PASS: Evidence configuration "
    "matches experiment design."
)


# 12. GOLD EVIDENCE CHECK

print(
    "\n[6] GOLD EVIDENCE CHECK"
)


missing_gold = [

    record[
        "question_id"
    ]

    for record
    in evaluation_records

    if not record[
        "gold_evidence"
    ]
]


print(
    f"Records without gold evidence: "
    f"{len(missing_gold)}"
)


if missing_gold:

    raise ValueError(
        "Some evaluation records "
        "lack gold evidence."
    )


print(
    "PASS: Every answer has benchmark "
    "gold evidence."
)


# 13. SAMPLE

TARGET_ID = "17610439"


print(
    "\n[7] WEEKEND QUESTION EVALUATION INPUT"
)


for record in evaluation_records:

    if (
        record[
            "question_id"
        ]
        ==
        TARGET_ID

        and

        record[
            "system"
        ]
        ==
        "advanced_rag"
    ):

        print(
            f"System: "
            f"{record['system']}"
        )


        print(
            "\nQuestion:"
        )

        print(
            record[
                "question"
            ]
        )


        print(
            "\nGenerated answer:"
        )

        print(
            record[
                "generated_answer"
            ]
        )


        print(
            "\nGold evidence passages:"
        )

        print(
            len(
                record[
                    "gold_evidence"
                ]
            )
        )


        print(
            "\nProvided evidence passages:"
        )

        print(
            len(
                record[
                    "provided_evidence"
                ]
            )
        )


        break


# 14. SAVE

save_json(
    evaluation_records,
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
    "SECTION 18 COMPLETE"
)

print(
    "=" * 75
)
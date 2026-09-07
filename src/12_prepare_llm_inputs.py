import json
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

GENERATION_DIR = (
    PROJECT_ROOT / "data" / "generation"
)


DEVELOPMENT_QUESTIONS_PATH = (
    PROCESSED_DIR / "development_questions.json"
)

RETRIEVAL_CORPUS_PATH = (
    PROCESSED_DIR / "retrieval_corpus.json"
)

GROUND_TRUTH_PATH = (
    PROCESSED_DIR / "evaluation_ground_truth.json"
)

BASIC_CONTEXT_PATH = (
    GENERATION_DIR / "basic_rag_dev_contexts.json"
)

ADVANCED_CONTEXT_PATH = (
    GENERATION_DIR / "advanced_rag_dev_contexts.json"
)


BASELINE_OUTPUT = (
    GENERATION_DIR / "baseline_dev_inputs.json"
)

BASIC_OUTPUT = (
    GENERATION_DIR / "basic_rag_dev_inputs.json"
)

ADVANCED_OUTPUT = (
    GENERATION_DIR / "advanced_rag_dev_inputs.json"
)

GOLD_OUTPUT = (
    GENERATION_DIR / "gold_context_dev_inputs.json"
)

PROMPT_CONFIG_OUTPUT = (
    GENERATION_DIR / "generation_prompt_config.json"
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
print("SECTION 12 - PREPARE CONTROLLED LLM INPUTS")
print("=" * 75)


development_questions = load_json(
    DEVELOPMENT_QUESTIONS_PATH
)

retrieval_corpus = load_json(
    RETRIEVAL_CORPUS_PATH
)

ground_truth = load_json(
    GROUND_TRUTH_PATH
)

basic_contexts = load_json(
    BASIC_CONTEXT_PATH
)

advanced_contexts = load_json(
    ADVANCED_CONTEXT_PATH
)


print("\n[1] DATA LOADED")

print(
    f"Development questions : "
    f"{len(development_questions)}"
)

print(
    f"Corpus passages       : "
    f"{len(retrieval_corpus)}"
)

print(
    f"Basic context records : "
    f"{len(basic_contexts)}"
)

print(
    f"Advanced records      : "
    f"{len(advanced_contexts)}"
)


# 4. INDEX DATA

basic_by_id = {

    item["question_id"]: item

    for item in basic_contexts
}


advanced_by_id = {

    item["question_id"]: item

    for item in advanced_contexts
}


corpus_by_document_id = {

    document["document_id"]: document

    for document in retrieval_corpus
}


# 5. CORE PROMPT

# This instruction remains the same for all systems.
# Baseline:evidence = none
# RAG:evidence = retrieved passages

# Gold-control: evidence = known source passages

SYSTEM_PROMPT = """
You are answering biomedical research questions.

Your task is to:
1. Answer the question concisely and factually.
2. Give an overall decision: yes, no, or maybe.
3. Avoid unsupported factual claims.

If evidence passages are provided:
- Base factual claims on the supplied evidence.
- Do not add medical details that are not supported by the evidence.
- If the evidence is inconclusive, use "maybe".
- Cite supporting passages using their context labels such as C1 or C2.

If no evidence passages are provided:
- Answer using your existing knowledge.
- Do not invent references or citations.

Return ONLY valid JSON in this exact structure:

{
  "decision": "yes|no|maybe",
  "answer": "concise biomedical answer",
  "citations": ["C1", "C2"]
}

For a no-evidence question, citations must be an empty list.
""".strip()


# 6. FORMAT EVIDENCE

def format_contexts(contexts):

    formatted_contexts = []


    for index, context in enumerate(
        contexts,
        start=1
    ):

        label = f"C{index}"


        formatted_contexts.append(
            {
                "context_label":
                    label,

                "section":
                    context.get(
                        "section"
                    ),

                "text":
                    context["text"]
            }
        )


    return formatted_contexts


# 7. CREATE USER PROMPT

def build_user_prompt(
    question,
    contexts=None
):

    if not contexts:

        evidence_text = (
            "NO EXTERNAL EVIDENCE PROVIDED."
        )

    else:

        evidence_parts = []


        for context in contexts:

            evidence_parts.append(
                f"[{context['context_label']}] "
                f"Section: {context['section']}\n"
                f"{context['text']}"
            )


        evidence_text = "\n\n".join(
            evidence_parts
        )


    prompt = (
        f"QUESTION:\n"
        f"{question}\n\n"
        f"EVIDENCE:\n"
        f"{evidence_text}"
    )


    return prompt


# 8. OUTPUT STORAGE

baseline_inputs = []

basic_inputs = []

advanced_inputs = []

gold_inputs = []


# 9. BUILD ALL FOUR CONDITIONS

print(
    "\n[2] BUILDING EXPERIMENT CONDITIONS"
)


for question_record in development_questions:

    question_id = (
        question_record[
            "question_id"
        ]
    )

    question = (
        question_record[
            "question"
        ]
    )


    # Safety check

    if (
        ground_truth[
            question_id
        ]["split"]
        != "development"
    ):

        raise ValueError(
            "Test question detected."
        )


    
    # --------A. BASELINE LLM----------
    

    baseline_inputs.append(
        {
            "question_id":
                question_id,

            "system":
                "baseline_llm",

            "question":
                question,

            "contexts":
                [],

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    []
                )
        }
    )


  
    # -------B. BASIC RAG---------
    
    basic_raw_contexts = (
        basic_by_id[
            question_id
        ]["contexts"]
    )


    basic_formatted = format_contexts(
        basic_raw_contexts
    )


    basic_inputs.append(
        {
            "question_id":
                question_id,

            "system":
                "basic_rag",

            "question":
                question,

            "contexts":
                basic_formatted,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    basic_formatted
                )
        }
    )


    # --------C. ADVANCED RAG---------

    advanced_raw_contexts = (
        advanced_by_id[
            question_id
        ]["contexts"]
    )


    advanced_formatted = format_contexts(
        advanced_raw_contexts
    )


    advanced_inputs.append(
        {
            "question_id":
                question_id,

            "system":
                "advanced_rag",

            "question":
                question,

            "contexts":
                advanced_formatted,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    advanced_formatted
                )
        }
    )


    
    # ---------D. GOLD-CONTEXT CONTROL---------
  
    # IMPORTANT:
    # We use only gold CONTEXT passages.
    # We DO NOT give:
    # - reference answer
    # - final decision

    gold_context_ids = (
        ground_truth[
            question_id
        ]["gold_context_ids"]
    )


    gold_raw_contexts = []


    for document_id in gold_context_ids:

        document = (
            corpus_by_document_id[
                document_id
            ]
        )


        gold_raw_contexts.append(
            {
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


    gold_formatted = format_contexts(
        gold_raw_contexts
    )


    gold_inputs.append(
        {
            "question_id":
                question_id,

            "system":
                "gold_context_control",

            "question":
                question,

            "contexts":
                gold_formatted,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                build_user_prompt(
                    question,
                    gold_formatted
                )
        }
    )


# 10. SAVE INPUT FILES

save_json(
    baseline_inputs,
    BASELINE_OUTPUT
)

save_json(
    basic_inputs,
    BASIC_OUTPUT
)

save_json(
    advanced_inputs,
    ADVANCED_OUTPUT
)

save_json(
    gold_inputs,
    GOLD_OUTPUT
)


# 11. SAVE PROMPT CONFIGURATION

prompt_config = {

    "prompt_version":
        "v1",

    "system_prompt":
        SYSTEM_PROMPT,

    "output_format": {
        "decision":
            "yes|no|maybe",

        "answer":
            "string",

        "citations":
            "list of context labels"
    },

    "rules": {

        "same_core_prompt":
            True,

        "baseline_has_no_context":
            True,

        "basic_top_k":
            5,

        "advanced_top_k":
            5,

        "gold_context_uses_reference_answer":
            False,

        "gold_context_uses_gold_decision":
            False
    }
}


save_json(
    prompt_config,
    PROMPT_CONFIG_OUTPUT
)


# 12. VALIDATION

print(
    "\n[3] OUTPUT COUNTS"
)


print(
    f"Baseline inputs : "
    f"{len(baseline_inputs)}"
)

print(
    f"Basic inputs    : "
    f"{len(basic_inputs)}"
)

print(
    f"Advanced inputs : "
    f"{len(advanced_inputs)}"
)

print(
    f"Gold inputs     : "
    f"{len(gold_inputs)}"
)


# 13. LEAKAGE CHECK

print(
    "\n[4] LEAKAGE CHECK"
)


forbidden_strings = []


for question_id in [
    item["question_id"]
    for item in baseline_inputs
]:

    reference_answer = (
        ground_truth[
            question_id
        ][
            "reference_answer"
        ]
    )

    gold_decision = (
        ground_truth[
            question_id
        ][
            "gold_decision"
        ]
    )


    for dataset in [
        baseline_inputs,
        basic_inputs,
        advanced_inputs,
        gold_inputs
    ]:

        # Find same question
        item = next(
            x
            for x in dataset
            if x["question_id"]
            ==
            question_id
        )


        prompt = item[
            "user_prompt"
        ]


        if (
            reference_answer
            and
            reference_answer in prompt
        ):

            forbidden_strings.append(
                (
                    question_id,
                    item["system"],
                    "reference_answer"
                )
            )


print(
    f"Reference-answer leakage cases: "
    f"{len(forbidden_strings)}"
)


if forbidden_strings:

    raise ValueError(
        "Reference-answer leakage detected."
    )


print(
    "PASS: Reference answers are not "
    "present in generation prompts."
)


# 14. SAMPLE COMPARISON

TARGET_ID = "17610439"


print(
    "\n[5] SAMPLE PROMPTS - WEEKEND QUESTION"
)


for name, dataset in [

    (
        "BASELINE",
        baseline_inputs
    ),

    (
        "BASIC RAG",
        basic_inputs
    ),

    (
        "ADVANCED RAG",
        advanced_inputs
    ),

    (
        "GOLD CONTROL",
        gold_inputs
    )
]:

    item = next(

        record

        for record in dataset

        if record[
            "question_id"
        ] == TARGET_ID
    )


    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    print(
        item[
            "user_prompt"
        ]
    )



# 15. COMPLETE


print(
    "\n"
    + "=" * 75
)

print(
    "SECTION 12 COMPLETE"
)

print(
    "=" * 75
)

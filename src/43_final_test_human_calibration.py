import ast
import csv
import hashlib
import json
import math
import random

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


HALLUCINATION_FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_hallucination_judgment_manifest.json"
)


SECTION_29_PATH = (
    PROJECT_ROOT
    / "src"
    / "29_calibrate_hallucination_rates.py"
)


FINAL_HALLUCINATION_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_hallucination_v3_summary.json"
)


VALIDATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
    / "human_validation_system_balanced"
)


ANNOTATION_PATH = (
    VALIDATION_DIR
    / "system_balanced_validation_claims.csv"
)


KEY_PATH = (
    VALIDATION_DIR
    / "system_balanced_validation_key.json"
)


DEVELOPMENT_CALIBRATION_PATH = (
    VALIDATION_DIR
    / "human_calibrated_hallucination_rates.json"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "test"
    / "evaluation"
    / "final_test_human_calibrated_hallucination_sensitivity.json"
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


# 3. START

print("=" * 78)

print(
    "SECTION 43 - FINAL TEST HUMAN-CALIBRATED "
    "HALLUCINATION SENSITIVITY ANALYSIS"
)

print("=" * 78)


# 4. VERIFY SECTION 29 REMAINS FROZEN

print(
    "\n[1] VERIFYING FROZEN CALIBRATION METHOD"
)


original_manifest = load_json(
    ORIGINAL_FREEZE_MANIFEST
)


section_29_key = None


for relative_path in (
    original_manifest[
        "file_hashes"
    ]
):

    if relative_path.endswith(
        "29_calibrate_hallucination_rates.py"
    ):

        section_29_key = relative_path
        break


if section_29_key is None:

    raise RuntimeError(
        "Section 29 was not found "
        "in the original freeze manifest."
    )


expected_hash = (
    original_manifest[
        "file_hashes"
    ][
        section_29_key
    ]
)


actual_hash = sha256_file(
    SECTION_29_PATH
)


if actual_hash != expected_hash:

    raise RuntimeError(
        "Frozen Section 29 calibration "
        "script has changed."
    )


print(
    "PASS: Frozen Section 29 calibration "
    "script remains unchanged."
)


# 5. VERIFY FINAL V3 JUDGMENTS REMAIN FROZEN

print(
    "\n[2] VERIFYING FINAL HALLUCINATION FREEZE"
)


hallucination_manifest = load_json(
    HALLUCINATION_FREEZE_MANIFEST
)


changed_files = []
missing_files = []


for relative_path, expected_hash in (
    hallucination_manifest[
        "sha256"
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


print(
    f"Frozen artifacts checked : "
    f"{len(hallucination_manifest['sha256'])}"
)


print(
    f"Changed artifacts        : "
    f"{len(changed_files)}"
)


print(
    f"Missing artifacts        : "
    f"{len(missing_files)}"
)


if changed_files or missing_files:

    raise RuntimeError(
        "Frozen hallucination judgments "
        "have changed."
    )


print(
    "PASS: Final V3 judgments "
    "remain unchanged."
)


# 6. EXTRACT EXACT SECTION 29 CALIBRATION LOGIC

# We deliberately reuse the frozen implementation from
# Section 29 rather than manually recreating:
#   - clean()
#   - human_is_hallucinated()
#   - percentile()
#   - load_csv()
# This also preserves the original CSV encoding handling.


print(
    "\n[3] LOADING EXACT SECTION 29 CALIBRATION LOGIC"
)


source = SECTION_29_PATH.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source,
    filename=str(
        SECTION_29_PATH
    )
)


CONSTANT_NAMES = {

    "SYSTEM_ORDER",
    "N_BOOTSTRAP",
    "RANDOM_SEED"
}


FUNCTION_NAMES = {

    "clean",
    "human_is_hallucinated",
    "percentile",
    "load_csv"
}


selected_nodes = []


for node in tree.body:

    if isinstance(
        node,
        ast.Assign
    ):

        for target in node.targets:

            if (
                isinstance(
                    target,
                    ast.Name
                )
                and
                target.id
                in
                CONSTANT_NAMES
            ):

                selected_nodes.append(
                    node
                )

                break

    elif isinstance(
        node,
        ast.FunctionDef
    ):

        if node.name in FUNCTION_NAMES:

            selected_nodes.append(
                node
            )


module = ast.Module(
    body=selected_nodes,
    type_ignores=[]
)


ast.fix_missing_locations(
    module
)


namespace = {

    "math":
        math,

    "csv":
        csv
}


exec(
    compile(
        module,
        filename=str(
            SECTION_29_PATH
        ),
        mode="exec"
    ),
    namespace
)


required = (
    CONSTANT_NAMES
    |
    FUNCTION_NAMES
)


missing = (
    required
    -
    set(
        namespace.keys()
    )
)


if missing:

    raise RuntimeError(
        "Could not recover exact Section 29 "
        f"objects: {sorted(missing)}"
    )


SYSTEM_ORDER = (
    namespace[
        "SYSTEM_ORDER"
    ]
)


N_BOOTSTRAP = (
    namespace[
        "N_BOOTSTRAP"
    ]
)


RANDOM_SEED = (
    namespace[
        "RANDOM_SEED"
    ]
)


clean = (
    namespace[
        "clean"
    ]
)


human_is_hallucinated = (
    namespace[
        "human_is_hallucinated"
    ]
)


percentile = (
    namespace[
        "percentile"
    ]
)


load_csv = (
    namespace[
        "load_csv"
    ]
)


print(
    f"Systems              : "
    f"{SYSTEM_ORDER}"
)


print(
    f"Bootstrap iterations : "
    f"{N_BOOTSTRAP}"
)


print(
    f"Random seed          : "
    f"{RANDOM_SEED}"
)


if N_BOOTSTRAP != 10000:

    raise RuntimeError(
        "Unexpected frozen bootstrap count."
    )


if RANDOM_SEED != 42:

    raise RuntimeError(
        "Unexpected frozen calibration seed."
    )


print(
    "PASS: Exact frozen calibration "
    "functions loaded."
)


# 7. LOAD DEVELOPMENT HUMAN VALIDATION

print(
    "\n[4] LOADING FROZEN HUMAN VALIDATION DATA"
)


human_rows = load_csv(
    ANNOTATION_PATH
)


key_data = load_json(
    KEY_PATH
)


development_calibration = load_json(
    DEVELOPMENT_CALIBRATION_PATH
)


key = key_data[
    "key"
]


print(
    f"Human validation rows : "
    f"{len(human_rows)}"
)


if len(
    human_rows
) != 160:

    raise ValueError(
        "Expected exactly 160 "
        "system-balanced human "
        "validation claims."
    )


# 8. RECREATE HUMAN VALIDATION STRATA

validation_strata = {

    system: {

        "grounded":
            [],

        "hallucinated":
            []
    }

    for system
    in SYSTEM_ORDER
}


for row in human_rows:

    validation_id = (
        row[
            "validation_id"
        ].strip()
    )


    human_label = clean(
        row[
            "human_label"
        ]
    )


    if human_label not in {

        "grounded",
        "hallucinated"

    }:

        raise ValueError(
            f"Invalid human label "
            f"for {validation_id}: "
            f"{human_label}"
        )


    if validation_id not in key:

        raise KeyError(
            f"Validation ID missing "
            f"from key: {validation_id}"
        )


    key_record = key[
        validation_id
    ]


    system = (
        key_record[
            "system"
        ]
    )


    automatic_stratum = clean(
        key_record[
            "automatic_binary"
        ]
    )


    if system not in SYSTEM_ORDER:

        raise ValueError(
            f"Unexpected system: "
            f"{system}"
        )


    if automatic_stratum not in {

        "grounded",
        "hallucinated"

    }:

        raise ValueError(
            f"Unexpected automatic "
            f"stratum: "
            f"{automatic_stratum}"
        )


    validation_strata[
        system
    ][
        automatic_stratum
    ].append(
        human_is_hallucinated(
            human_label
        )
    )


# 9. VERIFY 20 CLAIMS PER SYSTEM / STRATUM

print(
    "\n[5] HUMAN CALIBRATION PARAMETERS"
)


human_rates = {}


for system in SYSTEM_ORDER:

    human_rates[
        system
    ] = {}


    print(
        f"\n{system}"
    )


    for stratum in [

        "grounded",
        "hallucinated"

    ]:

        values = (
            validation_strata[
                system
            ][
                stratum
            ]
        )


        if len(
            values
        ) != 20:

            raise ValueError(
                f"{system}/{stratum} "
                f"must contain exactly "
                f"20 validation claims; "
                f"found {len(values)}."
            )


        human_rate = (
            sum(
                values
            )
            /
            len(
                values
            )
        )


        human_rates[
            system
        ][
            stratum
        ] = human_rate


        print(
            f"  Auto {stratum:<12} "
            f"human hallucination="
            f"{human_rate:.4f}"
        )


# 10. VERIFY AGAINST SAVED DEVELOPMENT CALIBRATION

print(
    "\n[6] VERIFYING DEVELOPMENT CALIBRATION PARAMETERS"
)


for system in SYSTEM_ORDER:

    saved = (
        development_calibration[
            "systems"
        ][
            system
        ]
    )


    saved_grounded = (
        saved[
            "human_hallucination_rate_given_auto_grounded"
        ]
    )


    saved_hallucinated = (
        saved[
            "human_hallucination_rate_given_auto_hallucinated"
        ]
    )


    calculated_grounded = (
        human_rates[
            system
        ][
            "grounded"
        ]
    )


    calculated_hallucinated = (
        human_rates[
            system
        ][
            "hallucinated"
        ]
    )


    if (
        abs(
            saved_grounded
            -
            calculated_grounded
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Grounded-stratum calibration "
            f"mismatch for {system}."
        )


    if (
        abs(
            saved_hallucinated
            -
            calculated_hallucinated
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Hallucinated-stratum "
            f"calibration mismatch "
            f"for {system}."
        )


    print(
        f"PASS: {system}"
    )


print(
    "PASS: Human calibration parameters "
    "exactly reproduce Section 29."
)


# 11. LOAD FINAL TEST AUTOMATIC CLAIM POPULATION

print(
    "\n[7] LOADING FINAL TEST V3 CLAIM POPULATION"
)


test_summary = load_json(
    FINAL_HALLUCINATION_SUMMARY_PATH
)


system_summaries = (
    test_summary[
        "system_summaries"
    ]
)


population_counts = {}


for system in SYSTEM_ORDER:

    summary = (
        system_summaries[
            system
        ]
    )


    grounded = int(
        summary[
            "supported_claims"
        ]
    )


    hallucinated = int(

        summary[
            "unsupported_claims"
        ]

        +

        summary[
            "contradicted_claims"
        ]
    )


    total = (
        grounded
        +
        hallucinated
    )


    if (
        total
        !=
        summary[
            "total_claims"
        ]
    ):

        raise RuntimeError(
            f"Claim total mismatch "
            f"for {system}."
        )


    raw_rate = (
        hallucinated
        /
        total
    )


    frozen_rate = (
        summary[
            "micro_claim_hallucination_rate"
        ]
    )


    if (
        abs(
            raw_rate
            -
            frozen_rate
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Raw V3 rate mismatch "
            f"for {system}."
        )


    population_counts[
        system
    ] = {

        "grounded":
            grounded,

        "hallucinated":
            hallucinated,

        "total":
            total,

        "raw_rate":
            raw_rate
    }


    print(
        f"{system:<22} "
        f"grounded={grounded:<5} "
        f"hallucinated={hallucinated:<4} "
        f"total={total:<5} "
        f"raw={raw_rate:.4f}"
    )



# 12. APPLY EXACT POST-STRATIFIED CALIBRATION

# Calibrated rate =
#   P(auto grounded)
# * P(human hallucinated | auto grounded)
#   +
#   P(auto hallucinated)
#  * P(human hallucinated | auto hallucinated)


print(
    "\n[8] HUMAN-CALIBRATED FINAL TEST RATES"
)


results = {}


for system in SYSTEM_ORDER:

    counts = (
        population_counts[
            system
        ]
    )


    grounded_weight = (
        counts[
            "grounded"
        ]
        /
        counts[
            "total"
        ]
    )


    hallucinated_weight = (
        counts[
            "hallucinated"
        ]
        /
        counts[
            "total"
        ]
    )


    human_rate_auto_grounded = (
        human_rates[
            system
        ][
            "grounded"
        ]
    )


    human_rate_auto_hallucinated = (
        human_rates[
            system
        ][
            "hallucinated"
        ]
    )


    calibrated_rate = (

        grounded_weight
        *
        human_rate_auto_grounded

        +

        hallucinated_weight
        *
        human_rate_auto_hallucinated
    )


    results[
        system
    ] = {

        "total_claim_count":
            counts[
                "total"
            ],

        "automatic_grounded_count":
            counts[
                "grounded"
            ],

        "automatic_hallucinated_count":
            counts[
                "hallucinated"
            ],

        "automatic_hallucination_rate":
            counts[
                "raw_rate"
            ],

        "automatic_grounded_weight":
            grounded_weight,

        "automatic_hallucinated_weight":
            hallucinated_weight,

        "human_hallucination_rate_given_auto_grounded":
            human_rate_auto_grounded,

        "human_hallucination_rate_given_auto_hallucinated":
            human_rate_auto_hallucinated,

        "human_calibrated_hallucination_rate":
            calibrated_rate
    }


    print(
        f"\n{system}"
    )


    print(
        f"  Raw V3 rate       : "
        f"{counts['raw_rate']:.4f} "
        f"({counts['raw_rate'] * 100:.2f}%)"
    )


    print(
        f"  Human-calibrated  : "
        f"{calibrated_rate:.4f} "
        f"({calibrated_rate * 100:.2f}%)"
    )



# 13. EXACT STRATIFIED HUMAN BOOTSTRAP

# Same procedure as Section 29:
#  automatic test population weights remain fixed
# separately bootstrap the 20 human labels from the auto-grounded stratum
#  separately bootstrap the 20 human labels from the auto-hallucinated stratum combine using the final-test population weights
# 10,000 iterations
# random seed 42


print(
    "\n[9] STRATIFIED BOOTSTRAP 95% CIs"
)


rng = random.Random(
    RANDOM_SEED
)


for system in SYSTEM_ORDER:

    grounded_values = (
        validation_strata[
            system
        ][
            "grounded"
        ]
    )


    hallucinated_values = (
        validation_strata[
            system
        ][
            "hallucinated"
        ]
    )


    grounded_count = (
        population_counts[
            system
        ][
            "grounded"
        ]
    )


    hallucinated_count = (
        population_counts[
            system
        ][
            "hallucinated"
        ]
    )


    total_count = (
        grounded_count
        +
        hallucinated_count
    )


    grounded_weight = (
        grounded_count
        /
        total_count
    )


    hallucinated_weight = (
        hallucinated_count
        /
        total_count
    )


    bootstrap_rates = []


    for _ in range(
        N_BOOTSTRAP
    ):

        sampled_grounded = [

            rng.choice(
                grounded_values
            )

            for _ in range(
                len(
                    grounded_values
                )
            )
        ]


        sampled_hallucinated = [

            rng.choice(
                hallucinated_values
            )

            for _ in range(
                len(
                    hallucinated_values
                )
            )
        ]


        grounded_human_rate = (
            sum(
                sampled_grounded
            )
            /
            len(
                sampled_grounded
            )
        )


        hallucinated_human_rate = (
            sum(
                sampled_hallucinated
            )
            /
            len(
                sampled_hallucinated
            )
        )


        calibrated_bootstrap_rate = (

            grounded_weight
            *
            grounded_human_rate

            +

            hallucinated_weight
            *
            hallucinated_human_rate
        )


        bootstrap_rates.append(
            calibrated_bootstrap_rate
        )


    ci_lower = percentile(
        bootstrap_rates,
        0.025
    )


    ci_upper = percentile(
        bootstrap_rates,
        0.975
    )


    results[
        system
    ][
        "bootstrap_95_ci"
    ] = {

        "lower":
            ci_lower,

        "upper":
            ci_upper,

        "iterations":
            N_BOOTSTRAP,

        "seed":
            RANDOM_SEED
    }


    print(
        f"\n{system}"
    )


    print(
        f"  Calibrated rate : "
        f"{results[system]['human_calibrated_hallucination_rate']:.4f}"
    )


    print(
        f"  95% CI          : "
        f"[{ci_lower:.4f}, "
        f"{ci_upper:.4f}]"
    )


# 14. DESCRIPTIVE ROBUSTNESS COMPARISON

print(
    "\n[10] CALIBRATED DESCRIPTIVE COMPARISON"
)


baseline_rate = (
    results[
        "baseline_llm"
    ][
        "human_calibrated_hallucination_rate"
    ]
)


for system in [

    "basic_rag",
    "advanced_rag",
    "gold_context_control"

]:

    system_rate = (
        results[
            system
        ][
            "human_calibrated_hallucination_rate"
        ]
    )


    difference = (
        baseline_rate
        -
        system_rate
    )


    relative_reduction = (

        difference
        /
        baseline_rate

        if baseline_rate > 0

        else None
    )


    results[
        system
    ][
        "calibrated_absolute_reduction_vs_baseline"
    ] = difference


    results[
        system
    ][
        "calibrated_relative_reduction_vs_baseline"
    ] = relative_reduction


    print(
        f"\nBaseline vs {system}"
    )


    print(
        f"  Absolute reduction : "
        f"{difference:.4f} "
        f"({difference * 100:.2f} pp)"
    )


    print(
        f"  Relative reduction : "
        f"{relative_reduction * 100:.2f}%"
    )


# 15. ADVANCED VS BASIC DESCRIPTIVE DIFFERENCE

# Sensitivity analysis only.

# Do NOT attach a significance interpretation to this
# descriptive calibrated difference.


advanced_minus_basic = (

    results[
        "advanced_rag"
    ][
        "human_calibrated_hallucination_rate"
    ]

    -

    results[
        "basic_rag"
    ][
        "human_calibrated_hallucination_rate"
    ]
)


print(
    "\nAdvanced - Basic calibrated difference : "
    f"{advanced_minus_basic * 100:+.2f} pp"
)


# 16. SAVE

output = {

    "split":
        "test",

    "analysis":
        (
            "pre-specified human-calibrated "
            "sensitivity analysis"
        ),

    "method":
        (
            "Post-stratified human calibration "
            "of frozen automatic V3 hallucination "
            "rates. Within each system, the "
            "final-test claim population is "
            "divided into automatic grounded and "
            "automatic hallucinated strata. "
            "Human hallucination prevalence "
            "measured in the frozen 20-claim "
            "development validation sample from "
            "each stratum is applied to the "
            "final-test population stratum weight."
        ),

    "primary_v3_results_replaced":
        False,

    "inference_performed":
        False,

    "important_limitation":
        (
            "These calibrated estimates correct "
            "only automatic evidence-label "
            "classification. They do not validate "
            "or correct the V3 claim "
            "extraction/exclusion stage."
        ),

    "bootstrap": {

        "iterations":
            N_BOOTSTRAP,

        "seed":
            RANDOM_SEED,

        "resampling":
            (
                "human validation claims "
                "separately within automatic "
                "grounded and hallucinated strata"
            )
    },

    "development_calibration_source":
        str(
            DEVELOPMENT_CALIBRATION_PATH
            .relative_to(
                PROJECT_ROOT
            )
        ).replace(
            "\\",
            "/"
        ),

    "human_validation_hashes": {

        "annotations_sha256":
            sha256_file(
                ANNOTATION_PATH
            ),

        "key_sha256":
            sha256_file(
                KEY_PATH
            ),

        "development_calibration_sha256":
            sha256_file(
                DEVELOPMENT_CALIBRATION_PATH
            ),

        "section_29_sha256":
            sha256_file(
                SECTION_29_PATH
            )
    },

    "systems":
        results,

    "advanced_minus_basic_calibrated_difference":
        advanced_minus_basic
}


save_json(
    output,
    OUTPUT_PATH
)


# 17. SCORECARD

print(
    "\n[11] HUMAN-CALIBRATED SENSITIVITY SCORECARD"
)


print(
    f"{'System':<24}"
    f"{'Raw V3':>12}"
    f"{'Calibrated':>14}"
    f"{'CI low':>12}"
    f"{'CI high':>12}"
)


print(
    "-" * 74
)


for system in SYSTEM_ORDER:

    result = (
        results[
            system
        ]
    )


    print(
        f"{system:<24}"
        f"{result['automatic_hallucination_rate']:>12.4f}"
        f"{result['human_calibrated_hallucination_rate']:>14.4f}"
        f"{result['bootstrap_95_ci']['lower']:>12.4f}"
        f"{result['bootstrap_95_ci']['upper']:>12.4f}"
    )


# 18. FINAL OUTPUT

print(
    "\n[12] FILE SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)


print(
    "SECTION 43 COMPLETE - "
    "HUMAN-CALIBRATED TEST SENSITIVITY ANALYSIS"
)


print(
    "=" * 78
)

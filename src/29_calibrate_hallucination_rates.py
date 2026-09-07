import csv
import json
import random
from pathlib import Path


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation"
)

HALLUCINATION_PATH = (
    GENERATION_DIR
    / "development_hallucination_v3.json"
)

VALIDATION_DIR = (
    GENERATION_DIR
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

OUTPUT_PATH = (
    VALIDATION_DIR
    / "human_calibrated_hallucination_rates.json"
)


SYSTEM_ORDER = [
    "baseline_llm",
    "basic_rag",
    "advanced_rag",
    "gold_context_control"
]


N_BOOTSTRAP = 10000

RANDOM_SEED = 42


# 2. HELPERS

def clean(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def auto_binary(label):

    label = clean(
        label
    )

    if label == "supported":
        return "grounded"

    if label in {
        "unsupported",
        "contradicted"
    }:
        return "hallucinated"

    raise ValueError(
        f"Unexpected automatic label: {label}"
    )


def human_is_hallucinated(label):

    label = clean(
        label
    )

    if label == "grounded":
        return 0

    if label == "hallucinated":
        return 1

    raise ValueError(
        f"Unexpected human label: {label}"
    )


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


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


    encodings = [
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]


    last_error = None


    for encoding in encodings:

        try:

            with path.open(
                "r",
                encoding=encoding,
                newline=""
            ) as file:

                rows = list(
                    csv.DictReader(
                        file
                    )
                )

            print(
                f"CSV encoding used : {encoding}"
            )

            return rows


        except UnicodeDecodeError as error:

            last_error = error


    raise last_error


def percentile(
    values,
    probability
):

    values = sorted(
        values
    )

    if not values:
        return None

    position = (
        probability
        *
        (
            len(values)
            -
            1
        )
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        position
        -
        lower
    )

    return (
        values[lower]
        *
        (
            1
            -
            fraction
        )
        +
        values[upper]
        *
        fraction
    )


# 3. LOAD DATA

print("=" * 78)

print(
    "SECTION 29 - HUMAN-CALIBRATED HALLUCINATION RATES"
)

print("=" * 78)


hallucination_data = load_json(
    HALLUCINATION_PATH
)


human_rows = load_csv(
    ANNOTATION_PATH
)


key_data = load_json(
    KEY_PATH
)


key = key_data[
    "key"
]


print(
    "\n[1] DATA LOADED"
)


print(
    f"Development questions : "
    f"{len(hallucination_data['judgments'])}"
)


print(
    f"Human validation rows : "
    f"{len(human_rows)}"
)


if len(human_rows) != 160:

    raise ValueError(
        "Expected 160 human-validation rows."
    )


# 4. FULL DEVELOPMENT AUTOMATIC CLAIM COUNTS

print(
    "\n[2] FULL DEVELOPMENT AUTOMATIC CLAIM COUNTS"
)


population_counts = {

    system: {
        "grounded": 0,
        "hallucinated": 0
    }

    for system in SYSTEM_ORDER
}


for judgment in hallucination_data[
    "judgments"
]:

    for system in SYSTEM_ORDER:

        system_result = (
            judgment[
                "systems"
            ][
                system
            ]
        )


        for claim in system_result.get(
            "claims",
            []
        ):

            binary_label = auto_binary(
                claim[
                    "label"
                ]
            )


            population_counts[
                system
            ][
                binary_label
            ] += 1


for system in SYSTEM_ORDER:

    grounded = (
        population_counts[
            system
        ][
            "grounded"
        ]
    )

    hallucinated = (
        population_counts[
            system
        ][
            "hallucinated"
        ]
    )

    total = (
        grounded
        +
        hallucinated
    )

    raw_rate = (
        hallucinated
        /
        total
    )


    print(
        f"\n{system}"
    )

    print(
        f"  Grounded claims     : "
        f"{grounded}"
    )

    print(
        f"  Hallucinated claims : "
        f"{hallucinated}"
    )

    print(
        f"  Total claims        : "
        f"{total}"
    )

    print(
        f"  Raw V3 rate         : "
        f"{raw_rate:.4f} "
        f"({raw_rate * 100:.2f}%)"
    )


# 5. HUMAN VALIDATION STRATA

print(
    "\n[3] HUMAN HALLUCINATION RATE WITHIN V3 STRATA"
)


validation_strata = {

    system: {
        "grounded": [],
        "hallucinated": []
    }

    for system in SYSTEM_ORDER
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
            f"Invalid human label for "
            f"{validation_id}: "
            f"{human_label}"
        )


    key_record = key[
        validation_id
    ]


    system = key_record[
        "system"
    ]


    automatic_stratum = clean(
        key_record[
            "automatic_binary"
        ]
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


for system in SYSTEM_ORDER:

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


        if len(values) != 20:

            raise ValueError(
                f"{system}/{stratum} "
                f"should contain 20 "
                f"validation claims, "
                f"found {len(values)}."
            )


        human_rate = (
            sum(values)
            /
            len(values)
        )


        print(
            f"  Auto {stratum:<12}: "
            f"human hallucination rate = "
            f"{human_rate:.4f} "
            f"({human_rate * 100:.2f}%)"
        )


# 6. STRATIFIED HUMAN-CALIBRATED ESTIMATES

print(
    "\n[4] HUMAN-CALIBRATED DEVELOPMENT RATES"
)


results = {}


for system in SYSTEM_ORDER:

    auto_grounded_count = (
        population_counts[
            system
        ][
            "grounded"
        ]
    )

    auto_hallucinated_count = (
        population_counts[
            system
        ][
            "hallucinated"
        ]
    )


    total_claims = (
        auto_grounded_count
        +
        auto_hallucinated_count
    )


    grounded_weight = (
        auto_grounded_count
        /
        total_claims
    )


    hallucinated_weight = (
        auto_hallucinated_count
        /
        total_claims
    )


    human_rate_auto_grounded = (

        sum(
            validation_strata[
                system
            ][
                "grounded"
            ]
        )

        /
        len(
            validation_strata[
                system
            ][
                "grounded"
            ]
        )
    )


    human_rate_auto_hallucinated = (

        sum(
            validation_strata[
                system
            ][
                "hallucinated"
            ]
        )

        /
        len(
            validation_strata[
                system
            ][
                "hallucinated"
            ]
        )
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


    raw_rate = (
        auto_hallucinated_count
        /
        total_claims
    )


    results[
        system
    ] = {

        "total_claim_count":
            total_claims,

        "automatic_grounded_count":
            auto_grounded_count,

        "automatic_hallucinated_count":
            auto_hallucinated_count,

        "automatic_hallucination_rate":
            raw_rate,

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
        f"{raw_rate:.4f} "
        f"({raw_rate * 100:.2f}%)"
    )


    print(
        f"  Human-calibrated  : "
        f"{calibrated_rate:.4f} "
        f"({calibrated_rate * 100:.2f}%)"
    )


# 7. STRATIFIED BOOTSTRAP

print(
    "\n[5] STRATIFIED BOOTSTRAP 95% CIs"
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


# 8. DESCRIPTIVE COMPARISON

print(
    "\n[6] CALIBRATED DESCRIPTIVE COMPARISON"
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
        f"{relative_reduction:.4f} "
        f"({relative_reduction * 100:.2f}%)"
    )



# 9. SAVE


output = {

    "method": (
        "Post-stratified human calibration of "
        "automatic V3 hallucination rates. "
        "Within each system, the full development "
        "claim population is divided into automatic "
        "grounded and automatic hallucinated strata. "
        "Human hallucination prevalence measured "
        "within the 20-claim validation sample from "
        "each stratum is applied to the full "
        "population stratum weight."
    ),

    "important_limitation": (
        "These calibrated estimates correct only "
        "automatic evidence-label classification. "
        "They do not validate or correct the V3 "
        "claim extraction/exclusion stage."
    ),

    "bootstrap": {
        "iterations":
            N_BOOTSTRAP,

        "seed":
            RANDOM_SEED
    },

    "systems":
        results
}


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "\n[7] RESULTS SAVED"
)


print(
    OUTPUT_PATH
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 29 COMPLETE"
)

print(
    "=" * 78
)

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TALENTPULSE AI - GROUND TRUTH DATASET GENERATOR
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

EVALUATION_DIR = PROCESSED_DIR / "transformer_evaluation"

CANDIDATE_FILE = PROCESSED_DIR / "candidates_ml_ready.parquet"
JD_FILE = PROCESSED_DIR / "jd_requirements.json"

OUTPUT_FILE = EVALUATION_DIR / "candidate_review_template.csv"

RANDOM_CONTROL_COUNT = 50
RANDOM_STATE = 42

MODEL_NAMES = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-mpnet-base-cos-v1",
    "msmarco-MiniLM-L6-cos-v5",
    "bge-small-en-v1.5",
    "bge-base-en-v1.5",
    "e5-small-v2",
    "gte-small"
]


def load_jd():

    with open(JD_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model_results():

    all_results = []

    for model_name in MODEL_NAMES:

        file_path = (
            EVALUATION_DIR / f"top20_{model_name}.csv"
        )

        if not file_path.exists():
            print(f"WARNING: Missing result file: {model_name}")
            continue

        df = pd.read_csv(file_path)

        if "record_id" not in df.columns:
            print(f"WARNING: record_id missing in {model_name}")
            continue

        if "transformer_similarity" not in df.columns:
            print(
                f"WARNING: Similarity column missing in {model_name}"
            )
            continue

        df = df.reset_index(drop=True)

        df["model_name"] = model_name
        df["model_rank"] = np.arange(1, len(df) + 1)

        all_results.append(
            df[
                [
                    "record_id",
                    "model_name",
                    "model_rank",
                    "transformer_similarity"
                ]
            ]
        )

        print(
            f"Loaded {len(df)} candidates from {model_name}"
        )

    if not all_results:
        raise ValueError("No model result files were found.")

    return pd.concat(all_results, ignore_index=True)


def create_candidate_pool():

    print("\nLoading model recommendations...")

    results = load_model_results()

    # Candidate IDs retrieved by at least one model.
    retrieved_ids = results["record_id"].dropna().unique()

    print(
        "\nUnique candidates retrieved:",
        len(retrieved_ids)
    )

    # --------------------------------------------------------
    # MERGE ALL MODEL RANKINGS
    # --------------------------------------------------------

    rankings = results.pivot_table(
        index="record_id",
        columns="model_name",
        values="model_rank",
        aggfunc="min"
    )

    rankings.columns = [
        f"rank_{column}" for column in rankings.columns
    ]

    similarities = results.pivot_table(
        index="record_id",
        columns="model_name",
        values="transformer_similarity",
        aggfunc="max"
    )

    similarities.columns = [
        f"similarity_{column}" for column in similarities.columns
    ]

    # --------------------------------------------------------
    # LOAD CANDIDATE DETAILS
    # --------------------------------------------------------

    print("\nLoading candidate details...")

    candidates = pd.read_parquet(CANDIDATE_FILE)

    if "record_id" not in candidates.columns:
        raise ValueError("record_id column missing.")

    candidate_columns = [
        "record_id",
        "experience_years",
        "current_role",
        "previous_role",
        "role_family",
        "key_skills",
        "education_level",
        "candidate_match_text"
    ]

    candidate_columns = [
        column for column in candidate_columns
        if column in candidates.columns
    ]

    candidates = candidates[candidate_columns].copy()

    # One record per candidate ID.
    candidates = candidates.drop_duplicates(
        subset="record_id"
    )

    # --------------------------------------------------------
    # ADD RANDOM CONTROL CANDIDATES
    # --------------------------------------------------------

    retrieved_set = set(retrieved_ids)

    controls = candidates[
        ~candidates["record_id"].isin(retrieved_set)
    ]

    control_count = min(
        RANDOM_CONTROL_COUNT,
        len(controls)
    )

    controls = controls.sample(
        n=control_count,
        random_state=RANDOM_STATE
    )

    controls["candidate_source"] = "RANDOM_CONTROL"

    print(
        "Random control candidates added:",
        len(controls)
    )

    # --------------------------------------------------------
    # CREATE REVIEW POOL
    # --------------------------------------------------------

    retrieved = candidates[
        candidates["record_id"].isin(retrieved_ids)
    ].copy()

    retrieved["candidate_source"] = "MODEL_RETRIEVED"

    pool = pd.concat(
        [retrieved, controls],
        ignore_index=True
    )

    # Attach all model rankings and similarities.
    pool = pool.merge(
        rankings,
        left_on="record_id",
        right_index=True,
        how="left"
    )

    pool = pool.merge(
        similarities,
        left_on="record_id",
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # REVIEW COLUMNS
    # --------------------------------------------------------

    pool["human_relevance"] = pd.Series(
        pd.NA,
        index=pool.index,
        dtype="Int64"
    )

    pool["reviewer_notes"] = ""

    # Keep model ranking information before review fields.
    model_columns = sorted([
        column for column in pool.columns
        if column.startswith("rank_")
        or column.startswith("similarity_")
    ])

    preferred_columns = [
        "record_id",
        "candidate_source",
        "experience_years",
        "current_role",
        "previous_role",
        "role_family",
        "key_skills",
        "education_level",
        "candidate_match_text"
    ]

    final_columns = (
        [c for c in preferred_columns if c in pool.columns]
        + model_columns
        + ["human_relevance", "reviewer_notes"]
    )

    pool = pool[final_columns]

    # Put candidates with more model appearances first.
    rank_columns = [
        column for column in pool.columns
        if column.startswith("rank_")
    ]

    pool["models_retrieving_candidate"] = (
        pool[rank_columns].notna().sum(axis=1)
    )

    pool = pool.sort_values(
        by="models_retrieving_candidate",
        ascending=False
    )

    return pool


def main():

    print("\n========================================")
    print("TALENTPULSE AI - GROUND TRUTH GENERATOR")
    print("========================================")

    if not EVALUATION_DIR.exists():
        raise FileNotFoundError(EVALUATION_DIR)

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(CANDIDATE_FILE)

    if not JD_FILE.exists():
        raise FileNotFoundError(JD_FILE)

    jd = load_jd()

    print("\nCurrent job title:", jd.get("job_title"))

    pool = create_candidate_pool()

    pool.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== REVIEW DATASET SUMMARY ==========")

    print("Candidates to review:", len(pool))

    print(
        "Retrieved by at least one model:",
        (pool["candidate_source"] == "MODEL_RETRIEVED").sum()
    )

    print(
        "Random control candidates:",
        (pool["candidate_source"] == "RANDOM_CONTROL").sum()
    )

    print("\nReview file saved to:")
    print(OUTPUT_FILE)

    print("\nNext action:")
    print("Open the CSV and fill human_relevance with 0, 1, 2 or 3.")
    print("Do not modify the model ranking or similarity columns.")

    print("\n========================================")
    print("GROUND TRUTH TEMPLATE CREATED")
    print("========================================")


if __name__ == "__main__":
    main()

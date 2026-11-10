
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# TALENTPULSE AI - CANDIDATE SCORING AND RANKING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

CANDIDATE_FILE = PROCESSED_DIR / "candidates_ml_ready.parquet"
JD_FILE = PROCESSED_DIR / "jd_requirements.json"
WEIGHTS_FILE = PROCESSED_DIR / "jd_dynamic_weights.json"

ALL_SCORES_FILE = PROCESSED_DIR / "candidates_scored.parquet"
TOP20_FILE = PROCESSED_DIR / "top20_ranked_candidates.csv"

TOP_N = 20


# ============================================================
# 1. LOAD INPUT FILES
# ============================================================

def load_inputs():

    for file_path in [CANDIDATE_FILE, JD_FILE, WEIGHTS_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    print("\nLoading candidate dataset...")

    candidates = pd.read_parquet(CANDIDATE_FILE)

    with open(JD_FILE, "r", encoding="utf-8") as file:
        jd = json.load(file)

    with open(WEIGHTS_FILE, "r", encoding="utf-8") as file:
        weight_data = json.load(file)

    print(f"Candidates loaded: {len(candidates):,}")

    return candidates, jd, weight_data["weights"]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value).strip().lower()


def parse_skills(value):
    """
    Convert candidate skills into a consistent Python list.
    Supports lists, arrays and delimited strings.
    """

    if value is None:
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [
            clean_text(skill)
            for skill in value
            if clean_text(skill)
        ]

    if isinstance(value, float) and pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Handle string representations of Python lists
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return [
                clean_text(skill)
                for skill in parsed
                if clean_text(skill)
            ]
    except (ValueError, SyntaxError):
        pass

    return [
        clean_text(skill)
        for skill in re.split(r"[,;|\n]", text)
        if clean_text(skill)
    ]


def tokenize(text):

    return set(re.findall(r"[a-z0-9+#.]+", clean_text(text)))


# ============================================================
# 3. SKILL MATCHING SCORE
# ============================================================

def calculate_skill_score(candidate_skills, required, preferred):

    candidate_skills = set(candidate_skills)

    required = [clean_text(s) for s in required]
    preferred = [clean_text(s) for s in preferred]

    matched_required = [
        skill for skill in required
        if skill in candidate_skills
    ]

    matched_preferred = [
        skill for skill in preferred
        if skill in candidate_skills
    ]

    missing_required = [
        skill for skill in required
        if skill not in candidate_skills
    ]

    # Required skills carry more importance than preferred skills.
    if required and preferred:

        score = (
            0.8 * (len(matched_required) / len(required))
            + 0.2 * (len(matched_preferred) / len(preferred))
        ) * 100

    elif required:

        score = (
            len(matched_required) / len(required)
        ) * 100

    elif preferred:

        score = (
            len(matched_preferred) / len(preferred)
        ) * 100

    else:
        score = 100.0

    return (
        round(score, 2),
        matched_required,
        matched_preferred,
        missing_required
    )


# ============================================================
# 4. EXPERIENCE MATCHING SCORE
# ============================================================

def calculate_experience_score(
    experience,
    minimum_experience,
    maximum_experience
):

    if pd.isna(experience):
        return 0.0

    experience = float(experience)

    if minimum_experience is None and maximum_experience is None:
        return 100.0

    if minimum_experience is not None:

        if experience >= minimum_experience:
            # Meeting the minimum requirement earns full score.
            # Exceeding the maximum is not automatically a failure.
            return 100.0

        if minimum_experience == 0:
            return 100.0

        return round(
            (experience / minimum_experience) * 100, 2
        )

    if maximum_experience is not None:

        if experience <= maximum_experience:
            return 100.0

        return 80.0

    return 100.0


# ============================================================
# 5. ROLE RELEVANCE SCORE
# ============================================================

def calculate_role_score(role_text, job_title):

    job_tokens = tokenize(job_title)
    candidate_tokens = tokenize(role_text)

    if not job_tokens:
        return 100.0

    if not candidate_tokens:
        return 0.0

    overlap = len(job_tokens.intersection(candidate_tokens))

    return round(
        (overlap / len(job_tokens)) * 100, 2
    )


# ============================================================
# 6. EDUCATION MATCHING SCORE
# ============================================================

def calculate_education_score(candidate_education, requirements):

    candidate_education = clean_text(candidate_education)

    if not requirements:
        return 100.0

    if not candidate_education:
        return 0.0

    # Common education wording variations
    candidate_tokens = tokenize(candidate_education)

    for requirement in requirements:

        requirement = clean_text(requirement)

        if requirement in candidate_education:
            return 100.0

        requirement_tokens = tokenize(requirement)

        if not requirement_tokens:
            continue

        # Handle variations such as "bachelor" and "bachelor's degree"
        if requirement_tokens.issubset(candidate_tokens):
            return 100.0

        # Partial education match receives a proportional score.
        overlap = len(requirement_tokens.intersection(candidate_tokens))

        if overlap:
            partial_score = (
                overlap / len(requirement_tokens)
            ) * 100

            return round(partial_score, 2)

    return 0.0


# ============================================================
# 7. TF-IDF TEXT SIMILARITY
# ============================================================

def calculate_text_similarity(candidates, jd):

    print("\nCalculating TF-IDF text similarity...")
    print("This stage may take some time for 384,035 profiles.")

    candidate_text = (
        candidates["candidate_match_text"]
        .fillna("")
        .astype(str)
    )

    skills = jd.get("skills", {})

    jd_text = " ".join([
        str(jd.get("job_title", "")),
        " ".join(skills.get("required", [])),
        " ".join(skills.get("preferred", [])),
        " ".join(jd.get("education", []))
    ])

    documents = candidate_text.tolist() + [jd_text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32
    )

    matrix = vectorizer.fit_transform(documents)

    print("TF-IDF matrix:", matrix.shape)

    jd_vector = matrix[-1]
    candidate_matrix = matrix[:-1]

    similarities = cosine_similarity(
        candidate_matrix,
        jd_vector
    ).ravel()

    print("Text similarity calculated.")

    # Convert similarity from 0-1 into 0-100.
    return np.round(similarities * 100, 2)


# ============================================================
# 8. CANDIDATE SCORING
# ============================================================

def score_candidates(candidates, jd, weights):

    print("\nPreparing candidate scoring...")

    required = jd.get("skills", {}).get("required", [])
    preferred = jd.get("skills", {}).get("preferred", [])

    minimum_experience = jd.get(
        "experience", {}
    ).get("minimum_years")

    maximum_experience = jd.get(
        "experience", {}
    ).get("maximum_years")

    education_requirements = jd.get("education", [])

    # Validate required candidate columns
    required_columns = [
        "experience_years",
        "role_text",
        "education_level",
        "candidate_match_text"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in candidates.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in candidate dataset: {missing_columns}"
        )

    if "skills_normalized" in candidates.columns:
        skill_column = "skills_normalized"
    elif "key_skills" in candidates.columns:
        skill_column = "key_skills"
    else:
        raise ValueError(
            "Neither skills_normalized nor key_skills exists."
        )

    # Check that every active weight has a corresponding scoring dimension.
    score_dimensions = {
        "skills",
        "experience",
        "role_relevance",
        "education",
        "semantic_similarity"
    }

    if set(weights) != score_dimensions:
        raise ValueError(
            "Weight dimensions do not match the five scoring "
            f"dimensions. Found: {list(weights.keys())}"
        )

    # Normalize weights to fractions.
    weight_fractions = {
        key: value / 100
        for key, value in weights.items()
    }

    total_weight = sum(weight_fractions.values())

    if not np.isclose(total_weight, 1.0, atol=0.001):
        raise ValueError(
            f"Invalid weight total: {total_weight * 100}"
        )

    print("Calculating skill, experience, role and education scores...")

    skills_data = candidates[skill_column].apply(parse_skills)

    skill_results = [
        calculate_skill_score(
            candidate_skills,
            required,
            preferred
        )
        for candidate_skills in skills_data
    ]

    candidates["skill_score"] = [
        result[0] for result in skill_results
    ]

    candidates["matched_required_skills"] = [
        ", ".join(result[1]) for result in skill_results
    ]

    candidates["matched_preferred_skills"] = [
        ", ".join(result[2]) for result in skill_results
    ]

    candidates["missing_required_skills"] = [
        ", ".join(result[3]) for result in skill_results
    ]

    candidates["experience_score"] = candidates[
        "experience_years"
    ].apply(
        lambda value: calculate_experience_score(
            value,
            minimum_experience,
            maximum_experience
        )
    )

    candidates["role_score"] = [
        calculate_role_score(role, jd.get("job_title", ""))
        for role in candidates["role_text"]
    ]

    candidates["education_score"] = [
        calculate_education_score(education, education_requirements)
        for education in candidates["education_level"]
    ]

    # TF-IDF similarity
    candidates["text_similarity_score"] = calculate_text_similarity(
        candidates, jd
    )

    # Keep the column name aligned with Step 22's weight key.
    candidates["semantic_similarity_score"] = candidates[
        "text_similarity_score"
    ]

    # --------------------------------------------------------
    # 9. WEIGHTED OVERALL MATCHING SCORE
    # --------------------------------------------------------

    print("\nCalculating final weighted scores...")

    candidates["overall_score"] = (
        candidates["skill_score"] * weight_fractions["skills"]
        + candidates["experience_score"] * weight_fractions["experience"]
        + candidates["role_score"] * weight_fractions["role_relevance"]
        + candidates["education_score"] * weight_fractions["education"]
        + candidates["semantic_similarity_score"]
        * weight_fractions["semantic_similarity"]
    ).round(2)

    # --------------------------------------------------------
    # 10. MINIMUM EXPERIENCE ELIGIBILITY
    # --------------------------------------------------------

    if minimum_experience is not None:

        candidates["eligibility_status"] = np.where(
            pd.to_numeric(
                candidates["experience_years"],
                errors="coerce"
            ) >= minimum_experience,
            "ELIGIBLE",
            "NOT_ELIGIBLE"
        )

    else:
        candidates["eligibility_status"] = "ELIGIBLE"

    # Only eligible candidates enter the ranking.
    candidates["rank"] = pd.Series(
        pd.NA,
        index=candidates.index,
        dtype="Int64"
    )

    eligible_indices = candidates.loc[
        candidates["eligibility_status"] == "ELIGIBLE"
    ].sort_values(
        by="overall_score",
        ascending=False
    ).index

    candidates.loc[eligible_indices, "rank"] = range(
        1, len(eligible_indices) + 1
    )

    return candidates


# ============================================================
# 11. SAVE RESULTS
# ============================================================

def save_results(candidates):

    print("\nSaving complete candidate scoring results...")

    candidates.to_parquet(
        ALL_SCORES_FILE,
        index=False
    )

    eligible = candidates[
        candidates["eligibility_status"] == "ELIGIBLE"
    ]

    top_candidates = eligible.sort_values(
        by="overall_score",
        ascending=False
    ).head(TOP_N)

    top_candidates.to_csv(
        TOP20_FILE,
        index=False
    )

    print("\n========== RESULTS ==========")

    print("Total candidates:", f"{len(candidates):,}")

    print(
        "Eligible candidates:",
        f"{(candidates['eligibility_status'] == 'ELIGIBLE').sum():,}"
    )

    print(
        "Not eligible candidates:",
        f"{(candidates['eligibility_status'] == 'NOT_ELIGIBLE').sum():,}"
    )

    print("\n========== TOP 20 CANDIDATES ==========")

    display_columns = [
        column for column in [
            "record_id",
            "experience_years",
            "current_role",
            "role_family",
            "overall_score",
            "eligibility_status",
            "rank"
        ]
        if column in top_candidates.columns
    ]

    print(
        top_candidates[display_columns].to_string(index=False)
    )

    print("\nComplete scores saved to:")
    print(ALL_SCORES_FILE)

    print("\nTop 20 candidates saved to:")
    print(TOP20_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" TALENTPULSE AI: CANDIDATE MATCHING ENGINE")
    print("==========================================")

    candidates, jd, weights = load_inputs()

    print("\nJob title:", jd.get("job_title"))

    print("\nWeights being used:")

    for criterion, weight in weights.items():
        print(f"{criterion}: {weight}%")

    scored_candidates = score_candidates(
        candidates,
        jd,
        weights
    )

    save_results(scored_candidates)

    print("\n==========================================")
    print(" CANDIDATE SCORING COMPLETED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()

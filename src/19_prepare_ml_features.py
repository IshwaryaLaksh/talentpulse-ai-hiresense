
import pandas as pd
import re
from pathlib import Path

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_employment_confidence_all.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "candidates_ml_ready.parquet"
SAMPLE_FILE = OUTPUT_DIR / "candidates_ml_sample.csv"

print("\n========== TALENTPULSE AI: ML FEATURE PREPARATION ==========")


# =====================================================
# 2. LOAD DATA
# =====================================================

df = pd.read_parquet(INPUT_FILE)

print("Total candidate records:", len(df))


# =====================================================
# 3. VALIDATE AND IDENTIFY COLUMNS
# =====================================================

required_columns = ["record_id", "experience_years"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

optional_columns = [
    "current_role",
    "previous_role",
    "role_family",
    "key_skills",
    "n_skills",
    "education_level",
    "grad_year"
]

for col in optional_columns:
    if col not in df.columns:
        df[col] = None


# =====================================================
# 4. MISSING VALUE HANDLING
# =====================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in [
        "none", "nan", "null", "n/a",
        "not available", "not provided"
    ]:
        return ""

    return value


text_columns = [
    "current_role",
    "previous_role",
    "role_family",
    "education_level",
    "key_skills"
]

for col in text_columns:
    df[col] = df[col].apply(clean_text)


# =====================================================
# 5. SKILL STANDARDIZATION
# =====================================================

def normalize_skill(skill):

    skill = skill.lower().strip()

    # Normalize extra spaces
    skill = re.sub(r"\s+", " ", skill)

    # Remove surrounding punctuation
    skill = skill.strip(" .;")

    return skill


def extract_skills(skill_text):

    if not skill_text:
        return []

    # Split common skill delimiters
    raw_skills = re.split(r"[,;|\n\r]+", skill_text)

    normalized_skills = []

    for skill in raw_skills:

        skill = normalize_skill(skill)

        if skill and skill not in normalized_skills:
            normalized_skills.append(skill)

    return normalized_skills


df["skills_normalized"] = df["key_skills"].apply(
    extract_skills
)

df["skills_text"] = df["skills_normalized"].apply(
    lambda skills: " ".join(skills)
)

df["actual_skill_count"] = df["skills_normalized"].apply(
    len
)


# =====================================================
# 6. EXPERIENCE FEATURE
# =====================================================

df["experience_years"] = pd.to_numeric(
    df["experience_years"],
    errors="coerce"
)

def experience_band(years):

    if pd.isna(years):
        return "unknown"

    if years < 0:
        return "invalid"

    if years < 1:
        return "less_than_1_year"

    if years < 3:
        return "1_to_3_years"

    if years < 5:
        return "3_to_5_years"

    if years < 8:
        return "5_to_8_years"

    if years < 12:
        return "8_to_12_years"

    return "12_plus_years"


df["ml_experience_band"] = df["experience_years"].apply(
    experience_band
)


# =====================================================
# 7. ROLE FEATURES
# =====================================================

df["role_text"] = (
    df["current_role"] + " " +
    df["previous_role"] + " " +
    df["role_family"]
).str.strip()

df["role_text"] = df["role_text"].str.lower()


# =====================================================
# 8. EDUCATION FEATURES
# =====================================================

df["education_text"] = (
    df["education_level"]
    .str.lower()
    .str.strip()
)

df["graduation_year"] = pd.to_numeric(
    df["grad_year"],
    errors="coerce"
)


# =====================================================
# 9. CREATE COMBINED MATCHING TEXT
# =====================================================

df["candidate_match_text"] = (
    df["role_text"] + " " +
    df["skills_text"] + " " +
    df["education_text"]
).str.replace(r"\s+", " ", regex=True).str.strip()


# =====================================================
# 10. FEATURE AVAILABILITY FLAGS
# =====================================================

df["has_skills"] = df["actual_skill_count"] > 0

df["has_role_information"] = (
    df["role_text"].str.strip() != ""
)

df["has_education_information"] = (
    df["education_text"].str.strip() != ""
)

df["has_experience_information"] = (
    df["experience_years"].notna()
    & (df["experience_years"] >= 0)
)


# =====================================================
# 11. SELECT FINAL ML FEATURES
# =====================================================

final_columns = [
    "record_id",
    "current_role",
    "previous_role",
    "role_family",
    "role_text",
    "key_skills",
    "skills_normalized",
    "skills_text",
    "actual_skill_count",
    "experience_years",
    "ml_experience_band",
    "education_level",
    "education_text",
    "graduation_year",
    "candidate_match_text",
    "has_skills",
    "has_role_information",
    "has_education_information",
    "has_experience_information",
    "employment_confidence_status"
]

final_columns = [
    col for col in final_columns
    if col in df.columns
]

ml_df = df[final_columns].copy()


# =====================================================
# 12. QUALITY CHECKS
# =====================================================

print("\n========== ML FEATURE QUALITY ==========")

print("Total candidates:", len(ml_df))

print("Candidates with skills:", ml_df["has_skills"].sum())

print(
    "Candidates with role information:",
    ml_df["has_role_information"].sum()
)

print(
    "Candidates with experience:",
    ml_df["has_experience_information"].sum()
)

print(
    "Candidates with education:",
    ml_df["has_education_information"].sum()
)

print(
    "Candidates with matching text:",
    (ml_df["candidate_match_text"] != "").sum()
)


# =====================================================
# 13. SAMPLE OUTPUT
# =====================================================

print("\n========== SAMPLE CANDIDATES ==========")

print(
    ml_df[
        [
            "record_id",
            "skills_text",
            "role_text",
            "experience_years",
            "candidate_match_text"
        ]
    ].head(5).to_string(index=False)
)


# =====================================================
# 14. SAVE ML-READY DATASET
# =====================================================

ml_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

ml_df.head(100).to_csv(
    SAMPLE_FILE,
    index=False
)

print("\n========== OUTPUT FILES ==========")

print("ML dataset:", OUTPUT_FILE)
print("Sample dataset:", SAMPLE_FILE)

print("\n========== STEP 19 COMPLETED ==========")

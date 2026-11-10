
import pandas as pd
from pathlib import Path

# =====================================================
# 1. FILE PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_employment_analysis.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\n========== EMPLOYMENT CONFIDENCE ASSESSMENT ==========")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))


# =====================================================
# 3. VALIDATE REQUIRED COLUMNS
# =====================================================

required_columns = [
    "record_id",
    "experience_years",
    "current_role",
    "previous_role"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def is_missing(value):

    if pd.isna(value):
        return True

    return str(value).strip().lower() in [
        "", "none", "nan", "null",
        "n/a", "not available",
        "not provided"
    ]


def has_information(value):

    return not is_missing(value)


# =====================================================
# 5. EXPERIENCE VALIDATION
# =====================================================

df["experience_years"] = pd.to_numeric(
    df["experience_years"],
    errors="coerce"
)

df["experience_data_status"] = "VALID"

df.loc[
    df["experience_years"].isna(),
    "experience_data_status"
] = "MISSING"

df.loc[
    df["experience_years"] < 0,
    "experience_data_status"
] = "INVALID"


# =====================================================
# 6. EMPLOYMENT FIELD COMPLETENESS
# =====================================================

df["current_role_available"] = (
    df["current_role"].apply(has_information)
)

df["previous_role_available"] = (
    df["previous_role"].apply(has_information)
)

def classify_role_completeness(row):

    current = row["current_role_available"]
    previous = row["previous_role_available"]

    if current and previous:
        return "BOTH_ROLES_AVAILABLE"

    if current:
        return "CURRENT_ROLE_ONLY"

    if previous:
        return "PREVIOUS_ROLE_ONLY"

    return "BOTH_ROLES_MISSING"


df["role_completeness"] = df.apply(
    classify_role_completeness,
    axis=1
)


# =====================================================
# 7. SKILLS AND PROFILE INFORMATION
# =====================================================

if "n_skills" in df.columns:

    df["n_skills"] = pd.to_numeric(
        df["n_skills"],
        errors="coerce"
    )

    df["skills_available"] = (
        df["n_skills"].fillna(0) > 0
    )

elif "key_skills" in df.columns:

    df["skills_available"] = (
        df["key_skills"].apply(has_information)
    )

else:

    df["skills_available"] = False


if "has_profile_summary" in df.columns:

    df["profile_summary_available"] = (
        df["has_profile_summary"]
        .fillna(False)
        .astype(bool)
    )

else:

    df["profile_summary_available"] = False


# =====================================================
# 8. EMPLOYMENT DATA CLASSIFICATION
# =====================================================

def classify_employment(row):

    experience = row["experience_years"]

    current = row["current_role_available"]
    previous = row["previous_role_available"]

    experience_status = row["experience_data_status"]

    if experience_status == "INVALID":
        return "INVALID_EXPERIENCE"

    if experience_status == "MISSING":
        return "EXPERIENCE_UNKNOWN"

    if experience == 0:

        if not current and not previous:
            return "POTENTIAL_FRESHER"

        return "ZERO_EXPERIENCE_WITH_ROLE"

    if current and previous:
        return "EMPLOYMENT_VERIFIED"

    if current or previous:
        return "EMPLOYMENT_PARTIAL"

    return "EXPERIENCE_UNVERIFIED"


df["employment_confidence_status"] = df.apply(
    classify_employment,
    axis=1
)


# =====================================================
# 9. PROFILE SUPPORTING INFORMATION
# =====================================================

def classify_profile_completeness(row):

    signals = [
        row["skills_available"],
        row["profile_summary_available"]
    ]

    if all(signals):
        return "GOOD_SUPPORTING_INFORMATION"

    if any(signals):
        return "PARTIAL_SUPPORTING_INFORMATION"

    return "LIMITED_SUPPORTING_INFORMATION"


df["profile_information_status"] = df.apply(
    classify_profile_completeness,
    axis=1
)


# =====================================================
# 10. INVESTIGATION PRIORITY
# =====================================================

def assign_investigation_priority(row):

    status = row["employment_confidence_status"]

    if status in [
        "INVALID_EXPERIENCE",
        "ZERO_EXPERIENCE_WITH_ROLE"
    ]:
        return "HIGH"

    if status in [
        "EXPERIENCE_UNVERIFIED",
        "EXPERIENCE_UNKNOWN"
    ]:
        return "REVIEW"

    return "NO_IMMEDIATE_REVIEW"


df["employment_review_priority"] = df.apply(
    assign_investigation_priority,
    axis=1
)


# =====================================================
# 11. CREATE FINAL DATASETS
# =====================================================

verified_candidates = df[
    df["employment_confidence_status"]
    == "EMPLOYMENT_VERIFIED"
].copy()

partial_candidates = df[
    df["employment_confidence_status"]
    == "EMPLOYMENT_PARTIAL"
].copy()

unverified_candidates = df[
    df["employment_confidence_status"]
    == "EXPERIENCE_UNVERIFIED"
].copy()

potential_freshers = df[
    df["employment_confidence_status"]
    == "POTENTIAL_FRESHER"
].copy()

review_candidates = df[
    df["employment_review_priority"].isin(
        ["HIGH", "REVIEW"]
    )
].copy()


# =====================================================
# 12. GENERATE REPORTS
# =====================================================

print("\n========== EMPLOYMENT CLASSIFICATION ==========")

print(
    df["employment_confidence_status"]
    .value_counts(dropna=False)
)

print("\n========== ROLE COMPLETENESS ==========")

print(
    df["role_completeness"]
    .value_counts(dropna=False)
)

print("\n========== PROFILE INFORMATION ==========")

print(
    df["profile_information_status"]
    .value_counts(dropna=False)
)

print("\n========== INVESTIGATION PRIORITY ==========")

print(
    df["employment_review_priority"]
    .value_counts(dropna=False)
)


# =====================================================
# 13. SAVE FINAL OUTPUTS
# =====================================================

df.to_parquet(
    OUTPUT_DIR / "candidates_employment_confidence_all.parquet",
    index=False
)

verified_candidates.to_parquet(
    OUTPUT_DIR / "candidates_employment_verified.parquet",
    index=False
)

partial_candidates.to_parquet(
    OUTPUT_DIR / "candidates_employment_partial.parquet",
    index=False
)

unverified_candidates.to_parquet(
    OUTPUT_DIR / "candidates_employment_unverified.parquet",
    index=False
)

potential_freshers.to_parquet(
    OUTPUT_DIR / "candidates_potential_freshers.parquet",
    index=False
)

review_candidates.to_csv(
    OUTPUT_DIR / "candidates_employment_review.csv",
    index=False
)


# =====================================================
# 14. FINAL SUMMARY
# =====================================================

print("\n========== FINAL SUMMARY ==========")

print("Total candidates:", len(df))

print("Employment verified:", len(verified_candidates))

print("Employment partial:", len(partial_candidates))

print("Employment unverified:", len(unverified_candidates))

print("Potential freshers:", len(potential_freshers))

print("Candidates requiring review:", len(review_candidates))

print("\nOUTPUT FILES GENERATED:")

print("- candidates_employment_confidence_all.parquet")
print("- candidates_employment_verified.parquet")
print("- candidates_employment_partial.parquet")
print("- candidates_employment_unverified.parquet")
print("- candidates_potential_freshers.parquet")
print("- candidates_employment_review.csv")

print("\n========== ASSESSMENT COMPLETED ==========")

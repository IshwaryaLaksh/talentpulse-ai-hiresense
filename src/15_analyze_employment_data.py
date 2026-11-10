
import pandas as pd
from pathlib import Path

# =====================================================
# 1. FILE PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_classified_all.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. LOAD DATA
# =====================================================

print("\n========== EMPLOYMENT DATA ANALYSIS ==========")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))

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
    raise ValueError(f"Missing columns: {missing_columns}")

# =====================================================
# 3. EXPERIENCE ANALYSIS
# =====================================================

df["experience_years"] = pd.to_numeric(
    df["experience_years"],
    errors="coerce"
)

print("\n========== EXPERIENCE STATISTICS ==========")

print(df["experience_years"].describe())

print("\nMissing experience:",
      df["experience_years"].isna().sum())

print("Zero experience:",
      (df["experience_years"] == 0).sum())

print("Negative experience:",
      (df["experience_years"] < 0).sum())

# =====================================================
# 4. CREATE EXPERIENCE BANDS
# =====================================================

def create_experience_band(value):

    if pd.isna(value):
        return "UNKNOWN"

    if value < 0:
        return "INVALID"

    if value <= 1:
        return "0-1 years"

    if value <= 2:
        return "1-2 years"

    if value <= 5:
        return "2-5 years"

    if value <= 10:
        return "5-10 years"

    return "10+ years"


df["analysis_experience_band"] = (
    df["experience_years"].apply(
        create_experience_band
    )
)

print("\n========== EXPERIENCE BAND DISTRIBUTION ==========")

print(
    df["analysis_experience_band"].value_counts(
        dropna=False
    )
)

# =====================================================
# 5. IDENTIFY MISSING ROLES
# =====================================================

def is_missing(value):

    if pd.isna(value):
        return True

    return str(value).strip().lower() in [
        "", "none", "nan", "null"
    ]


df["current_missing"] = (
    df["current_role"].apply(is_missing)
)

df["previous_missing"] = (
    df["previous_role"].apply(is_missing)
)

# =====================================================
# 6. ROLE AVAILABILITY ANALYSIS
# =====================================================

print("\n========== ROLE AVAILABILITY ==========")

role_distribution = pd.crosstab(
    df["current_missing"],
    df["previous_missing"],
    margins=True
)

print(role_distribution)

# =====================================================
# 7. EXPERIENCE VS ROLE AVAILABILITY
# =====================================================

print("\n========== EXPERIENCE VS ROLE AVAILABILITY ==========")

experience_role_analysis = pd.crosstab(
    [
        df["current_missing"],
        df["previous_missing"]
    ],
    df["analysis_experience_band"],
    margins=True
)

print(experience_role_analysis)

# =====================================================
# 8. CANDIDATE EMPLOYMENT CLASSIFICATION
# =====================================================

def classify_employment(row):

    experience = row["experience_years"]

    current_missing = row["current_missing"]
    previous_missing = row["previous_missing"]

    if pd.isna(experience):
        return "EXPERIENCE_UNKNOWN"

    if experience < 0:
        return "INVALID_EXPERIENCE"

    if experience == 0:

        if current_missing and previous_missing:
            return "POTENTIAL_FRESHER"

        return "ZERO_EXPERIENCE_WITH_ROLE"

    if current_missing and previous_missing:
        return "EXPERIENCED_WITHOUT_ROLE"

    if current_missing or previous_missing:
        return "EMPLOYMENT_INFORMATION_PARTIAL"

    return "EMPLOYMENT_INFORMATION_AVAILABLE"


df["employment_classification"] = df.apply(
    classify_employment,
    axis=1
)

print("\n========== EMPLOYMENT CLASSIFICATION ==========")

print(
    df["employment_classification"].value_counts(
        dropna=False
    )
)

# =====================================================
# 9. IDENTIFY CANDIDATES NEEDING INVESTIGATION
# =====================================================

investigation_candidates = df[
    df["employment_classification"].isin([
        "EXPERIENCED_WITHOUT_ROLE",
        "EXPERIENCE_UNKNOWN",
        "INVALID_EXPERIENCE",
        "ZERO_EXPERIENCE_WITH_ROLE"
    ])
].copy()

print("\n========== INVESTIGATION SUMMARY ==========")

print(
    "Candidates needing investigation:",
    len(investigation_candidates)
)

# =====================================================
# 10. INSPECT CANDIDATES WITH BOTH ROLES MISSING
# =====================================================

both_roles_missing = df[
    df["current_missing"] &
    df["previous_missing"]
].copy()

print("\n========== BOTH ROLES MISSING ==========")

print("Total:", len(both_roles_missing))

inspection_columns = [
    "record_id",
    "experience_years",
    "analysis_experience_band",
    "education_level",
    "grad_year",
    "key_skills",
    "n_skills",
    "has_profile_summary"
]

inspection_columns = [
    col for col in inspection_columns
    if col in df.columns
]

print(
    both_roles_missing[inspection_columns]
    .head(30)
    .to_string(index=False)
)

# =====================================================
# 11. SAVE OUTPUT REPORTS
# =====================================================

df.to_parquet(
    OUTPUT_DIR / "candidates_employment_analysis.parquet",
    index=False
)

investigation_candidates.to_csv(
    OUTPUT_DIR / "candidates_employment_investigation.csv",
    index=False
)

both_roles_missing.to_csv(
    OUTPUT_DIR / "candidates_both_roles_missing.csv",
    index=False
)

experience_role_analysis.to_csv(
    OUTPUT_DIR / "experience_role_distribution.csv"
)

# =====================================================
# 12. FINAL SUMMARY
# =====================================================

print("\n========== FINAL SUMMARY ==========")

print("Total candidates:", len(df))

print(
    "Employment information available:",
    (df["employment_classification"] ==
     "EMPLOYMENT_INFORMATION_AVAILABLE").sum()
)

print(
    "Employment information partial:",
    (df["employment_classification"] ==
     "EMPLOYMENT_INFORMATION_PARTIAL").sum()
)

print(
    "Experienced candidates without roles:",
    (df["employment_classification"] ==
     "EXPERIENCED_WITHOUT_ROLE").sum()
)

print(
    "Potential freshers:",
    (df["employment_classification"] ==
     "POTENTIAL_FRESHER").sum()
)

print(
    "Candidates requiring investigation:",
    len(investigation_candidates)
)

print("\nOutput files generated:")
print("- candidates_employment_analysis.parquet")
print("- candidates_employment_investigation.csv")
print("- candidates_both_roles_missing.csv")
print("- experience_role_distribution.csv")

print("\n========== ANALYSIS COMPLETED ==========")

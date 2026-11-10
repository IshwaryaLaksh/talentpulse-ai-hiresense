
import pandas as pd
from pathlib import Path

# =====================================================
# 1. PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_employment_confidence_all.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== CANDIDATE FEATURE READINESS AUDIT ==========")

# =====================================================
# 2. LOAD DATA
# =====================================================

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))
print("Total columns:", len(df.columns))


# =====================================================
# 3. IDENTIFY IMPORTANT FEATURES
# =====================================================

feature_groups = {
    "Candidate Identifier": [
        "record_id"
    ],
    "Employment": [
        "current_role",
        "previous_role",
        "experience_years",
        "role_family",
        "role_family_source"
    ],
    "Skills": [
        "key_skills",
        "n_skills"
    ],
    "Education": [
        "education_level",
        "grad_year"
    ],
    "Profile Information": [
        "has_profile_summary"
    ]
}

available_features = []

for group, columns in feature_groups.items():

    for column in columns:

        if column in df.columns:

            available_features.append({
                "feature_group": group,
                "column": column,
                "available": True,
                "data_type": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_percentage": round(
                    df[column].isna().mean() * 100, 2
                )
            })

        else:

            available_features.append({
                "feature_group": group,
                "column": column,
                "available": False,
                "data_type": "NOT AVAILABLE",
                "missing_count": None,
                "missing_percentage": None
            })

feature_report = pd.DataFrame(available_features)


# =====================================================
# 4. FEATURE AVAILABILITY REPORT
# =====================================================

print("\n========== FEATURE AVAILABILITY ==========")

print(
    feature_report[
        ["feature_group", "column", "available",
         "data_type", "missing_percentage"]
    ].to_string(index=False)
)

feature_report.to_csv(
    OUTPUT_DIR / "candidate_feature_availability.csv",
    index=False
)


# =====================================================
# 5. ROLE FAMILY DISTRIBUTION
# =====================================================

if "role_family" in df.columns:

    print("\n========== ROLE FAMILY DISTRIBUTION ==========")

    role_distribution = (
        df["role_family"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    role_distribution.columns = [
        "role_family", "candidate_count"
    ]

    role_distribution["percentage"] = (
        role_distribution["candidate_count"]
        / len(df) * 100
    ).round(2)

    print(role_distribution.head(20).to_string(index=False))

    role_distribution.to_csv(
        OUTPUT_DIR / "candidate_role_family_distribution.csv",
        index=False
    )


# =====================================================
# 6. EXPERIENCE DISTRIBUTION
# =====================================================

if "experience_years" in df.columns:

    print("\n========== EXPERIENCE DISTRIBUTION ==========")

    experience = pd.to_numeric(
        df["experience_years"],
        errors="coerce"
    )

    print(experience.describe())

    experience_bands = [
        0, 1, 3, 5, 8, 12, float("inf")
    ]

    experience_labels = [
        "0-1 years",
        "1-3 years",
        "3-5 years",
        "5-8 years",
        "8-12 years",
        "12+ years"
    ]

    df["feature_experience_band"] = pd.cut(
        experience,
        bins=experience_bands,
        labels=experience_labels,
        include_lowest=True,
        right=False
    )

    experience_distribution = (
        df["feature_experience_band"]
        .value_counts(sort=False)
        .reset_index()
    )

    experience_distribution.columns = [
        "experience_band", "candidate_count"
    ]

    print(experience_distribution.to_string(index=False))

    experience_distribution.to_csv(
        OUTPUT_DIR / "candidate_experience_distribution.csv",
        index=False
    )


# =====================================================
# 7. SKILLS DATA INSPECTION
# =====================================================

if "key_skills" in df.columns:

    print("\n========== SKILLS DATA INSPECTION ==========")

    skills_available = (
        df["key_skills"].notna()
        & df["key_skills"].astype(str).str.strip().ne("")
    )

    print("Candidates with skills:", int(skills_available.sum()))
    print("Candidates without skills:", int((~skills_available).sum()))

    print("\nSample skill records:")

    print(
        df.loc[skills_available, "key_skills"]
        .head(10)
        .to_string(index=False)
    )


# =====================================================
# 8. EDUCATION DISTRIBUTION
# =====================================================

if "education_level" in df.columns:

    print("\n========== EDUCATION DISTRIBUTION ==========")

    education_distribution = (
        df["education_level"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    education_distribution.columns = [
        "education_level", "candidate_count"
    ]

    print(education_distribution.head(20).to_string(index=False))

    education_distribution.to_csv(
        OUTPUT_DIR / "candidate_education_distribution.csv",
        index=False
    )


# =====================================================
# 9. SAMPLE CANDIDATE RECORDS
# =====================================================

sample_columns = [
    "record_id",
    "experience_years",
    "current_role",
    "previous_role",
    "role_family",
    "key_skills",
    "n_skills",
    "education_level",
    "grad_year"
]

sample_columns = [
    col for col in sample_columns
    if col in df.columns
]

sample_records = df[sample_columns].head(25)

sample_records.to_csv(
    OUTPUT_DIR / "candidate_feature_sample.csv",
    index=False
)


# =====================================================
# 10. SAVE AUDIT SUMMARY
# =====================================================

print("\n========== AUDIT OUTPUTS ==========")

print("- candidate_feature_availability.csv")
print("- candidate_role_family_distribution.csv")
print("- candidate_experience_distribution.csv")
print("- candidate_education_distribution.csv")
print("- candidate_feature_sample.csv")

print("\n========== FEATURE READINESS AUDIT COMPLETED ==========")

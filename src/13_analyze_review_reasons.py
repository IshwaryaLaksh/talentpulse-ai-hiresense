
import pandas as pd
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_extraction_validated_all.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD VALIDATED DATASET
# ==========================================

print("\nLoading validated candidate dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))


# ==========================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "record_id",
    "current_role",
    "previous_role",
    "current_role_validation",
    "previous_role_validation",
    "candidate_validation_status"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================
# 4. FILTER INVESTIGATION CANDIDATES
# ==========================================

# Priority 1:
# Candidates with no usable role

priority_1_candidates = df[
    df["candidate_validation_status"] == "NEEDS_REVIEW"
].copy()


# Priority 2:
# Candidates with one usable role
# and another role requiring review

priority_2_candidates = df[
    df["candidate_validation_status"] == "USABLE_WITH_REVIEW"
].copy()


# Combine both categories

review_candidates = df[
    df["candidate_validation_status"].isin(
        ["NEEDS_REVIEW", "USABLE_WITH_REVIEW"]
    )
].copy()


# Assign investigation priority

review_candidates["review_priority"] = (
    review_candidates["candidate_validation_status"]
    .map({
        "NEEDS_REVIEW": "PRIORITY_1",
        "USABLE_WITH_REVIEW": "PRIORITY_2"
    })
)


print("\n========== INVESTIGATION COUNTS ==========")

print("Priority 1 candidates:",
      len(priority_1_candidates))

print("Priority 2 candidates:",
      len(priority_2_candidates))

print("Total candidates for investigation:",
      len(review_candidates))


# ==========================================
# 5. IDENTIFY REVIEW REASONS
# ==========================================

def identify_review_reason(row):

    statuses = [
        row["current_role_validation"],
        row["previous_role_validation"]
    ]

    reasons = []

    if "MANUAL_REVIEW" in statuses:
        reasons.append("MANUAL_REVIEW")

    if "EXTRACTION_ERROR" in statuses:
        reasons.append("EXTRACTION_ERROR")

    return " | ".join(reasons) if reasons else "OTHER"


review_candidates["review_reason"] = (
    review_candidates.apply(
        identify_review_reason,
        axis=1
    )
)


# ==========================================
# 6. REVIEW REASON DISTRIBUTION
# ==========================================

print("\n========== REVIEW REASON DISTRIBUTION ==========")

print(
    review_candidates["review_reason"]
    .value_counts(dropna=False)
)


# ==========================================
# 7. CURRENT ROLE VALIDATION
# ==========================================

print("\n========== CURRENT ROLE VALIDATION ==========")

print(
    review_candidates["current_role_validation"]
    .value_counts(dropna=False)
)


# ==========================================
# 8. PREVIOUS ROLE VALIDATION
# ==========================================

print("\n========== PREVIOUS ROLE VALIDATION ==========")

print(
    review_candidates["previous_role_validation"]
    .value_counts(dropna=False)
)


# ==========================================
# 9. PRIORITY-WISE REVIEW ANALYSIS
# ==========================================

print("\n========== PRIORITY-WISE ANALYSIS ==========")

print(
    review_candidates.groupby(
        ["review_priority", "review_reason"],
        dropna=False
    ).size()
)


# ==========================================
# 10. VALIDATION STATUS COMBINATIONS
# ==========================================

status_combinations = (
    review_candidates.groupby(
        [
            "current_role_validation",
            "previous_role_validation"
        ],
        dropna=False
    )
    .size()
    .reset_index(name="candidate_count")
    .sort_values(
        "candidate_count",
        ascending=False
    )
)

print("\n========== STATUS COMBINATIONS ==========")

print(status_combinations.to_string(index=False))


# ==========================================
# 11. SELECT AVAILABLE DETAIL COLUMNS
# ==========================================

columns = [
    "record_id",
    "current_role",
    "current_role_extracted_role",
    "current_role_extracted_company",
    "current_role_extraction_status",
    "current_role_validation",
    "previous_role",
    "previous_role_extracted_role",
    "previous_role_extracted_company",
    "previous_role_extraction_status",
    "previous_role_validation",
    "candidate_validation_status",
    "review_priority",
    "review_reason"
]

# Retain columns that actually exist

available_columns = [
    col for col in columns
    if col in review_candidates.columns
]

investigation_details = review_candidates[
    available_columns
].copy()


# ==========================================
# 12. DISPLAY SAMPLE INVESTIGATION RECORDS
# ==========================================

print("\n========== SAMPLE INVESTIGATION RECORDS ==========")

print(
    investigation_details
    .head(30)
    .to_string(index=False)
)


# ==========================================
# 13. SAVE INVESTIGATION DATASETS
# ==========================================

# Complete investigation dataset

investigation_details.to_parquet(
    OUTPUT_DIR / "candidates_investigation_review.parquet",
    index=False
)

# Detailed CSV for manual inspection

investigation_details.to_csv(
    OUTPUT_DIR / "review_candidates_detailed.csv",
    index=False
)

# Priority 1 dataset

priority_1_candidates.to_parquet(
    OUTPUT_DIR / "candidates_priority_1_review.parquet",
    index=False
)

# Priority 2 dataset

priority_2_candidates.to_parquet(
    OUTPUT_DIR / "candidates_priority_2_review.parquet",
    index=False
)

# Status combinations report

status_combinations.to_csv(
    OUTPUT_DIR / "review_status_combinations.csv",
    index=False
)


# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n========== FINAL INVESTIGATION SUMMARY ==========")

print("Original candidates:", len(df))

print("Priority 1 candidates:",
      len(priority_1_candidates))

print("Priority 2 candidates:",
      len(priority_2_candidates))

print("Total investigated:",
      len(review_candidates))

print("\nOutput files generated:")

print("- candidates_investigation_review.parquet")
print("- candidates_priority_1_review.parquet")
print("- candidates_priority_2_review.parquet")
print("- review_candidates_detailed.csv")
print("- review_status_combinations.csv")

print("\n========== INVESTIGATION COMPLETED ==========")

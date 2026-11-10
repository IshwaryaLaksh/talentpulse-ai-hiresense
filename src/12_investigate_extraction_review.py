
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

# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading validated candidate dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))

# ==========================================
# 3. FILTER REVIEW CANDIDATES
# ==========================================

review_candidates = df[
    df["candidate_validation_status"] == "NEEDS_REVIEW"
].copy()

print("\nCandidates requiring investigation:",
      len(review_candidates))

# ==========================================
# 4. IDENTIFY REVIEW REASONS
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

    return " | ".join(reasons)


review_candidates["review_reason"] = (
    review_candidates.apply(
        identify_review_reason,
        axis=1
    )
)

# ==========================================
# 5. DISPLAY INVESTIGATION RESULTS
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
    "review_reason"
]

print("\n========== REVIEW CANDIDATES ==========")

print(
    review_candidates[columns]
    .head(50)
    .to_string(index=False)
)

# ==========================================
# 6. REVIEW REASON SUMMARY
# ==========================================

print("\n========== REVIEW REASON SUMMARY ==========")

print(
    review_candidates["review_reason"]
    .value_counts()
)

# ==========================================
# 7. SAVE INVESTIGATION DATA
# ==========================================

review_candidates[columns].to_parquet(
    OUTPUT_DIR / "candidates_investigation_review.parquet",
    index=False
)

print("\nInvestigation file saved successfully.")

print("\n========== INVESTIGATION COMPLETED ==========")

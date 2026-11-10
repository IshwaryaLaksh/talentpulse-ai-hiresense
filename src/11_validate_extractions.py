
import pandas as pd
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_all_with_extracted_roles.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading candidate dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))

# ==========================================
# 3. ROLE-LEVEL VALIDATION
# ==========================================

def validate_role(row, role_type):

    original = row[role_type]

    extraction_status = row[
        f"{role_type}_extraction_status"
    ]

    extracted_role = row[
        f"{role_type}_extracted_role"
    ]

    extracted_company = row[
        f"{role_type}_extracted_company"
    ]

    original_status = row.get(
        f"{role_type}_status", None
    )

    # Check missing original information
    if pd.isna(original) or not str(original).strip():
        return "MISSING"

    # Successful extraction
    if extraction_status == "EXTRACTED":

        if pd.isna(extracted_role) or not str(extracted_role).strip():
            return "EXTRACTION_ERROR"

        if pd.isna(extracted_company) or not str(extracted_company).strip():
            return "EXTRACTION_ERROR"

        original_text = str(original).lower()

        role_text = str(extracted_role).strip().lower()
        company_text = str(extracted_company).strip().lower()

        if role_text not in original_text:
            return "EXTRACTION_ERROR"

        if company_text not in original_text:
            return "EXTRACTION_ERROR"

        return "VALIDATED"

    # Records requiring human inspection
    if extraction_status == "MANUAL_REVIEW":
        return "MANUAL_REVIEW"

    # Entries that did not require extraction
    if extraction_status == "NOT_APPLICABLE":

        if original_status == "VALID":
            return "ALREADY_CLEAN"

        if original_status == "MISSING":
            return "MISSING"

        return "MANUAL_REVIEW"

    # Unknown extraction status
    return "MANUAL_REVIEW"


# ==========================================
# 4. APPLY ROLE VALIDATION
# ==========================================

df["current_role_validation"] = df.apply(
    lambda row: validate_role(row, "current_role"),
    axis=1
)

df["previous_role_validation"] = df.apply(
    lambda row: validate_role(row, "previous_role"),
    axis=1
)

# ==========================================
# 5. CANDIDATE-LEVEL CLASSIFICATION
# ==========================================

def classify_candidate(row):

    statuses = [
        row["current_role_validation"],
        row["previous_role_validation"]
    ]

    usable = {"VALIDATED", "ALREADY_CLEAN"}

    review = {"MANUAL_REVIEW", "EXTRACTION_ERROR"}

    has_usable = any(s in usable for s in statuses)
    has_review = any(s in review for s in statuses)

    if has_usable and has_review:
        return "USABLE_WITH_REVIEW"

    if has_usable:
        return "USABLE"

    if has_review:
        return "NEEDS_REVIEW"

    return "NO_ROLE_DATA"


df["candidate_validation_status"] = df.apply(
    classify_candidate,
    axis=1
)

# ==========================================
# 6. VALIDATION REPORT
# ==========================================

print("\n========== EXTRACTION VALIDATION REPORT ==========")

print("\nCurrent role:")
print(df["current_role_validation"].value_counts())

print("\nPrevious role:")
print(df["previous_role_validation"].value_counts())

print("\nCandidate-level classification:")
print(df["candidate_validation_status"].value_counts())

# ==========================================
# 7. SEPARATE OUTPUT DATASETS
# ==========================================

usable_candidates = df[
    df["candidate_validation_status"].isin(
        ["USABLE", "USABLE_WITH_REVIEW"]
    )
].copy()

review_candidates = df[
    df["candidate_validation_status"].isin(
        ["NEEDS_REVIEW", "USABLE_WITH_REVIEW"]
    )
].copy()

no_role_candidates = df[
    df["candidate_validation_status"] == "NO_ROLE_DATA"
].copy()

# ==========================================
# 8. SAVE OUTPUTS
# ==========================================

df.to_parquet(
    OUTPUT_DIR / "candidates_extraction_validated_all.parquet",
    index=False
)

usable_candidates.to_parquet(
    OUTPUT_DIR / "candidates_extraction_usable.parquet",
    index=False
)

review_candidates.to_parquet(
    OUTPUT_DIR / "candidates_extraction_review.parquet",
    index=False
)

no_role_candidates.to_parquet(
    OUTPUT_DIR / "candidates_no_role_data.parquet",
    index=False
)

# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("Total candidates:", len(df))

print("Usable candidates:", len(usable_candidates))

print("Candidates requiring review:", len(review_candidates))

print("Candidates with no role data:", len(no_role_candidates))

print("\n========== VALIDATION COMPLETED ==========")

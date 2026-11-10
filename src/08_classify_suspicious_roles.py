
import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "candidates_all_with_flags.parquet"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# ==========================================
# 2. COMPANY DETECTION PATTERNS
# ==========================================

company_patterns = [
    r"\bprivate\s+limited\b",
    r"\bpvt\.?\s*ltd\.?\b",
    r"\blimited\b",
    r"\bltd\.?\b",
    r"\bllp\b",
    r"\bincorporated\b",
    r"\binc\.?\b",
    r"\bcorporation\b",
    r"\bcorp\.?\b",
    r"\b[a-z0-9][a-z0-9-]*\.(?:ai|com|in|io|co|org|net)\b"
]

company_regex = re.compile(
    "|".join(company_patterns),
    flags=re.IGNORECASE
)

# ==========================================
# 3. LOAD DATA
# ==========================================

print("Loading candidate dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))

# ==========================================
# 4. CLASSIFY ROLE ENTRIES
# ==========================================

job_title_keywords = [
    "engineer",
    "developer",
    "tester",
    "analyst",
    "consultant",
    "associate",
    "manager",
    "intern",
    "trainee",
    "architect",
    "administrator",
    "executive",
    "specialist",
    "lead",
    "technician",
    "designer",
    "programmer",
    "scientist",
    "director",
    "officer",
    "sdet",
    "qa",
    "team lead",
    "product manager",
    "software engineer",
    "software developer",
    "quality analyst",
    "quality assurance engineer",
    "automation tester",
    "manual tester"
]

job_title_regex = re.compile(
    r"\b(" + "|".join(
        re.escape(keyword) for keyword in job_title_keywords
    ) + r")\b",
    flags=re.IGNORECASE
)


def classify_role(value):

    if pd.isna(value) or not str(value).strip():
        return "MISSING"

    text = str(value).strip()

    company_check_text = re.sub(
        r"\bASP\.NET\b",
    "ASP_NET",
    text,
    flags=re.IGNORECASE
)
    has_company_indicator = bool(
    company_regex.search(company_check_text)
)
    
    has_job_title = bool(
        job_title_regex.search(text)
    )

    if has_company_indicator and has_job_title:
        return "MIXED"

    if has_company_indicator:
        return "COMPANY_ONLY"

    return "VALID"



# ==========================================
# 5. APPLY CLASSIFICATION
# ==========================================

df["current_role_status"] = (
    df["current_role"].apply(classify_role)
)

df["previous_role_status"] = (
    df["previous_role"].apply(classify_role)
)

# ==========================================
# 6. CANDIDATE QUALITY STATUS
# ==========================================

# ==========================================
# 6. CANDIDATE QUALITY STATUS
# ==========================================

def candidate_status(row):

    statuses = [
        row["current_role_status"],
        row["previous_role_status"]
    ]

    if "MIXED" in statuses or "COMPANY_ONLY" in statuses:
        return "NEEDS_REVIEW"

    if statuses.count("MISSING") == 2:
        return "NO_ROLE_DATA"

    return "VALID"


df["candidate_quality_status"] = df.apply(
    candidate_status,
    axis=1
)

# ==========================================
# 7. FIELD-LEVEL FLAGS
# ==========================================

df["current_role_flag"] = (
    df["current_role_status"].isin(
        ["MIXED", "COMPANY_ONLY"]
    )
)

df["previous_role_flag"] = (
    df["previous_role_status"].isin(
        ["MIXED", "COMPANY_ONLY"]
    )
)

df["role_data_quality_flag"] = (
    df["current_role_flag"] |
    df["previous_role_flag"]
)

# ==========================================
# 8. SEPARATE OUTPUT DATASETS
# ==========================================

valid_candidates = df[
    df["candidate_quality_status"] == "VALID"
].copy()

review_candidates = df[
    df["candidate_quality_status"] == "NEEDS_REVIEW"
].copy()

# Keep candidates with at least one usable role
# ==========================================
# CANDIDATES WITH AT LEAST ONE USABLE ROLE
# ==========================================

usable_candidates = df[
    (df["current_role_status"] == "VALID") |
    (df["previous_role_status"] == "VALID")
].copy()
# ==========================================
# 9. SAVE OUTPUTS
# ==========================================

df.to_parquet(
    OUTPUT_DIR / "candidates_classified_all.parquet",
    index=False
)

valid_candidates.to_parquet(
    OUTPUT_DIR / "candidates_valid.parquet",
    index=False
)

review_candidates.to_parquet(
    OUTPUT_DIR / "candidates_review.parquet",
    index=False
)

usable_candidates.to_parquet(
    OUTPUT_DIR / "candidates_usable.parquet",
    index=False
)

# ==========================================
# 10. REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print("Total candidates:", len(df))

print("\nCurrent role status:")
print(df["current_role_status"].value_counts())

print("\nPrevious role status:")
print(df["previous_role_status"].value_counts())

print("\nCandidate quality:")
print(df["candidate_quality_status"].value_counts())

print("\nCandidates with at least one usable role:",
      len(usable_candidates))

print("\n========== PROCESS COMPLETED ==========")

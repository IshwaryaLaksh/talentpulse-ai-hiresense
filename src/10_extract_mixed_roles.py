
import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "candidates_classified_all.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# ==========================================
# 2. COMPANY DETECTION
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

# Prevent ASP.NET from being mistaken for a company domain
def has_company_indicator(text):

    text = re.sub(
        r"\bASP\.NET\b",
        "ASP_NET",
        str(text),
        flags=re.IGNORECASE
    )

    return bool(company_regex.search(text))


# ==========================================
# 3. JOB TITLE DETECTION
# ==========================================

job_title_keywords = [
    "engineer", "developer", "tester", "testing",
    "analyst", "consultant", "associate",
    "manager", "intern", "trainee", "architect",
    "administrator", "executive", "specialist",
    "lead", "software", "quality", "assurance",
    "automation", "manual", "support",
    "technician", "designer", "programmer",
    "scientist", "director", "officer", "sdet", "qa"
]

job_title_regex = re.compile(
    r"\b(" + "|".join(
        re.escape(word) for word in job_title_keywords
    ) + r")\b",
    flags=re.IGNORECASE
)

# ==========================================
# 4. SPLIT PATTERNS
# ==========================================

split_regex = re.compile(
    r"\s+(?:at|with|for|in)\s+"
    r"|\s+-\s+"
    r"|,\s*"
    r"|\s+\|\s+"
    r"|\s*\(\s*"
    r"|\s*\)\s*",
    flags=re.IGNORECASE
)


# ==========================================
# 5. EXTRACT ROLE AND COMPANY
# ==========================================

def extract_mixed_entry(value):

    if pd.isna(value) or not str(value).strip():
        return None, None, "EMPTY"

    text = str(value).strip()

    # Find possible separation points
    separators = list(split_regex.finditer(text))

    best_match = None
    best_score = -1

    for separator in separators:

        left = text[:separator.start()].strip(" ,-|()")
        right = text[separator.end():].strip(" ,-|()")

        # Case 1: Job title first, company second
        possibilities = [
            (left, right),
            (right, left)
        ]

        for role_part, company_part in possibilities:

            if not role_part or not company_part:
                continue

            has_role = bool(
                job_title_regex.search(role_part)
            )

            has_company = has_company_indicator(
                company_part
            )

            if has_role and has_company:

                # Prefer a role with more recognizable title terms
                score = len(
                    job_title_regex.findall(role_part)
                )

                if score > best_score:
                    best_score = score
                    best_match = (
                        role_part.strip(),
                        company_part.strip()
                    )

    if best_match is not None:

        role, company = best_match

        return role, company, "EXTRACTED"

    # Do not guess when separation is uncertain
    return None, None, "MANUAL_REVIEW"


# ==========================================
# 6. LOAD DATA
# ==========================================

print("Loading classified candidate dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))


# ==========================================
# 7. PROCESS CURRENT AND PREVIOUS ROLES
# ==========================================

role_columns = [
    ("current_role", "current_role_status"),
    ("previous_role", "previous_role_status")
]


# ==========================================
# 7. PROCESS CURRENT AND PREVIOUS ROLES
# ==========================================

for role_column, status_column in [
    ("current_role", "current_role_status"),
    ("previous_role", "previous_role_status")
]:

    # Keep the complete original column name
    prefix = role_column

    df[f"{prefix}_extracted_role"] = None
    df[f"{prefix}_extracted_company"] = None
    df[f"{prefix}_extraction_status"] = "NOT_APPLICABLE"

    mixed_indices = df.index[
        df[status_column] == "MIXED"
    ]

    for idx in mixed_indices:

        role, company, status = extract_mixed_entry(
            df.at[idx, role_column]
        )

        df.at[idx, f"{prefix}_extracted_role"] = role

        df.at[idx, f"{prefix}_extracted_company"] = company

        df.at[idx, f"{prefix}_extraction_status"] = status



# ==========================================
# 8. CREATE EXTRACTION DATASETS
# ==========================================

mixed_candidates = df[
    (df["current_role_status"] == "MIXED") |
    (df["previous_role_status"] == "MIXED")
].copy()

extracted_candidates = df[
    (
        df["current_role_extraction_status"] == "EXTRACTED"
    ) |
    (
        df["previous_role_extraction_status"] == "EXTRACTED"
    )
].copy()

manual_review = df[
    (
        df["current_role_extraction_status"] == "MANUAL_REVIEW"
    ) |
    (
        df["previous_role_extraction_status"] == "MANUAL_REVIEW"
    )
].copy()


# ==========================================
# 9. SAVE OUTPUTS
# ==========================================

df.to_parquet(
    OUTPUT_DIR / "candidates_all_with_extracted_roles.parquet",
    index=False
)

mixed_candidates.to_parquet(
    OUTPUT_DIR / "candidates_mixed_role_extracted.parquet",
    index=False
)

extracted_candidates.to_parquet(
    OUTPUT_DIR / "candidates_extracted_successfully.parquet",
    index=False
)

manual_review.to_parquet(
    OUTPUT_DIR / "candidates_manual_review.parquet",
    index=False
)


# ==========================================
# 10. EXTRACTION REPORT
# ==========================================

print("\n========== EXTRACTION REPORT ==========")

print("Total candidates:", len(df))

print("Mixed candidates:", len(mixed_candidates))

for role_column in [
    "current_role_extraction_status",
    "previous_role_extraction_status"
]:

    print(f"\n{role_column}:")
    print(df[role_column].value_counts())

print("\nSuccessfully extracted candidates:",
      len(extracted_candidates))

print("Candidates requiring manual review:",
      len(manual_review))

print("\n========== SAMPLE EXTRACTIONS ==========")


columns = [
    "record_id",

    "current_role",
    "current_role_extracted_role",
    "current_role_extracted_company",
    "current_role_extraction_status",

    "previous_role",
    "previous_role_extracted_role",
    "previous_role_extracted_company",
    "previous_role_extraction_status"
]


print(
    mixed_candidates[columns]
    .head(20)
    .to_string(index=False)
)

print("\n========== PROCESS COMPLETED ==========")

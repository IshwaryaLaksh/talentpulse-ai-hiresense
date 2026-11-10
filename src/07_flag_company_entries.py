
import pandas as pd
import re
from pathlib import Path

# -----------------------------------
# 1. FILE LOCATIONS
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "candidates.parquet"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. COMPANY NAME DETECTION PATTERNS
# -----------------------------------

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

    # Website/domain-style company names
    r"\b[a-z0-9][a-z0-9-]*\.(?:ai|com|in|io|co|org|net)\b"
]

company_regex = re.compile(
    "|".join(company_patterns),
    flags=re.IGNORECASE
)

# -----------------------------------
# 3. LOAD DATA
# -----------------------------------

print("Loading candidate dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))

# -----------------------------------
# 4. IDENTIFY SUSPICIOUS ROLE ENTRIES
# -----------------------------------

role_columns = ["current_role", "previous_role"]

for column in role_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

def detect_company_entry(value):

    if pd.isna(value):
        return False

    return bool(company_regex.search(str(value)))


df["current_role_company_flag"] = (
    df["current_role"].apply(detect_company_entry)
)

df["previous_role_company_flag"] = (
    df["previous_role"].apply(detect_company_entry)
)

# -----------------------------------
# 5. CREATE OVERALL QUALITY FLAG
# -----------------------------------

df["role_data_quality_flag"] = (
    df["current_role_company_flag"] |
    df["previous_role_company_flag"]
)

def get_flag_reason(row):

    reasons = []

    if row["current_role_company_flag"]:
        reasons.append("Company-like entry in current_role")

    if row["previous_role_company_flag"]:
        reasons.append("Company-like entry in previous_role")

    return "; ".join(reasons) if reasons else "No suspicious entry"

df["role_flag_reason"] = df.apply(
    get_flag_reason,
    axis=1
)

# -----------------------------------
# 6. SEPARATE CANDIDATES
# -----------------------------------

flagged_candidates = df[
    df["role_data_quality_flag"]
].copy()

clean_candidates = df[
    ~df["role_data_quality_flag"]
].copy()

# -----------------------------------
# 7. SAVE OUTPUT FILES
# -----------------------------------

df.to_parquet(
    OUTPUT_DIR / "candidates_all_with_flags.parquet",
    index=False
)

clean_candidates.to_parquet(
    OUTPUT_DIR / "candidates_clean.parquet",
    index=False
)

flagged_candidates.to_parquet(
    OUTPUT_DIR / "candidates_flagged.parquet",
    index=False
)

# -----------------------------------
# 8. PRINT QUALITY REPORT
# -----------------------------------

print("\n========== ROLE DATA QUALITY REPORT ==========")

print("Total candidates:", len(df))

print("Clean candidates:", len(clean_candidates))

print("Flagged candidates:", len(flagged_candidates))

print(
    "Current role flags:",
    df["current_role_company_flag"].sum()
)

print(
    "Previous role flags:",
    df["previous_role_company_flag"].sum()
)

print(
    "Flagged percentage:",
    round(len(flagged_candidates) / len(df) * 100, 2),
    "%"
)

print("\nSample flagged records:")

print(
    flagged_candidates[
        [
            "record_id",
            "current_role",
            "previous_role",
            "role_flag_reason"
        ]
    ].head(15).to_string(index=False)
)

print("\n========== PROCESS COMPLETED ==========")

print("Clean file:", OUTPUT_DIR / "candidates_clean.parquet")
print("Flagged file:", OUTPUT_DIR / "candidates_flagged.parquet")


import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_employment_analysis.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading employment analysis dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run Step 15 before running Step 16."
    )

df = pd.read_parquet(INPUT_FILE)

print("Total candidates:", len(df))


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def is_missing(value):

    if pd.isna(value):
        return True

    return str(value).strip().lower() in {
        "", "none", "nan", "null", "n/a",
        "not available", "not provided"
    }


def get_text(value):

    if is_missing(value):
        return ""

    return str(value).strip()


# ==========================================
# 3. IDENTIFY CANDIDATES WITH NO ROLES
# ==========================================

required_columns = [
    "current_role",
    "previous_role",
    "experience_years"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing required columns: {missing_columns}"
    )

current_missing = df["current_role"].apply(is_missing)
previous_missing = df["previous_role"].apply(is_missing)

experience = pd.to_numeric(
    df["experience_years"],
    errors="coerce"
)

investigation = df[
    current_missing &
    previous_missing &
    (experience > 0)
].copy()

print("\nCandidates with experience but no role:",
      len(investigation))


# ==========================================
# 4. ROLE KEYWORD DETECTION
# ==========================================

role_keywords = [
    "engineer",
    "developer",
    "analyst",
    "consultant",
    "tester",
    "testing",
    "manager",
    "architect",
    "administrator",
    "specialist",
    "executive",
    "associate",
    "intern",
    "trainee",
    "designer",
    "programmer",
    "scientist",
    "director",
    "officer",
    "technician",
    "team lead",
    "project lead",
    "software",
    "quality assurance",
    "automation",
    "support",
    "recruiter",
    "accountant",
    "business analyst",
    "data analyst",
    "data scientist",
    "sales representative",
    "product manager",
    "operations manager",
    "system administrator"
]

role_regex = re.compile(
    r"\b(" +
    "|".join(
        re.escape(word)
        for word in sorted(
            role_keywords,
            key=len,
            reverse=True
        )
    ) +
    r")\b",
    flags=re.IGNORECASE
)


# ==========================================
# 5. COMPANY INDICATOR DETECTION
# ==========================================

company_regex = re.compile(
    r"\b("
    r"private limited|pvt\.?\s*ltd\.?|"
    r"limited|ltd\.?|llp|incorporated|"
    r"inc\.?|corporation|corp\.?"
    r")\b",
    flags=re.IGNORECASE
)


# ==========================================
# 6. IDENTIFY AVAILABLE TEXT FIELDS
# ==========================================

possible_text_columns = [
    "candidate_text",
    "profile_summary",
    "summary",
    "about",
    "headline",
    "description",
    "resume_text",
    "experience_text"
]

text_columns = [
    col for col in possible_text_columns
    if col in investigation.columns
]

print("\nAvailable narrative fields:", text_columns)

if not text_columns:
    print(
        "WARNING: No narrative text fields found. "
        "Only structured fields will be investigated."
    )


# ==========================================
# 7. ANALYZE EACH CANDIDATE
# ==========================================

def investigate_candidate(row):

    combined_text = " ".join(
        get_text(row[col])
        for col in text_columns
    )

    # Check existing role family information

    role_family = get_text(
        row.get("role_family", "")
    )

    valid_role_family = (
        role_family.lower()
        not in {
            "", "none", "nan", "unknown",
            "other / unparsed", "other",
            "not available"
        }
    )

    # Search narrative text for role keywords

    detected_roles = list(
        dict.fromkeys(
            match.group(0).lower()
            for match in role_regex.finditer(
                combined_text
            )
        )
    )

    # Check company indicators

    company_matches = list(
        dict.fromkeys(
            match.group(0)
            for match in company_regex.finditer(
                combined_text
            )
        )
    )

    # Check profile summary availability

    has_profile_summary = (
        bool(get_text(row.get("profile_summary", "")))
        or bool(get_text(row.get("candidate_text", "")))
        or bool(get_text(row.get("summary", "")))
    )

    # Evidence-based classification

    if valid_role_family or detected_roles:

        evidence_status = "POTENTIAL_ROLE_EVIDENCE"

    elif has_profile_summary:

        evidence_status = "TEXT_REVIEW_REQUIRED"

    else:

        evidence_status = "NO_ROLE_EVIDENCE_FOUND"

    return pd.Series({
        "detected_role_keywords": " | ".join(
            detected_roles
        ),

        "detected_company_indicators": " | ".join(
            company_matches
        ),

        "existing_role_family": role_family,

        "has_profile_text": has_profile_summary,

        "potential_company_evidence": bool(
            company_matches
        ),

        "recovery_evidence_status": evidence_status,

        "candidate_text_sample": combined_text[:1000]
    })


print("\nRunning employment evidence investigation...")

investigation_results = investigation.apply(
    investigate_candidate,
    axis=1
)

investigation = pd.concat(
    [investigation, investigation_results],
    axis=1
)


# ==========================================
# 8. INVESTIGATION SUMMARY
# ==========================================

print("\n========== EMPLOYMENT RECOVERY REPORT ==========")

print("\nRecovery evidence distribution:")

print(
    investigation[
        "recovery_evidence_status"
    ].value_counts(dropna=False)
)

print("\nPotential role evidence:")

print(
    investigation[
        "detected_role_keywords"
    ].replace("", "NO KEYWORDS")
    .value_counts()
    .head(20)
)

print("\nExisting role family distribution:")

if "role_family" in investigation.columns:

    print(
        investigation["role_family"]
        .value_counts(dropna=False)
        .head(20)
    )

print("\nCompany indicator evidence:")

print(
    investigation[
        "potential_company_evidence"
    ].value_counts()
)


# ==========================================
# 9. CREATE SEPARATE INVESTIGATION DATASETS
# ==========================================

potential_recovery = investigation[
    investigation["recovery_evidence_status"]
    == "POTENTIAL_ROLE_EVIDENCE"
].copy()

text_review = investigation[
    investigation["recovery_evidence_status"]
    == "TEXT_REVIEW_REQUIRED"
].copy()

no_evidence = investigation[
    investigation["recovery_evidence_status"]
    == "NO_ROLE_EVIDENCE_FOUND"
].copy()


# ==========================================
# 10. SAVE OUTPUT FILES
# ==========================================

investigation.to_parquet(
    OUTPUT_DIR /
    "candidates_employment_recovery_analysis.parquet",
    index=False
)

potential_recovery.to_csv(
    OUTPUT_DIR /
    "candidates_potential_role_recovery.csv",
    index=False
)

text_review.to_csv(
    OUTPUT_DIR /
    "candidates_text_review.csv",
    index=False
)

no_evidence.to_csv(
    OUTPUT_DIR /
    "candidates_no_employment_evidence.csv",
    index=False
)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========== FINAL INVESTIGATION SUMMARY ==========")

print("Original candidates:", len(df))

print("Candidates with experience but no roles:",
      len(investigation))

print("Potential role evidence:",
      len(potential_recovery))

print("Candidates requiring text review:",
      len(text_review))

print("Candidates with no role evidence:",
      len(no_evidence))

print("\nOUTPUT FILES GENERATED:")

print("- candidates_employment_recovery_analysis.parquet")
print("- candidates_potential_role_recovery.csv")
print("- candidates_text_review.csv")
print("- candidates_no_employment_evidence.csv")

print("\n========== INVESTIGATION COMPLETED ==========")

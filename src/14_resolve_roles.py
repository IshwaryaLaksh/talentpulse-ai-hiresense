
import pandas as pd
import re
from pathlib import Path

# =====================================================
# 1. FILE PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_investigation_review.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# =====================================================
# 2. CONFIGURATION
# =====================================================

USABLE_STATUSES = {"ALREADY_CLEAN", "VALIDATED"}

COMPANY_PATTERNS = [
    r"\bprivate\s+limited\b",
    r"\bpvt\.?\s*ltd\.?\b",
    r"\blimited\b",
    r"\bltd\.?\b",
    r"\bllp\b",
    r"\bincorporated\b",
    r"\binc\.?\b",
    r"\bcorporation\b",
    r"\bcorp\.?\b",
    r"\btechnologies\b",
    r"\btechnology\b",
    r"\bsolutions\b",
    r"\bsystems\b",
    r"\bsoftware\b",
    r"\bconsulting\b",
    r"\bservices\b",
    r"\bventures\b",
    r"\bindustries\b",
    r"\bgroup\b",
    r"\b[a-z0-9-]+\.(?:ai|com|in|io|co|org|net)\b"
]

COMPANY_REGEX = re.compile(
    "|".join(COMPANY_PATTERNS),
    re.IGNORECASE
)

JOB_TITLE_KEYWORDS = [
    "engineer", "developer", "tester", "testing",
    "analyst", "consultant", "associate",
    "manager", "intern", "trainee", "architect",
    "administrator", "executive", "specialist",
    "lead", "software", "quality", "assurance",
    "automation", "manual", "support",
    "technician", "designer", "programmer",
    "scientist", "director", "officer",
    "sdet", "qa", "accountant", "designer",
    "coordinator", "administrator", "researcher",
    "technician", "devops", "network", "security"
]

JOB_REGEX = re.compile(
    r"\b(" + "|".join(
        re.escape(word) for word in JOB_TITLE_KEYWORDS
    ) + r")\b",
    re.IGNORECASE
)

SPLIT_REGEX = re.compile(
    r"\s+(?:at|with|for)\s+"
    r"|\s+-\s+"
    r"|\s*\|\s*"
    r"|,\s*",
    re.IGNORECASE
)

# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def is_present(value):

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    return bool(str(value).strip())


def clean_text(value):

    if not is_present(value):
        return None

    text = str(value).strip()

    text = re.sub(r"\s+", " ", text)

    return text.strip(" ,-|()")


def count_job_keywords(text):

    if not is_present(text):
        return 0

    return len(JOB_REGEX.findall(str(text)))


def has_company_indicator(text):

    if not is_present(text):
        return False

    return bool(COMPANY_REGEX.search(str(text)))


# =====================================================
# 4. ROLE AND COMPANY RESOLUTION ENGINE
# =====================================================

def resolve_mixed_role(value):

    text = clean_text(value)

    if not text:
        return {
            "role": None,
            "company": None,
            "status": "MISSING_SOURCE",
            "method": "NONE",
            "confidence": 0.0
        }

    separators = list(SPLIT_REGEX.finditer(text))

    best_match = None
    best_score = -1

    for separator in separators:

        left = clean_text(text[:separator.start()])
        right = clean_text(text[separator.end():])

        if not left or not right:
            continue

        possibilities = [
            (left, right),
            (right, left)
        ]

        for role_part, company_part in possibilities:

            role_score = count_job_keywords(role_part)

            company_found = has_company_indicator(
                company_part
            )

            if role_score > 0 and company_found:

                score = (
                    role_score * 2
                    + len(COMPANY_REGEX.findall(company_part))
                )

                if score > best_score:

                    best_score = score

                    best_match = (
                        role_part,
                        company_part
                    )

    # Strong role and company match
    if best_match:

        role, company = best_match

        return {
            "role": role,
            "company": company,
            "status": "AUTO_RESOLVED",
            "method": "RULE_BASED_SPLIT",
            "confidence": 0.95
        }

    # Role exists, but company could not be identified
    if count_job_keywords(text) > 0:

        return {
            "role": text,
            "company": None,
            "status": "ROLE_ONLY",
            "method": "JOB_TITLE_KEYWORD_MATCH",
            "confidence": 0.70
        }

    # Company exists, but role could not be identified
    if has_company_indicator(text):

        return {
            "role": None,
            "company": text,
            "status": "COMPANY_ONLY",
            "method": "COMPANY_INDICATOR_MATCH",
            "confidence": 0.70
        }

    # Do not make unsupported assumptions
    return {
        "role": None,
        "company": None,
        "status": "MANUAL_REVIEW",
        "method": "NO_CONFIDENT_PATTERN",
        "confidence": 0.0
    }


# =====================================================
# 5. PROCESS INDIVIDUAL ROLE
# =====================================================

def process_role(row, role_type):

    original = clean_text(row.get(role_type))

    validation = row.get(
        f"{role_type}_validation"
    )

    extraction_status = row.get(
        f"{role_type}_extraction_status"
    )

    extracted_role = clean_text(
        row.get(f"{role_type}_extracted_role")
    )

    extracted_company = clean_text(
        row.get(f"{role_type}_extracted_company")
    )

    # Preserve already-clean role information
    if validation == "ALREADY_CLEAN":

        return {
            "role": original,
            "company": extracted_company,
            "status": "ALREADY_CLEAN",
            "method": "ORIGINAL_VALIDATED_DATA",
            "confidence": 1.0
        }

    # Preserve previously validated extraction
    if (
        validation == "VALIDATED"
        and extracted_role
        and extracted_company
    ):

        return {
            "role": extracted_role,
            "company": extracted_company,
            "status": "VALIDATED_EXTRACTION",
            "method": "EXISTING_VALIDATED_EXTRACTION",
            "confidence": 1.0
        }

    # Missing original information
    if not original:

        return {
            "role": None,
            "company": None,
            "status": "MISSING_SOURCE",
            "method": "NO_ORIGINAL_ROLE_TEXT",
            "confidence": 0.0
        }

    # Try extracting from original role text
    result = resolve_mixed_role(original)

    # Existing extracted values are retained as suggestions
    if (
        extraction_status == "EXTRACTED"
        and extracted_role
        and extracted_company
        and result["status"] != "AUTO_RESOLVED"
    ):

        result["suggested_role"] = extracted_role
        result["suggested_company"] = extracted_company

    return result


# =====================================================
# 6. LOAD INVESTIGATION DATA
# =====================================================

print("\nLoading investigation dataset...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_parquet(INPUT_FILE)

print("Total investigation candidates:", len(df))

# Validate required columns
required_columns = [
    "record_id",
    "current_role",
    "previous_role",
    "current_role_validation",
    "previous_role_validation"
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
# 7. PRIORITY CLASSIFICATION
# =====================================================

def determine_priority(row):

    statuses = [
        row["current_role_validation"],
        row["previous_role_validation"]
    ]

    has_usable = any(
        status in USABLE_STATUSES
        for status in statuses
    )

    if has_usable:
        return "PRIORITY_2"

    return "PRIORITY_1"


df["resolution_priority"] = df.apply(
    determine_priority,
    axis=1
)

# =====================================================
# 8. APPLY ROLE RESOLUTION
# =====================================================

print("\nRunning role resolution engine...")

for role_type in ["current_role", "previous_role"]:

    results = df.apply(
        lambda row: process_role(row, role_type),
        axis=1
    )

    df[f"{role_type}_resolved_role"] = results.apply(
        lambda x: x.get("role")
    )

    df[f"{role_type}_resolved_company"] = results.apply(
        lambda x: x.get("company")
    )

    df[f"{role_type}_resolution_status"] = results.apply(
        lambda x: x.get("status")
    )

    df[f"{role_type}_resolution_method"] = results.apply(
        lambda x: x.get("method")
    )

    df[f"{role_type}_resolution_confidence"] = results.apply(
        lambda x: x.get("confidence")
    )

    df[f"{role_type}_suggested_role"] = results.apply(
        lambda x: x.get("suggested_role")
    )

    df[f"{role_type}_suggested_company"] = results.apply(
        lambda x: x.get("suggested_company")
    )

# =====================================================
# 9. CANDIDATE-LEVEL RESOLUTION STATUS
# =====================================================

def classify_resolution(row):

    statuses = [
        row["current_role_resolution_status"],
        row["previous_role_resolution_status"]
    ]

    if "AUTO_RESOLVED" in statuses:
        return "HAS_AUTO_RESOLUTION"

    if "MANUAL_REVIEW" in statuses:
        return "NEEDS_MANUAL_REVIEW"

    if "ROLE_ONLY" in statuses or "COMPANY_ONLY" in statuses:
        return "PARTIAL_RESOLUTION"

    if "MISSING_SOURCE" in statuses:
        return "MISSING_SOURCE_DATA"

    return "RESOLVED_OR_VALIDATED"


df["candidate_resolution_status"] = df.apply(
    classify_resolution,
    axis=1
)

# =====================================================
# 10. GENERATE SEPARATE DATASETS
# =====================================================

priority_1 = df[
    df["resolution_priority"] == "PRIORITY_1"
].copy()

priority_2 = df[
    df["resolution_priority"] == "PRIORITY_2"
].copy()

auto_resolved = df[
    (
        (df["current_role_resolution_status"] == "AUTO_RESOLVED")
        |
        (df["previous_role_resolution_status"] == "AUTO_RESOLVED")
    )
].copy()

manual_review = df[
    df["candidate_resolution_status"] == "NEEDS_MANUAL_REVIEW"
].copy()

partial_resolution = df[
    df["candidate_resolution_status"] == "PARTIAL_RESOLUTION"
].copy()

# =====================================================
# 11. SAVE OUTPUT DATASETS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_parquet(
    OUTPUT_DIR / "candidates_role_resolution_all.parquet",
    index=False
)

priority_1.to_parquet(
    OUTPUT_DIR / "candidates_role_resolution_priority1.parquet",
    index=False
)

priority_2.to_parquet(
    OUTPUT_DIR / "candidates_role_resolution_priority2.parquet",
    index=False
)

auto_resolved.to_parquet(
    OUTPUT_DIR / "candidates_auto_resolved.parquet",
    index=False
)

manual_review.to_parquet(
    OUTPUT_DIR / "candidates_final_manual_review.parquet",
    index=False
)

partial_resolution.to_parquet(
    OUTPUT_DIR / "candidates_partial_resolution.parquet",
    index=False
)

# CSV for easy human inspection
df.to_csv(
    OUTPUT_DIR / "candidates_role_resolution_report.csv",
    index=False
)

# =====================================================
# 12. RESOLUTION REPORT
# =====================================================

print("\n========== ROLE RESOLUTION REPORT ==========")

print("\nTotal candidates:", len(df))

print("\nPriority distribution:")

print(df["resolution_priority"].value_counts())

print("\nCurrent role resolution:")

print(
    df["current_role_resolution_status"].value_counts(
        dropna=False
    )
)

print("\nPrevious role resolution:")

print(
    df["previous_role_resolution_status"].value_counts(
        dropna=False
    )
)

print("\nCandidate-level resolution:")

print(
    df["candidate_resolution_status"].value_counts(
        dropna=False
    )
)

# =====================================================
# 13. FINAL SUMMARY
# =====================================================

print("\n========== FINAL SUMMARY ==========")

print("Total candidates investigated:", len(df))

print("Priority 1 candidates:", len(priority_1))

print("Priority 2 candidates:", len(priority_2))

print("Candidates with auto-resolved roles:", len(auto_resolved))

print("Candidates requiring manual review:", len(manual_review))

print("Candidates with partial resolution:", len(partial_resolution))

print("\nOutput files generated:")

print("- candidates_role_resolution_all.parquet")
print("- candidates_role_resolution_priority1.parquet")
print("- candidates_role_resolution_priority2.parquet")
print("- candidates_auto_resolved.parquet")
print("- candidates_final_manual_review.parquet")
print("- candidates_partial_resolution.parquet")
print("- candidates_role_resolution_report.csv")

print("\n========== RESOLUTION COMPLETED ==========")

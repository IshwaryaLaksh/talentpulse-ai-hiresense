
import os
import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_FOLDER = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    PROJECT_FOLDER,
    "data",
    "raw",
    "candidate_dataset.xlsx"
)

OUTPUT_FILE = os.path.join(
    PROJECT_FOLDER,
    "data",
    "processed",
    "candidates.parquet"
)

BATCH_SIZE = 5000

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

print("TalentPulse AI - Candidate Preparation")
print("---------------------------------------")

# ==========================================
# 2. OPEN EXCEL IN READ-ONLY MODE
# ==========================================

workbook = openpyxl.load_workbook(
    INPUT_FILE,
    read_only=True,
    data_only=True
)

# Automatically identify candidate worksheet

sheet = None

for ws in workbook.worksheets:

    first_row = next(
        ws.iter_rows(
            min_row=1,
            max_row=1,
            values_only=True
        )
    )

    normalized = [
        str(x).strip().lower() if x is not None else ""
        for x in first_row
    ]

    if "record_id" in normalized and "key_skills" in normalized:
        sheet = ws
        break

if sheet is None:
    workbook.close()
    raise ValueError("Candidate worksheet not found.")

print("Worksheet identified:", sheet.title)

# ==========================================
# 3. MAP COLUMN NAMES
# ==========================================

headers = next(
    sheet.iter_rows(
        min_row=1,
        max_row=1,
        values_only=True
    )
)

positions = {
    str(name).strip().lower(): index
    for index, name in enumerate(headers)
    if name is not None
}

required_columns = [
    "record_id",
    "experience_years",
    "current_role",
    "previous_role",
    "role_family",
    "education_level",
    "key_skills",
    "n_skills",
    "near_duplicate_flag"
]

missing_columns = [
    col for col in required_columns
    if col not in positions
]

if missing_columns:
    workbook.close()
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ==========================================
# 4. DEFINE PARQUET STRUCTURE
# ==========================================

schema = pa.schema([
    ("record_id", pa.string()),
    ("experience_years", pa.float64()),
    ("current_role", pa.string()),
    ("previous_role", pa.string()),
    ("role_family", pa.string()),
    ("education_level", pa.string()),
    ("key_skills", pa.string()),
    ("n_skills", pa.int64()),
    ("near_duplicate_flag", pa.bool_()),
    ("candidate_text", pa.string())
])

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def clean_text(value):

    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def parse_experience(value):

    try:
        if value is None or str(value).strip() == "":
            return None

        return float(value)

    except (ValueError, TypeError):
        return None


def parse_integer(value):

    try:
        if value is None or str(value).strip() == "":
            return None

        return int(float(value))

    except (ValueError, TypeError):
        return None


def parse_boolean(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in (
        "true", "1", "yes"
    )


# ==========================================
# 6. PROCESS CANDIDATES IN BATCHES
# ==========================================

writer = None

batch = []

processed = 0

try:

    for row in sheet.iter_rows(
        min_row=2,
        values_only=True
    ):

        record_id = clean_text(
            row[positions["record_id"]]
        )

        current_role = clean_text(
            row[positions["current_role"]]
        )

        previous_role = clean_text(
            row[positions["previous_role"]]
        )

        role_family = clean_text(
            row[positions["role_family"]]
        )

        education = clean_text(
            row[positions["education_level"]]
        )

        skills = clean_text(
            row[positions["key_skills"]]
        )

        experience = parse_experience(
            row[positions["experience_years"]]
        )

        skill_count = parse_integer(
            row[positions["n_skills"]]
        )

        duplicate_flag = parse_boolean(
            row[positions["near_duplicate_flag"]]
        )

        # Standardize skill separators

        normalized_skills = " | ".join(
            dict.fromkeys(
                skill.strip()
                for skill in skills.split("|")
                if skill.strip()
            )
        )

        # Construct candidate text

        text_parts = [
            f"Role Family: {role_family}",
            f"Current Role: {current_role}",
            f"Previous Role: {previous_role}",
            f"Skills: {normalized_skills}",
            f"Education: {education}"
        ]

        candidate_text = "\n".join(
            part for part in text_parts
            if part.split(":", 1)[1].strip()
        )

        candidate = {
            "record_id": record_id or None,
            "experience_years": experience,
            "current_role": current_role or None,
            "previous_role": previous_role or None,
            "role_family": role_family or None,
            "education_level": education or None,
            "key_skills": normalized_skills or None,
            "n_skills": skill_count,
            "near_duplicate_flag": duplicate_flag,
            "candidate_text": candidate_text
        }

        batch.append(candidate)

        if len(batch) >= BATCH_SIZE:

            table = pa.Table.from_pylist(
                batch,
                schema=schema
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    schema
                )

            writer.write_table(table)

            processed += len(batch)

            print(
                f"Processed {processed:,} candidates"
            )

            batch.clear()

    # Process remaining records

    if batch:

        table = pa.Table.from_pylist(
            batch,
            schema=schema
        )

        if writer is None:

            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                schema
            )

        writer.write_table(table)

        processed += len(batch)

    print("\nPreparation completed!")

    print("Total processed:", processed)

    print("Output file:", OUTPUT_FILE)

finally:

    if writer is not None:
        writer.close()

    workbook.close()

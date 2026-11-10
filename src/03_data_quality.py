
import os
import openpyxl
from collections import Counter

# ---------------------------------------
# 1. Locate the dataset
# ---------------------------------------

project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

file_path = os.path.join(
    project_folder,
    "data",
    "raw",
    "candidate_dataset.xlsx"
)

print("Starting full dataset analysis...")

# ---------------------------------------
# 2. Open Excel in read-only mode
# ---------------------------------------

workbook = openpyxl.load_workbook(
    file_path,
    read_only=True,
    data_only=True
)

# Identify candidate worksheet automatically
sheet = None

for ws in workbook.worksheets:

    headers = next(
        ws.iter_rows(
            min_row=1,
            max_row=1,
            values_only=True
        )
    )

    normalized = [
        str(h).strip().lower() if h is not None else ""
        for h in headers
    ]

    if "record_id" in normalized:
        sheet = ws
        break

if sheet is None:
    workbook.close()
    raise ValueError("Candidate worksheet not found.")

# Map column names to positions
headers = [
    str(h).strip().lower() if h is not None else ""
    for h in headers
]

positions = {
    name: index
    for index, name in enumerate(headers)
}

# ---------------------------------------
# 3. Initialize counters
# ---------------------------------------

important_columns = [
    "experience_years",
    "current_role",
    "previous_role",
    "role_family",
    "education_level",
    "key_skills",
    "n_skills",
    "has_profile_summary"
]

missing_counts = Counter()
role_counts = Counter()
role_family_counts = Counter()
skill_counts = Counter()
flag_counts = Counter()

total_records = 0
valid_experience = 0
invalid_experience = 0
experience_values = []

# ---------------------------------------
# 4. Read all candidate records
# ---------------------------------------

for row in sheet.iter_rows(
    min_row=2,
    values_only=True
):

    total_records += 1

    if total_records % 50000 == 0:
        print("Processed:", total_records, "records")

    # Check missing values
    for column in important_columns:

        index = positions.get(column)

        if index is None:
            continue

        value = row[index]

        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            missing_counts[column] += 1

    # Current role distribution
    role_index = positions.get("current_role")

    if role_index is not None:
        role = row[role_index]

        if role:
            role_counts[str(role).strip()] += 1

    # Role family distribution
    family_index = positions.get("role_family")

    if family_index is not None:
        family = row[family_index]

        if family:
            role_family_counts[str(family).strip()] += 1

    # Skill distribution
    skill_index = positions.get("key_skills")

    if skill_index is not None:
        skills = row[skill_index]

        if isinstance(skills, str):
            for skill in skills.split("|"):
                skill = skill.strip().lower()

                if skill:
                    skill_counts[skill] += 1

    # Experience analysis
    exp_index = positions.get("experience_years")

    if exp_index is not None:
        value = row[exp_index]

        try:
            if value is not None and str(value).strip():
                number = float(value)

                if 0 <= number <= 100:
                    experience_values.append(number)
                    valid_experience += 1
                else:
                    invalid_experience += 1
            else:
                invalid_experience += 1

        except (ValueError, TypeError):
            invalid_experience += 1

    # Flag distribution
    for flag in [
        "has_profile_summary",
        "near_duplicate_flag"
    ]:

        index = positions.get(flag)

        if index is not None:
            value = row[index]

            flag_counts[(flag, str(value))] += 1

# ---------------------------------------
# 5. Generate report
# ---------------------------------------

print("\n========== DATA QUALITY REPORT ==========")

print("\nTotal candidate records:", total_records)

print("\nMISSING VALUES")

for column in important_columns:
    missing = missing_counts[column]

    percentage = (
        missing / total_records * 100
        if total_records else 0
    )

    print(
        f"{column}: {missing} missing "
        f"({percentage:.2f}%)"
    )

print("\nEXPERIENCE ANALYSIS")
print("Valid experience:", valid_experience)
print("Missing or invalid experience:", invalid_experience)

if experience_values:
    experience_values.sort()

    n = len(experience_values)

    if n % 2:
        median = experience_values[n // 2]
    else:
        median = (
            experience_values[n // 2 - 1]
            + experience_values[n // 2]
        ) / 2

    print("Minimum:", experience_values[0])
    print("Maximum:", experience_values[-1])
    print("Median:", median)

print("\nTOP 15 ROLE FAMILIES")

for name, count in role_family_counts.most_common(15):
    print(name, ":", count)

print("\nTOP 15 CURRENT ROLES")

for name, count in role_counts.most_common(15):
    print(name, ":", count)

print("\nTOP 20 SKILLS")

for name, count in skill_counts.most_common(20):
    print(name, ":", count)

print("\nPROFILE AND DUPLICATE FLAGS")

for (flag, value), count in flag_counts.most_common():
    print(flag, value, ":", count)

workbook.close()

print("\nFull dataset analysis completed!")


import os
import openpyxl

# Locate the Excel file
project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

file_path = os.path.join(
    project_folder,
    "data",
    "raw",
    "candidate_dataset.xlsx"
)

workbook = openpyxl.load_workbook(
    file_path,
    read_only=True,
    data_only=True
)

print("Available worksheets:", workbook.sheetnames)

# Find the candidate worksheet using its column headers
candidate_sheet = None

for sheet in workbook.worksheets:

    headers = next(
        sheet.iter_rows(
            min_row=1,
            max_row=1,
            values_only=True
        )
    )

    headers = [
        str(value).strip().lower()
        if value is not None else ""
        for value in headers
    ]

    if "record_id" in headers and "experience_years" in headers:
        candidate_sheet = sheet
        break

if candidate_sheet is None:
    print("Candidate worksheet not found.")
    workbook.close()
    raise SystemExit

print("\nCandidate worksheet:", candidate_sheet.title)

# Map column names to positions
headers = next(
    candidate_sheet.iter_rows(
        min_row=1,
        max_row=1,
        values_only=True
    )
)

column_positions = {
    str(name).strip().lower(): index
    for index, name in enumerate(headers)
    if name is not None
}

# Select useful professional fields
selected_columns = [
    "experience_years",
    "experience_band",
    "current_role",
    "previous_role",
    "role_family",
    "education_level",
    "key_skills",
    "n_skills",
    "has_profile_summary"
]

print("\n========== CANDIDATE SAMPLES ==========")

# Read five candidate records
for number, row in enumerate(
    candidate_sheet.iter_rows(
        min_row=2,
        max_row=6,
        values_only=True
    ),
    start=1
):

    print(f"\nCandidate Sample {number}")

    for column in selected_columns:

        index = column_positions.get(column)

        if index is not None:
            value = row[index]

            if isinstance(value, str):
                value = value[:200]

            print(f"{column}: {value}")

workbook.close()

print("\nSample inspection completed!")


import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

file_path = (
    project_dir / "data" / "processed" /
    "candidates.parquet"
)

df = pd.read_parquet(
    file_path,
    columns=[
        "record_id",
        "current_role",
        "previous_role",
        "role_family",
        "experience_years",
        "near_duplicate_flag"
    ]
)

# Find records with suspicious company-like values
keywords = [
    "10xscale",
    "technologies private limited",
    "private limited",
    "pvt ltd",
    "solutions inc"
]

print("===== SUSPICIOUS ROLE INVESTIGATION =====")

for column in ["current_role", "previous_role"]:

    print(f"\nChecking column: {column}")

    for keyword in keywords:

        matches = df[
            df[column]
            .fillna("")
            .str.contains(
                keyword,
                case=False,
                regex=False
            )
        ]

        if not matches.empty:

            print(f"\nKeyword: {keyword}")
            print("Matches:", len(matches))

            print(
                matches.head(10).to_string(index=False)
            )

# Role-family source investigation is not possible
# from this Parquet file because that field was
# not included in its current schema.

print("\n===== INVESTIGATION COMPLETED =====")

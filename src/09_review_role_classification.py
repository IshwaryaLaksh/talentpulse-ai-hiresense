
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "candidates_classified_all.parquet"
)

df = pd.read_parquet(INPUT_FILE)

# Inspect each category separately

for column in ["current_role_status", "previous_role_status"]:

    print("\n" + "=" * 60)
    print("INSPECTING:", column)
    print("=" * 60)

    for status in ["MIXED", "COMPANY_ONLY"]:

        sample = df[df[column] == status]

        print(f"\n{status}: Total records = {len(sample)}")

        print(
            sample[
                [
                    "record_id",
                    "current_role",
                    "previous_role",
                    column
                ]
            ].head(20).to_string(index=False)
        )

print("\nInspection completed!")

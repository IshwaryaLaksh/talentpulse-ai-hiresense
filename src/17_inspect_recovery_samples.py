
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_employment_recovery_analysis.parquet"
)

df = pd.read_parquet(INPUT_FILE)

print("Total records:", len(df))

columns = [
    "record_id",
    "experience_years",
    "current_role",
    "previous_role",
    "role_family",
    "role_family_source",
    "key_skills",
    "candidate_text",
    "detected_role_keywords",
    "recovery_evidence_status"
]

available_columns = [
    col for col in columns if col in df.columns
]

# Display candidates with potential role evidence
samples = df[
    df["recovery_evidence_status"]
    == "POTENTIAL_ROLE_EVIDENCE"
].head(20)

print("\n========== CANDIDATE TEXT INSPECTION ==========")

for _, row in samples.iterrows():

    print("\n" + "=" * 80)

    for col in available_columns:

        value = row[col]

        if pd.isna(value):
            value = "MISSING"

        print(f"\n{col}:")
        print(str(value)[:1500])

print("\n========== INSPECTION COMPLETED ==========")

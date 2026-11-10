
from pathlib import Path
import pyarrow.parquet as pq

# Locate the prepared dataset
project_dir = Path(__file__).resolve().parent.parent

file_path = (
    project_dir / "data" / "processed" /
    "candidates.parquet"
)

# Open Parquet metadata
parquet_file = pq.ParquetFile(file_path)

print("===== DATASET VALIDATION =====")

print("Total rows:", parquet_file.metadata.num_rows)
print("Total columns:", len(parquet_file.schema_arrow))
print("Row groups:", parquet_file.num_row_groups)

print("\nCOLUMN NAMES:")
for column in parquet_file.schema_arrow.names:
    print("-", column)

# Read only a small sample
sample = next(
    parquet_file.iter_batches(batch_size=5)
).to_pandas()

print("\n===== FIRST 5 CANDIDATES =====")

for index, row in sample.iterrows():
    print(f"\nCandidate {index + 1}")
    print("Experience:", row["experience_years"])
    print("Near duplicate:", row["near_duplicate_flag"])
    print("Candidate text:")
    print(row["candidate_text"])

# Validate expected row count
expected_rows = 384035

assert parquet_file.metadata.num_rows == expected_rows, (
    "Row count does not match!"
)

print("\nVALIDATION SUCCESSFUL!")

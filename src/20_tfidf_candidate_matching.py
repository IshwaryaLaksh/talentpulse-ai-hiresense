
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# 1. PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "candidates_ml_ready.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "top20_candidate_matches.csv"

print("\n========== TALENTPULSE AI: TF-IDF MATCHING MODEL ==========")


# =====================================================
# 2. LOAD CANDIDATE DATA
# =====================================================

df = pd.read_parquet(INPUT_FILE)

print("Candidates loaded:", len(df))

df["candidate_match_text"] = (
    df["candidate_match_text"]
    .fillna("")
    .astype(str)
)

# Exclude empty profiles
df = df[
    df["candidate_match_text"].str.strip() != ""
].copy()

df = df.reset_index(drop=True)

print("Candidates available for matching:", len(df))


# =====================================================
# 3. GET JOB DESCRIPTION
# =====================================================

print("\nEnter the job description below.")
print("Type END on a new line when finished.\n")

jd_lines = []

while True:

    line = input()

    if line.strip().upper() == "END":
        break

    jd_lines.append(line)

job_description = " ".join(jd_lines).strip()

if not job_description:
    raise ValueError("Job description cannot be empty.")

print("\nJob description received successfully.")


# =====================================================
# 4. TF-IDF VECTORIZATION
# =====================================================

print("\nPreparing TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=150000,
    sublinear_tf=True
)

# Learn vocabulary from candidate profiles
candidate_vectors = vectorizer.fit_transform(
    df["candidate_match_text"]
)

print("Candidate matrix shape:", candidate_vectors.shape)

# Transform the job description using the same vocabulary
jd_vector = vectorizer.transform([job_description])

print("Job description vector created.")


# =====================================================
# 5. CALCULATE COSINE SIMILARITY
# =====================================================

print("\nCalculating candidate similarity scores...")

similarity_scores = cosine_similarity(
    jd_vector,
    candidate_vectors
).flatten()

df["tfidf_similarity_score"] = similarity_scores

df["tfidf_similarity_percentage"] = (
    df["tfidf_similarity_score"] * 100
).round(2)


# =====================================================
# 6. RANK CANDIDATES
# =====================================================

ranked_candidates = df.sort_values(
    by="tfidf_similarity_score",
    ascending=False
).copy()

ranked_candidates = ranked_candidates.head(20)

ranked_candidates.insert(
    0,
    "match_rank",
    range(1, len(ranked_candidates) + 1)
)


# =====================================================
# 7. SELECT OUTPUT COLUMNS
# =====================================================

output_columns = [
    "match_rank",
    "record_id",
    "tfidf_similarity_percentage",
    "current_role",
    "previous_role",
    "role_family",
    "skills_text",
    "experience_years",
    "education_level",
    "candidate_match_text"
]

output_columns = [
    col for col in output_columns
    if col in ranked_candidates.columns
]

results = ranked_candidates[output_columns]


# =====================================================
# 8. DISPLAY TOP 20 MATCHES
# =====================================================

print("\n========== TOP 20 MATCHING CANDIDATES ==========")

display_columns = [
    "match_rank",
    "record_id",
    "tfidf_similarity_percentage",
    "current_role",
    "experience_years",
    "role_family"
]

display_columns = [
    col for col in display_columns
    if col in results.columns
]

print(
    results[display_columns].to_string(index=False)
)


# =====================================================
# 9. SAVE RESULTS
# =====================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== MODEL EXECUTION COMPLETED ==========")

print("Total candidates evaluated:", len(df))
print("Candidates returned:", len(results))
print("Results saved to:", OUTPUT_FILE)

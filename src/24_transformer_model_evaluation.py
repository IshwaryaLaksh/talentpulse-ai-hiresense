
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# TALENTPULSE AI - TRANSFORMER MODEL EVALUATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

CANDIDATE_FILE = PROCESSED_DIR / "candidates_ml_ready.parquet"
JD_FILE = PROCESSED_DIR / "jd_requirements.json"

OUTPUT_DIR = PROCESSED_DIR / "transformer_evaluation"

SAMPLE_SIZE = 20000
BATCH_SIZE = 64
TOP_N = 20
RANDOM_STATE = 42

# Run one model at a time to manage RAM.
MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "general"
    },
    {
        "name": "all-MiniLM-L12-v2",
        "checkpoint": "sentence-transformers/all-MiniLM-L12-v2",
        "type": "general"
    },
    {
        "name": "all-mpnet-base-v2",
        "checkpoint": "sentence-transformers/all-mpnet-base-v2",
        "type": "general"
    },
    {
        "name": "multi-qa-MiniLM-L6-cos-v1",
        "checkpoint": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "type": "retrieval"
    },
    {
        "name": "multi-qa-mpnet-base-cos-v1",
        "checkpoint": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "type": "retrieval"
    },
    {
        "name": "msmarco-MiniLM-L6-cos-v5",
        "checkpoint": "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        "type": "retrieval"
    },
    {
        "name": "bge-small-en-v1.5",
        "checkpoint": "BAAI/bge-small-en-v1.5",
        "type": "bge"
    },
    {
        "name": "bge-base-en-v1.5",
        "checkpoint": "BAAI/bge-base-en-v1.5",
        "type": "bge"
    },
    {
        "name": "e5-small-v2",
        "checkpoint": "intfloat/e5-small-v2",
        "type": "e5"
    },
    {
        "name": "gte-small",
        "checkpoint": "thenlper/gte-small",
        "type": "general"
    }
]


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading candidate dataset...")

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(CANDIDATE_FILE)

    if not JD_FILE.exists():
        raise FileNotFoundError(JD_FILE)

    candidates = pd.read_parquet(CANDIDATE_FILE)

    with open(JD_FILE, "r", encoding="utf-8") as file:
        jd = json.load(file)

    if "candidate_match_text" not in candidates.columns:
        raise ValueError(
            "candidate_match_text column is missing."
        )

    # Reproducible sample for fair model comparison.
    sample = candidates.sample(
        n=min(SAMPLE_SIZE, len(candidates)),
        random_state=RANDOM_STATE
    ).copy()

    sample["candidate_match_text"] = (
        sample["candidate_match_text"]
        .fillna("")
        .astype(str)
    )

    print("Total candidates:", f"{len(candidates):,}")
    print("Evaluation sample:", f"{len(sample):,}")
    print("Job title:", jd.get("job_title"))

    return sample, jd


# ============================================================
# 2. PREPARE JOB DESCRIPTION
# ============================================================

def prepare_jd_text(jd):

    skills = jd.get("skills", {})
    experience = jd.get("experience", {})

    required = skills.get("required", [])
    preferred = skills.get("preferred", [])

    education = jd.get("education", [])

    sections = [
        "Job Title: " + str(jd.get("job_title", "")),
        "Required Skills: " + ", ".join(required),
        "Preferred Skills: " + ", ".join(preferred),
        "Education: " + ", ".join(education)
    ]

    if experience.get("minimum_years") is not None:
        sections.append(
            "Minimum Experience: "
            + str(experience["minimum_years"])
            + " years"
        )

    if experience.get("maximum_years") is not None:
        sections.append(
            "Maximum Experience: "
            + str(experience["maximum_years"])
            + " years"
        )

    return ". ".join(sections)


# ============================================================
# 3. MODEL-SPECIFIC TEXT FORMATTING
# ============================================================

def prepare_model_texts(model_type, jd_text, candidate_texts):

    if model_type == "e5":
        query = "query: " + jd_text
        documents = [
            "passage: " + text for text in candidate_texts
        ]

    elif model_type == "bge":
        query = (
            "Represent this sentence for searching relevant passages: "
            + jd_text
        )
        documents = candidate_texts

    else:
        query = jd_text
        documents = candidate_texts

    return query, documents


# ============================================================
# 4. RUN ONE MODEL
# ============================================================

def evaluate_model(model_info, sample, jd_text, device):

    model_name = model_info["name"]
    checkpoint = model_info["checkpoint"]
    model_type = model_info["type"]

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("CHECKPOINT:", checkpoint)
    print("=" * 65)

    process = psutil.Process()

    start_memory = process.memory_info().rss / (1024 ** 3)
    start_time = time.time()

    model = None

    try:

        print("\nLoading pretrained model...")

        model = SentenceTransformer(
            checkpoint,
            device=device
        )

        print("Model loaded successfully.")
        print("Device:", device)

        # Limit sequence length for manageable batch processing.
        model.max_seq_length = min(
            model.max_seq_length, 256
        )

        candidate_texts = (
            sample["candidate_match_text"].tolist()
        )

        query_text, documents = prepare_model_texts(
            model_type,
            jd_text,
            candidate_texts
        )

        print("\nGenerating candidate embeddings...")
        print("Batch size:", BATCH_SIZE)

        encoding_start = time.time()

        candidate_embeddings = model.encode(
            documents,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        candidate_embeddings = np.asarray(
            candidate_embeddings,
            dtype=np.float32
        )

        print("\nGenerating JD embedding...")

        query_embedding = model.encode(
            [query_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding[0],
            dtype=np.float32
        )

        encoding_time = time.time() - encoding_start

        # Normalized embeddings allow cosine similarity
        # to be calculated using the dot product.
        similarities = np.dot(
            candidate_embeddings,
            query_embedding
        )

        top_indices = np.argsort(
            similarities
        )[::-1][:TOP_N]

        top_candidates = sample.iloc[
            top_indices
        ].copy()

        top_candidates["transformer_similarity"] = np.round(
            similarities[top_indices] * 100, 2
        )

        top_candidates["model_name"] = model_name

        output_file = (
            OUTPUT_DIR / f"top20_{model_name}.csv"
        )

        top_candidates.to_csv(
            output_file,
            index=False
        )

        total_time = time.time() - start_time

        peak_memory = (
            process.memory_info().rss / (1024 ** 3)
        )

        embedding_dimension = candidate_embeddings.shape[1]

        result = {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "status": "SUCCESS",
            "sample_size": len(sample),
            "embedding_dimension": int(embedding_dimension),
            "encoding_time_seconds": round(encoding_time, 2),
            "total_time_seconds": round(total_time, 2),
            "memory_before_gb": round(start_memory, 2),
            "memory_after_gb": round(peak_memory, 2),
            "top20_average_similarity": round(
                float(np.mean(similarities[top_indices])) * 100,
                2
            ),
            "top20_file": str(output_file)
        }

        print("\nModel completed successfully.")
        print("Embedding dimension:", embedding_dimension)
        print("Encoding time:", result["encoding_time_seconds"], "seconds")
        print("Total time:", result["total_time_seconds"], "seconds")
        print("Memory after run:", result["memory_after_gb"], "GB")
        print(
            "Top 20 average similarity:",
            result["top20_average_similarity"], "%"
        )

        return result

    except Exception as error:

        print("\nMODEL FAILED:", model_name)
        print("Error:", str(error))

        return {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "status": "FAILED",
            "error": str(error)
        }

    finally:

        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 5. SAVE COMPARISON REPORT
# ============================================================

def save_report(results):

    results_df = pd.DataFrame(results)

    report_file = OUTPUT_DIR / "model_comparison.csv"

    results_df.to_csv(
        report_file,
        index=False
    )

    successful = results_df[
        results_df["status"] == "SUCCESS"
    ].copy()

    if not successful.empty:

        successful = successful.sort_values(
            by="encoding_time_seconds",
            ascending=True
        )

        print("\n" + "=" * 65)
        print("TRANSFORMER MODEL COMPARISON")
        print("=" * 65)

        columns = [
            "model_name",
            "embedding_dimension",
            "encoding_time_seconds",
            "total_time_seconds",
            "memory_after_gb",
            "top20_average_similarity"
        ]

        print(
            successful[columns].to_string(index=False)
        )

        print(
            "\nNote: Similarity percentages are not accuracy scores."
        )

    print("\nComparison report saved to:")
    print(report_file)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n============================================")
    print(" TALENTPULSE AI - TRANSFORMER BENCHMARK")
    print("============================================")

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Selected device:", device)

    sample, jd = load_data()

    jd_text = prepare_jd_text(jd)

    print("\nJD text prepared successfully.")

    results = []

    for index, model_info in enumerate(MODELS, start=1):

        print(f"\nMODEL PROGRESS: {index}/{len(MODELS)}")

        result = evaluate_model(
            model_info,
            sample,
            jd_text,
            device
        )

        results.append(result)

        # Save progress after every model.
        with open(
            OUTPUT_DIR / "model_comparison_progress.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(results, file, indent=4)

    save_report(results)

    print("\n============================================")
    print(" TRANSFORMER BENCHMARK FINISHED")
    print("============================================")


if __name__ == "__main__":
    main()

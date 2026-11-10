
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# TALENTPULSE AI - JD-DRIVEN DYNAMIC WEIGHT ALLOCATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "jd_requirements.json"
OUTPUT_FILE = PROCESSED_DIR / "jd_dynamic_weights.json"


def load_jd():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_weights(jd):

    # Read the actual nested structure from Step 21
    skills = jd.get("skills", {})
    experience = jd.get("experience", {})
    education = jd.get("education", [])

    required_skills = skills.get("required", [])
    preferred_skills = skills.get("preferred", [])

    min_experience = experience.get("minimum_years")
    max_experience = experience.get("maximum_years")

    job_title = jd.get("job_title")

    # --------------------------------------------------------
    # IMPORTANCE RATINGS
    # --------------------------------------------------------

    ratings = {}
    reasons = {}

    # 1. Skills
    if required_skills:
        ratings["skills"] = 5
        reasons["skills"] = (
            f"High importance: {len(required_skills)} "
            "required skills identified in the JD."
        )

    elif preferred_skills:
        ratings["skills"] = 3
        reasons["skills"] = (
            f"Moderate importance: {len(preferred_skills)} "
            "preferred skills identified in the JD."
        )

    # 2. Experience
    if min_experience is not None:
        ratings["experience"] = 5
        reasons["experience"] = (
            f"High importance: minimum experience requirement "
            f"is {min_experience} years."
        )

    elif max_experience is not None:
        ratings["experience"] = 3
        reasons["experience"] = (
            f"Moderate importance: maximum experience is "
            f"{max_experience} years."
        )

    # 3. Role relevance
    if job_title and str(job_title).strip():
        ratings["role_relevance"] = 4
        reasons["role_relevance"] = (
            f"Job title '{job_title}' is available for "
            "occupational relevance matching."
        )

    # 4. Education
    if education:
        ratings["education"] = 4
        reasons["education"] = (
            f"{len(education)} education requirements "
            "identified in the JD."
        )

    # 5. Semantic similarity
    if job_title or required_skills or preferred_skills:
        ratings["semantic_similarity"] = 3
        reasons["semantic_similarity"] = (
            "Supporting criterion for contextual matching "
            "between JD and candidate information."
        )

    if not ratings:
        raise ValueError("No usable JD requirements found.")

    # --------------------------------------------------------
    # NORMALIZE IMPORTANCE RATINGS TO 100%
    # --------------------------------------------------------

    total_rating = sum(ratings.values())

    weights = {
        criterion: round((rating / total_rating) * 100, 2)
        for criterion, rating in ratings.items()
    }

    # Ensure rounded weights total exactly 100
    difference = round(100 - sum(weights.values()), 2)

    if difference:
        highest = max(weights, key=weights.get)
        weights[highest] = round(weights[highest] + difference, 2)

    return ratings, weights, reasons


def main():

    print("\n========== TALENTPULSE AI ==========")
    print("JD-DRIVEN DYNAMIC WEIGHT ALLOCATION")
    print("====================================")

    jd = load_jd()

    ratings, weights, reasons = calculate_weights(jd)

    skills = jd.get("skills", {})
    experience = jd.get("experience", {})
    education = jd.get("education", [])

    print("\nJob title:", jd.get("job_title"))
    print("Required skills:", len(skills.get("required", [])))
    print("Preferred skills:", len(skills.get("preferred", [])))
    print("Unclassified skills:", len(skills.get("unclassified", [])))
    print("Education requirements:", len(education))
    print("Minimum experience:", experience.get("minimum_years"))
    print("Maximum experience:", experience.get("maximum_years"))

    print("\n========== IMPORTANCE RATINGS ==========")

    for criterion, rating in ratings.items():
        print(f"{criterion}: {rating}/5")

    print("\n========== FINAL MATCHING WEIGHTS ==========")

    for criterion, weight in weights.items():
        print(f"{criterion}: {weight}%")

    print("\n========== EXPLANATION ==========")

    for criterion, reason in reasons.items():
        print(f"\n{criterion}: {reason}")

    print("\nTotal weight:", sum(weights.values()))

    output = {
        "job_title": jd.get("job_title"),
        "weighting_method": "JD-driven importance rating",
        "methodology_note": (
            "Project-designed heuristic rubric, not an "
            "official industry-standard weighting matrix."
        ),
        "importance_ratings": ratings,
        "weights": weights,
        "reasons": reasons,
        "generated_at": datetime.now().isoformat()
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nWeights saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

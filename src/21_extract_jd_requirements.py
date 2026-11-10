
import pandas as pd
import re
import json
from pathlib import Path

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

OUTPUT_FILE = OUTPUT_DIR / "jd_requirements.json"

print("\n========== TALENTPULSE AI: IMPROVED JD EXTRACTION ==========")


# =====================================================
# 2. LOAD CANDIDATE DATA
# =====================================================

df = pd.read_parquet(INPUT_FILE)

print("Candidate records loaded:", len(df))


# =====================================================
# 3. CURATED SKILL DICTIONARY
# =====================================================

skill_categories = {

    "programming": [
        "python", "java", "javascript", "typescript",
        "c", "c++", "c#", "r", "scala"
    ],

    "database": [
        "sql", "mysql", "postgresql", "mongodb",
        "oracle", "sql server", "database management"
    ],

    "business_intelligence": [
        "power bi", "tableau", "excel",
        "data visualization", "dashboard development",
        "reporting", "dax", "power query"
    ],

    "data_analytics": [
        "data analytics", "data analysis",
        "data cleaning", "data modeling",
        "statistical analysis", "statistics",
        "data interpretation", "data mining"
    ],

    "machine_learning": [
        "machine learning", "deep learning",
        "natural language processing", "nlp",
        "artificial intelligence", "predictive modeling",
        "scikit-learn", "tensorflow", "pytorch"
    ],

    "cloud": [
        "aws", "azure", "google cloud",
        "gcp", "databricks", "snowflake"
    ],

    "testing": [
        "selenium", "cypress", "postman",
        "automation testing", "manual testing",
        "api testing", "performance testing",
        "regression testing", "playwright"
    ],

    "business_operations": [
        "supply chain management",
        "inventory management",
        "procurement", "logistics",
        "process improvement",
        "business analysis",
        "operations management",
        "demand forecasting",
        "project management"
    ],

    "project_tools": [
        "jira", "agile", "scrum",
        "git", "jenkins", "confluence"
    ]
}


# Flatten dictionary
all_skills = {}

for category, skills in skill_categories.items():

    for skill in skills:

        all_skills[skill] = category


print("Curated skills loaded:", len(all_skills))


# =====================================================
# 4. GET JOB DESCRIPTION
# =====================================================

print("\nPaste the complete Job Description.")
print("Type END on a new line when finished.\n")

jd_lines = []

while True:

    line = input()

    if line.strip().upper() == "END":
        break

    jd_lines.append(line)

jd_text = "\n".join(jd_lines).strip()

if not jd_text:
    raise ValueError("Job description cannot be empty.")


# Normalize dashes
jd_text = jd_text.replace("–", "-").replace("—", "-")

jd_lower = jd_text.lower()


# =====================================================
# 5. EXTRACT JOB TITLE
# =====================================================

job_title = "Not identified"

title_patterns = [
    r"job title\s*[:\-]\s*(.+)",
    r"position\s*[:\-]\s*(.+)",
    r"role\s*[:\-]\s*(.+)",
    r"designation\s*[:\-]\s*(.+)"
]

for pattern in title_patterns:

    match = re.search(
        pattern,
        jd_text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if match:

        job_title = match.group(1).strip()
        break


# =====================================================
# 6. EXTRACT EXPERIENCE
# =====================================================

min_experience = None
max_experience = None

experience_patterns = [

    # 2-4 years, 2 to 4 years
    (
        r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*"
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?"
    ),

    # 4+ years, minimum 4 years, at least 4 years
    (
        r"(?:minimum(?:\s+of)?|at least)\s*"
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?"
    ),

    (
        r"(\d+(?:\.\d+)?)\s*\+\s*years?"
    ),

    # 4 years of experience
    (
        r"(\d+(?:\.\d+)?)\s+years?\s+of\s+experience"
    )
]

for index, pattern in enumerate(experience_patterns):

    match = re.search(pattern, jd_lower)

    if not match:
        continue

    if index == 0:

        min_experience = float(match.group(1))
        max_experience = float(match.group(2))

    else:

        min_experience = float(match.group(1))
        max_experience = None

    break


# =====================================================
# 7. IDENTIFY SKILLS IN JD
# =====================================================

def find_skills(text):

    found = []

    for skill in all_skills:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, text, re.IGNORECASE):

            found.append(skill)

    return found


identified_skills = find_skills(jd_lower)


# =====================================================
# 8. CLASSIFY REQUIRED AND PREFERRED SKILLS
# =====================================================

required_skills = set()
preferred_skills = set()
unclassified_skills = set()

lines = jd_lower.splitlines()

current_section = "unclassified"

for line in lines:

    line = line.strip()

    if not line:
        continue

    # Detect section headings
    if any(word in line for word in [
        "preferred skills",
        "good to have",
        "nice to have",
        "desirable skills"
    ]):

        current_section = "preferred"
        continue

    if any(word in line for word in [
        "required skills",
        "mandatory skills",
        "must have",
        "essential skills"
    ]):

        current_section = "required"
        continue

    line_skills = find_skills(line)

    for skill in line_skills:

        if current_section == "required":
            required_skills.add(skill)

        elif current_section == "preferred":
            preferred_skills.add(skill)

        else:
            unclassified_skills.add(skill)


# Remove duplicate classifications
preferred_skills -= required_skills
unclassified_skills -= required_skills
unclassified_skills -= preferred_skills


# =====================================================
# 9. EXTRACT EDUCATION
# =====================================================

education_keywords = [
    "bachelor's degree",
    "bachelors degree",
    "bachelor",
    "b.tech",
    "b.e.",
    "b.sc",
    "master's degree",
    "masters degree",
    "master",
    "m.tech",
    "m.sc",
    "mba",
    "phd",
    "doctorate",
    "diploma"
]

education_requirements = []

for keyword in education_keywords:

    if keyword in jd_lower:

        education_requirements.append(keyword)


education_requirements = list(
    dict.fromkeys(education_requirements)
)


# =====================================================
# 10. EXTRACT RESPONSIBILITY TEXT
# =====================================================

responsibility_headers = [
    "responsibilities",
    "key responsibilities",
    "what you will do",
    "job duties",
    "roles and responsibilities"
]

responsibility_sections = []

for line in lines:

    if any(
        header in line
        for header in responsibility_headers
    ):

        responsibility_sections.append(line.strip())


# =====================================================
# 11. BUILD STRUCTURED REQUIREMENTS
# =====================================================

jd_requirements = {

    "job_title": job_title,

    "experience": {
        "minimum_years": min_experience,
        "maximum_years": max_experience
    },

    "skills": {

        "required": sorted(required_skills),

        "preferred": sorted(preferred_skills),

        "unclassified": sorted(unclassified_skills),

        "categories": {
            skill: all_skills[skill]
            for skill in identified_skills
        }
    },

    "education": education_requirements,

    "responsibility_sections": responsibility_sections,

    "extraction_metadata": {

        "total_jd_skills_identified": len(identified_skills),

        "required_skill_count": len(required_skills),

        "preferred_skill_count": len(preferred_skills),

        "unclassified_skill_count": len(unclassified_skills),

        "skill_dictionary_size": len(all_skills),

        "extraction_method": "Curated skill dictionary and rule-based NLP"
    }
}


# =====================================================
# 12. DISPLAY RESULTS
# =====================================================

print("\n========== EXTRACTED JOB REQUIREMENTS ==========")

print("\nJOB TITLE:")
print(job_title)

print("\nEXPERIENCE:")
print("Minimum:", min_experience)
print("Maximum:", max_experience)

print("\nREQUIRED SKILLS:")
print(sorted(required_skills))

print("\nPREFERRED SKILLS:")
print(sorted(preferred_skills))

print("\nUNCLASSIFIED SKILLS:")
print(sorted(unclassified_skills))

print("\nSKILL CATEGORIES:")
print(jd_requirements["skills"]["categories"])

print("\nEDUCATION:")
print(education_requirements)

print("\nEXTRACTION METADATA:")
print(jd_requirements["extraction_metadata"])


# =====================================================
# 13. SAVE JSON
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        jd_requirements,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nSaved structured JD:", OUTPUT_FILE)

print("\n========== JD EXTRACTION COMPLETED ==========")

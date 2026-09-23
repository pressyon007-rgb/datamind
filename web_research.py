import os
import json
import pandas as pd

from google import genai


# ==========================================================
# CREATE GEMINI CLIENT
# ==========================================================

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Please add it to your .env file."
        )

    return genai.Client(api_key=api_key)


# ==========================================================
# BUILD DATASET SUMMARY
# ==========================================================

def build_dataset_summary(df):

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),

        "data_types": {
            column: str(df[column].dtype)
            for column in df.columns
        },

        "missing_values": {
            column: int(df[column].isna().sum())
            for column in df.columns
        },

        "unique_values": {
            column: int(df[column].nunique(dropna=True))
            for column in df.columns
        }
    }

    # Sample values
    summary["sample_values"] = {}

    for column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        summary["sample_values"][column] = values

    # Numeric summary
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    summary["numeric_summary"] = {}

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) > 0:

            summary["numeric_summary"][column] = {
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "average": float(series.mean()),
                "median": float(series.median())
            }

    return summary


# ==========================================================
# DETECT DATA DOMAIN
# ==========================================================

def detect_domain(df):

    columns_text = " ".join(
        str(column).lower()
        for column in df.columns
    )

    sample_text = ""

    for column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .head(10)
            .tolist()
        )

        sample_text += " ".join(values).lower() + " "

    combined_text = columns_text + " " + sample_text

    domain_keywords = {

        "Healthcare / Medical": [
            "diagnosis",
            "cancer",
            "tumor",
            "patient",
            "disease",
            "medical",
            "hospital",
            "clinical",
            "symptom",
            "treatment",
            "biopsy",
            "stage",
            "survival",
            "benign",
            "malignant",
            "pathology"
        ],

        "Human Resources": [
            "employee",
            "attrition",
            "salary",
            "department",
            "jobrole",
            "job role",
            "satisfaction",
            "overtime",
            "years at company",
            "performance",
            "manager"
        ],

        "Marketing": [
            "campaign",
            "marketing",
            "conversion",
            "click",
            "impression",
            "response",
            "channel",
            "lead",
            "customer",
            "promotion",
            "advertisement"
        ],

        "Retail / Sales": [
            "sales",
            "revenue",
            "product",
            "quantity",
            "customer",
            "order",
            "store",
            "discount",
            "purchase",
            "invoice"
        ],

        "Finance": [
            "loan",
            "credit",
            "transaction",
            "account",
            "balance",
            "interest",
            "fraud",
            "payment",
            "bank",
            "default"
        ],

        "Education": [
            "student",
            "marks",
            "grade",
            "exam",
            "school",
            "college",
            "attendance",
            "education",
            "subject",
            "teacher"
        ]
    }

    scores = {}

    for domain, keywords in domain_keywords.items():

        score = 0

        for keyword in keywords:

            if keyword in combined_text:
                score += 1

        scores[domain] = score

    detected_domain = max(
        scores,
        key=scores.get
    )

    if scores[detected_domain] == 0:
        detected_domain = "General Data Analysis"

    return detected_domain, scores


# ==========================================================
# LOCAL DOMAIN KNOWLEDGE
# ==========================================================

def get_local_domain_context(domain):

    contexts = {

        "Healthcare / Medical": """
This dataset belongs to the healthcare/medical domain.

The analysis should focus on:
- patient or clinical data patterns
- diagnosis categories
- clinical variables
- data quality
- relationships between medical variables
- research-oriented analysis
- population/group comparisons
- possible areas for further investigation

Do not diagnose individuals.
Do not provide treatment recommendations.
Do not claim that correlation proves medical causation.
Treat findings as educational/research analysis.
""",

        "Human Resources": """
This dataset belongs to the Human Resources domain.

The analysis should focus on:
- employee characteristics
- attrition and retention
- departments and job roles
- employee satisfaction
- salary and compensation patterns
- workforce distribution
- employee experience
- HR process improvements
""",

        "Marketing": """
This dataset belongs to the Marketing domain.

The analysis should focus on:
- campaigns
- customer response
- conversion
- channels
- promotions
- customer segments
- campaign performance
- marketing opportunities
""",

        "Retail / Sales": """
This dataset belongs to the Retail/Sales domain.

The analysis should focus on:
- products
- customers
- sales
- revenue
- quantity
- stores
- discounts
- purchasing patterns
- product and customer opportunities
""",

        "Finance": """
This dataset belongs to the Finance domain.

The analysis should focus on:
- financial transactions
- credit
- loans
- payments
- account behaviour
- financial risk patterns
- data quality
- suspicious or unusual patterns where appropriate
""",

        "Education": """
This dataset belongs to the Education domain.

The analysis should focus on:
- students
- academic performance
- attendance
- grades
- subjects
- student groups
- educational performance patterns
- areas for academic improvement
"""
    }

    return contexts.get(
        domain,
        """
This is a general data-analysis dataset.
Focus on the actual variables and relationships present in the uploaded data.
Do not assume that it represents a business unless the dataset supports that conclusion.
"""
    )


# ==========================================================
# RESEARCH / AI ANALYSIS
# ==========================================================

def research_dataset(df):

    dataset_summary = build_dataset_summary(df)

    detected_domain, domain_scores = detect_domain(df)

    summary_json = json.dumps(
        dataset_summary,
        indent=2,
        default=str
    )

    domain_context = get_local_domain_context(
        detected_domain
    )

    # ------------------------------------------------------
    # Try Gemini
    # ------------------------------------------------------

    try:

        client = get_gemini_client()

        prompt = f"""
You are an expert data analyst.

A user uploaded a dataset to an automated data analysis platform.

Detected domain:
{detected_domain}

Domain detection scores:
{domain_scores}

Domain context:
{domain_context}

Dataset information:
{summary_json}

Your task is to analyze the actual uploaded dataset.

IMPORTANT:
Do not assume every dataset is a business dataset.

Explain:

1. DOMAIN IDENTIFICATION
2. DATASET SUBJECT
3. IMPORTANT VARIABLES
4. IMPORTANT DATA PATTERNS
5. REAL-WORLD CONTEXT
6. DATA LIMITATIONS
7. DOMAIN-SPECIFIC ANALYSIS IDEAS
8. RESEARCH / DEVELOPMENT OPPORTUNITIES

Use simple language.

Only make claims supported by the uploaded dataset.

Do not invent numerical findings.

For healthcare data:
- Treat the analysis as research/educational.
- Do not diagnose individual patients.
- Do not provide individual treatment advice.
- Do not claim correlation proves medical causation.
"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return {
            "domain": detected_domain,
            "domain_scores": domain_scores,
            "analysis": response.text,
            "response": response,
            "source": "Gemini AI"
        }

    # ------------------------------------------------------
    # FREE FALLBACK
    # ------------------------------------------------------

    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):

            fallback_message = f"""
DOMAIN IDENTIFICATION

The uploaded dataset has been identified as:
{detected_domain}

DATASET SUBJECT

The dataset contains {len(df):,} records and
{len(df.columns):,} columns.

IMPORTANT VARIABLES

The main variables are:

{", ".join(str(column) for column in df.columns)}

DATA QUALITY

The platform identified missing values and
different data types automatically.

IMPORTANT DATA PATTERNS

The dashboard and statistical analysis should be
used to investigate relationships between the
variables, category distributions, unusual values,
and missing-data patterns.

REAL-WORLD CONTEXT

{domain_context}

DATA LIMITATIONS

The dataset should be interpreted according to
its source, sample size, variable definitions,
missing values, and collection method.

DOMAIN-SPECIFIC ANALYSIS IDEAS

1. Compare important categories within the dataset.
2. Examine relationships between numeric variables.
3. Investigate differences between groups.
4. Identify unusual or outlying observations.
5. Examine missing-data patterns.
6. Identify variables that may require deeper analysis.

AI STATUS

Gemini API quota is currently unavailable.
The platform has automatically switched to
local analysis so the application can continue
working without additional API cost.

Gemini error:
{error_text[:300]}
"""

            return {
                "domain": detected_domain,
                "domain_scores": domain_scores,
                "analysis": fallback_message,
                "response": None,
                "source": "Local Free Fallback"
            }

        # --------------------------------------------------
        # Other Gemini errors
        # --------------------------------------------------

        fallback_message = f"""
DOMAIN IDENTIFICATION

Detected domain:
{detected_domain}

DATASET SIZE

Records: {len(df):,}
Columns: {len(df.columns):,}

IMPORTANT VARIABLES

{", ".join(str(column) for column in df.columns)}

DOMAIN CONTEXT

{domain_context}

The AI research service could not be reached.
The application is continuing with local dataset
analysis.

Error:
{error_text[:300]}
"""

        return {
            "domain": detected_domain,
            "domain_scores": domain_scores,
            "analysis": fallback_message,
            "response": None,
            "source": "Local Analysis Fallback"
        }


# ==========================================================
# GET WEB SOURCES
# ==========================================================

def get_research_sources(response):

    # Google Search grounding has been removed.
    # Therefore there are no web sources to extract.

    return []
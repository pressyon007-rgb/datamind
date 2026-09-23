# modules/recommendation_engine.py

import re
import numpy as np
import pandas as pd


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(df, aliases):
    """
    Find a column using exact or partial matching.
    """

    normalized_columns = {
        normalize_name(col): col
        for col in df.columns
    }

    # Exact match
    for alias in aliases:
        key = normalize_name(alias)

        if key in normalized_columns:
            return normalized_columns[key]

    # Partial match
    for col in df.columns:

        col_norm = normalize_name(col)

        for alias in aliases:

            alias_norm = normalize_name(alias)

            if alias_norm in col_norm or col_norm in alias_norm:
                return col

    return None


def format_number(value):

    try:

        value = float(value)

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    except Exception:

        return str(value)


def format_percent(value):

    try:
        return f"{float(value):.1f}%"

    except Exception:
        return "N/A"


# ============================================================
# DOMAIN DETECTION
# ============================================================

DOMAIN_KEYWORDS = {

    "HR": [
        "employee",
        "attrition",
        "overtime",
        "jobrole",
        "jobsatisfaction",
        "worklifebalance",
        "monthlyincome",
        "department",
        "businesstravel",
        "yearsatcompany",
        "joblevel"
    ],

    "Retail": [
        "sales",
        "store",
        "product",
        "quantity",
        "inventory",
        "stock",
        "discount",
        "category"
    ],

    "E-commerce": [
        "order",
        "customer",
        "cart",
        "purchase",
        "return",
        "shipping",
        "delivery",
        "rating",
        "review"
    ],

    "Marketing": [
        "campaign",
        "conversion",
        "converted",
        "response",
        "click",
        "impression",
        "lead",
        "channel",
        "marketing",
        "spend"
    ],

    "Finance": [
        "loan",
        "credit",
        "default",
        "fraud",
        "transaction",
        "balance",
        "interest",
        "risk",
        "payment"
    ],

    "Healthcare": [
        "patient",
        "diagnosis",
        "disease",
        "hospital",
        "doctor",
        "treatment",
        "admission",
        "discharge",
        "medical"
    ],

    "Education": [
        "student",
        "marks",
        "grade",
        "score",
        "attendance",
        "subject",
        "course",
        "exam",
        "school",
        "college"
    ],

    "Telecom": [
        "subscriber",
        "churn",
        "plan",
        "tenure",
        "usage",
        "contract",
        "internet",
        "monthlycharges",
        "calls"
    ],

    "Manufacturing": [
        "machine",
        "defect",
        "production",
        "quality",
        "maintenance",
        "downtime",
        "failure",
        "temperature",
        "pressure"
    ],

    "Supply Chain": [
        "supplier",
        "warehouse",
        "shipment",
        "delivery",
        "leadtime",
        "inventory",
        "stockout",
        "logistics"
    ]
}


def detect_industry(df):

    scores = {}

    for industry, keywords in DOMAIN_KEYWORDS.items():

        score = 0

        for column in df.columns:

            column_name = normalize_name(column)

            for keyword in keywords:

                if normalize_name(keyword) in column_name:

                    score += 1

        scores[industry] = score

    if not scores:
        return "General Business"

    industry = max(scores, key=scores.get)

    if scores[industry] == 0:

        return "General Business"

    return industry


# ============================================================
# TARGET DETECTION
# ============================================================

TARGET_ALIASES = [

    "Attrition",
    "Churn",
    "Churned",
    "Default",
    "Fraud",
    "Converted",
    "Conversion",
    "Response",
    "Responded",
    "Purchased",
    "Purchase",
    "Returned",
    "Return",
    "Defective",
    "Defect",
    "Late",
    "Stockout",
    "Exited",
    "Left",
    "Outcome",
    "Target",
    "Status"
]


def detect_target(df):

    target = find_column(
        df,
        TARGET_ALIASES
    )

    if target:

        values = (
            df[target]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        if len(values) <= 10:

            return target

    # Detect binary categorical column
    for column in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        if len(values) == 2:

            return column

    return None


# ============================================================
# KPI / METRIC DETECTION
# ============================================================

METRIC_ALIASES = [

    "Sales",
    "Revenue",
    "Profit",
    "Amount",
    "Income",
    "Salary",
    "Price",
    "Quantity",
    "Demand",
    "Units",
    "Score",
    "Rating",
    "Cost",
    "Spend",
    "Margin"
]


def detect_metric(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        return None

    metric = find_column(
        df,
        METRIC_ALIASES
    )

    if metric in numeric_columns:

        return metric

    # Ignore obvious ID columns
    for column in numeric_columns:

        name = normalize_name(column)

        if not any(
            word in name
            for word in [
                "id",
                "number",
                "code"
            ]
        ):

            return column

    return numeric_columns[0]


# ============================================================
# POSITIVE TARGET DETECTION
# ============================================================

POSITIVE_VALUES = {

    "yes",
    "y",
    "true",
    "1",

    "left",
    "attrition",

    "churn",
    "churned",

    "default",

    "fraud",

    "bad",
    "failed",

    "converted",
    "conversion",

    "responded",

    "purchased",

    "returned",

    "defective",
    "defect",

    "late",

    "stockout"
}


def create_positive_target(series):

    text = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive = text.isin(
        POSITIVE_VALUES
    )

    # If values are unknown but exactly two classes exist,
    # use the less frequent class as the event.
    if positive.sum() == 0:

        unique_values = text.unique()

        if len(unique_values) == 2:

            counts = text.value_counts()

            event = counts.index[-1]

            positive = text == event

    return positive.astype(int)


# ============================================================
# BASE RECOMMENDATION
# ============================================================

def create_recommendation(
    business_area,
    problem,
    status
):

    return {

        "Business Area": business_area,

        "Industry Problem": problem,

        "Problem": problem,

        "Status": status,

        "Data Evidence": "",

        "Evidence": "",

        "Detailed Analysis": "",

        "Factor Analysis": "",

        "Business Problem": "",

        "Business Impact": "",

        "Corrective Measures": [],

        "How to Improve Current Working": "",

        "How to Improve Working": "",

        "Workflow": "",

        "Development Opportunity": "",

        "Expected Outcome": "",

        "Additional Data Required": [],

        "Limitation": ""

    }


# ============================================================
# DATA QUALITY — MISSING VALUES
# ============================================================

def check_missing_values(df):

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        return None

    total_missing = int(
        missing.sum()
    )

    total_cells = max(
        df.shape[0] * df.shape[1],
        1
    )

    missing_percent = (
        total_missing /
        total_cells *
        100
    )

    affected_columns = []

    for column, count in (
        missing
        .sort_values(
            ascending=False
        )
        .head(8)
        .items()
    ):

        affected_columns.append(

            f"{column} → "
            f"{int(count):,} missing "
            f"({count / max(len(df), 1) * 100:.1f}%)"

        )

    recommendation = create_recommendation(

        "Data Quality — Missing Values",

        "Incomplete Information",

        "🔴 Problem Detected"

    )

    recommendation["Evidence"] = (

        f"The dataset contains "
        f"{total_missing:,} missing cells, "
        f"representing "
        f"{missing_percent:.1f}% "
        f"of all available cells."
    )

    recommendation["Detailed Analysis"] = (

        "Missing information can affect "
        "business calculations, "
        "segmentation, comparisons "
        "and predictive analysis. "
        "The affected columns should "
        "be reviewed according to "
        "their business importance."
    )

    recommendation["Factor Analysis"] = "\n".join(
        affected_columns
    )

    recommendation["Business Impact"] = (

        "Incomplete information may "
        "produce unreliable KPIs, "
        "incorrect comparisons and "
        "incomplete business decisions."
    )

    recommendation["Corrective Measures"] = [

        "Identify business-critical columns with missing values.",

        "Investigate why the information is missing.",

        "Check whether the missing values originate from the source system.",

        "Define documented rules for handling each important field.",

        "Monitor missing-value percentages whenever new data is uploaded."

    ]

    recommendation["How to Improve Working"] = (

        "Add an automated data-quality "
        "validation step before dashboards, "
        "reports and predictions are generated."
    )

    recommendation["Workflow"] = """

Upload Data
      ↓
Check Missing Values
      ↓
Identify Critical Columns
      ↓
Investigate Missing Records
      ↓
Correct Source / Apply Valid Rules
      ↓
Recalculate Analysis
      ↓
Monitor Data Quality

"""

    recommendation["Development Opportunity"] = (

        "Create a Data Quality Monitoring "
        "Dashboard showing missing counts, "
        "missing percentages and affected columns."
    )

    recommendation["Expected Outcome"] = (

        "Improved data completeness and "
        "more reliable business reporting."
    )

    recommendation["Additional Data Required"] = [

        "Reason for missing values",

        "Source-system information",

        "Mandatory-field definitions"

    ]

    recommendation["Limitation"] = (

        "Missing values alone do not indicate "
        "whether the underlying business process "
        "is poor; the reason for missingness "
        "must be investigated."
    )

    return recommendation


# ============================================================
# DATA QUALITY — DUPLICATES
# ============================================================

def check_duplicates(df):

    id_column = find_column(
        df,
        [
            "EmployeeID",
            "EmployeeNumber",
            "CustomerID",
            "CustomerNumber",
            "PatientID",
            "OrderID",
            "TransactionID",
            "ProductID",
            "StudentID",
            "RecordID",
            "ID"
        ]
    )

    exact_duplicates = int(
        df.duplicated().sum()
    )

    if id_column:

        duplicate_ids = int(
            df[id_column]
            .duplicated()
            .sum()
        )

    else:

        duplicate_ids = 0

    problem_exists = (
        exact_duplicates > 0
        or duplicate_ids > 0
    )

    status = (
        "🔴 Problem Detected"
        if problem_exists
        else
        "🟢 No Evidence Detected"
    )

    recommendation = create_recommendation(

        "Data Quality — Duplicate Records",

        "Duplicate Records",

        status

    )

    recommendation["Evidence"] = (

        f"Total records: "
        f"{len(df):,}. "
        f"Exact duplicate rows: "
        f"{exact_duplicates:,}. "
        f"Duplicate identifier records: "
        f"{duplicate_ids:,}."
    )

    if problem_exists:

        recommendation["Detailed Analysis"] = (

            "Repeated records were detected. "
            "They should be reviewed to determine "
            "whether they represent legitimate "
            "repeated business events or duplicate "
            "records."
        )

        recommendation["Business Impact"] = (

            "Duplicates can inflate record counts, "
            "distort KPIs and affect business reporting."
        )

        recommendation["Corrective Measures"] = [

            "Identify the source of repeated records.",

            "Separate valid repeated transactions from true duplicates.",

            "Define a unique-record rule.",

            "Remove or consolidate confirmed duplicates.",

            "Run duplicate validation whenever new data is uploaded."

        ]

    else:

        recommendation["Detailed Analysis"] = (

            "No duplicate records were detected "
            "using the available identifier and "
            "exact-row checks."
        )

        recommendation["Business Impact"] = (

            "No current duplicate-record issue "
            "was identified in the available data."
        )

        recommendation["Corrective Measures"] = [

            "Continue unique identifier validation.",

            "Check duplicate records whenever new data is uploaded.",

            "Maintain a master identifier rule."

        ]

    recommendation["How to Improve Working"] = (

        "Include duplicate validation as a "
        "standard step in the data ingestion process."
    )

    recommendation["Development Opportunity"] = (

        "Add an automated duplicate-record "
        "monitoring component to the BI platform."
    )

    recommendation["Expected Outcome"] = (

        "Prevent duplicate records from "
        "affecting future reporting."
    )

    recommendation["Limitation"] = (

        "The check is limited to available "
        "identifier fields and exact duplicate rows."
    )

    return recommendation


# ============================================================
# TARGET / FACTOR ANALYSIS
# ============================================================

def analyze_target_factors(
    df,
    target,
    industry
):

    recommendations = []

    positive = create_positive_target(
        df[target]
    )

    for factor in df.columns:

        if factor == target:
            continue

        unique_count = (
            df[factor]
            .nunique(
                dropna=True
            )
        )

        # Useful categorical / low-cardinality dimensions
        if unique_count < 2:
            continue

        if unique_count > 15:
            continue

        temp = pd.DataFrame({

            "factor":
                df[factor]
                .fillna("Unknown")
                .astype(str),

            "event":
                positive

        })

        grouped = (

            temp
            .groupby("factor")["event"]
            .agg(
                ["count", "mean"]
            )
            .reset_index()

        )

        grouped["rate"] = (
            grouped["mean"] * 100
        )

        if len(grouped) < 2:
            continue

        highest = (
            grouped
            .sort_values(
                "rate",
                ascending=False
            )
            .iloc[0]
        )

        lowest = (
            grouped
            .sort_values(
                "rate",
                ascending=True
            )
            .iloc[0]
        )

        spread = (
            highest["rate"]
            -
            lowest["rate"]
        )

        minimum_group_size = max(
            5,
            int(len(df) * 0.01)
        )

        reliable = (
            grouped["count"].min()
            >=
            minimum_group_size
        )

        if not reliable:

            status = (
                "🟡 Needs Investigation"
            )

        elif spread >= 5:

            status = (
                "🔴 Problem Detected"
            )

        else:

            status = (
                "🟢 No Evidence Detected"
            )

        recommendation = create_recommendation(

            f"{industry} — {factor} Analysis",

            f"{factor} / {target} Pattern",

            status

        )

        recommendation["Evidence"] = (

            f"The dataset contains "
            f"{len(df):,} records. "

            f"The highest observed "
            f"{target} rate is "
            f"{highest['rate']:.1f}% "
            f"for {highest['factor']}. "

            f"The lowest observed rate is "
            f"{lowest['rate']:.1f}% "
            f"for {lowest['factor']}. "

            f"The observed difference is "
            f"{spread:.1f} percentage points."
        )

        if status == "🔴 Problem Detected":

            analysis_text = (

                f"The data shows a noticeable "
                f"difference in {target.lower()} "
                f"across {factor.lower()} groups. "
                f"The group '{highest['factor']}' "
                f"has the highest observed rate. "
                f"This area should be investigated "
                f"using business context and "
                f"additional supporting data."
            )

        elif status == "🟢 No Evidence Detected":

            analysis_text = (

                f"The available data does not show "
                f"a large difference in {target.lower()} "
                f"across the available "
                f"{factor.lower()} groups under "
                f"the current screening rule."
            )

        else:

            analysis_text = (

                f"The available data contains "
                f"{factor.lower()} groups that are "
                f"too small for a reliable comparison. "
                f"Additional records are required."
            )

        recommendation["Detailed Analysis"] = (
            analysis_text
        )

        recommendation["Factor Analysis"] = "\n".join(

            [

                f"{row['factor']} → "
                f"{row['rate']:.1f}% "
                f"({int(row['count']):,} records)"

                for _, row
                in grouped
                .sort_values(
                    "rate",
                    ascending=False
                )
                .head(10)
                .iterrows()

            ]

        )

        recommendation["Business Impact"] = (

            f"Differences in {target.lower()} "
            f"across {factor.lower()} groups "
            "can help the business identify "
            "areas that require closer review."
        )

        if status == "🔴 Problem Detected":

            recommendation["Corrective Measures"] = [

                f"Review {target.lower()} by {factor}.",

                f"Identify the groups with the highest "
                f"observed {target.lower()} rate.",

                "Investigate the operational reasons behind the pattern.",

                "Check whether the pattern continues across time periods.",

                "Discuss the findings with the relevant business team.",

                "Create a monitoring process for the affected groups."

            ]

        elif status == "🟢 No Evidence Detected":

            recommendation["Corrective Measures"] = [

                f"Continue monitoring {target.lower()} by {factor}.",

                "Maintain consistent data collection.",

                "Check the same factor in future reporting periods."

            ]

        else:

            recommendation["Corrective Measures"] = [

                f"Collect more records for {factor}.",

                "Validate the group definitions.",

                "Repeat the comparison after sufficient data is available."

            ]

        recommendation["How to Improve Working"] = (

            f"Introduce a regular {target} review "
            f"by {factor}. Compare the results "
            "periodically instead of reviewing "
            "the metric only after a problem occurs."
        )

        recommendation["Workflow"] = (

            f"Upload Data\n"
            f"      ↓\n"
            f"Calculate {target}\n"
            f"      ↓\n"
            f"Segment by {factor}\n"
            f"      ↓\n"
            f"Compare Groups\n"
            f"      ↓\n"
            f"Identify High-Risk / Low-Performance Areas\n"
            f"      ↓\n"
            f"Business Investigation\n"
            f"      ↓\n"
            f"Corrective Action\n"
            f"      ↓\n"
            f"Monitor Result"
        )

        recommendation["Development Opportunity"] = (

            f"Develop a {industry} monitoring dashboard "
            f"containing {target.lower()} trends, "
            f"{factor.lower()} comparisons and "
            "automatic alerts for unusual changes."
        )

        recommendation["Expected Outcome"] = (

            f"Better visibility into {factor.lower()} "
            f"groups requiring investigation and "
            f"targeted business action."
        )

        recommendation["Limitation"] = (

            f"The observed relationship between "
            f"{factor} and {target} is an association "
            "in the available data. It does not prove "
            "that the factor caused the outcome."
        )

        recommendations.append(
            recommendation
        )

    return recommendations


# ============================================================
# NUMERIC KPI / FACTOR ANALYSIS
# ============================================================

def analyze_metric_factors(
    df,
    metric,
    industry
):

    recommendations = []

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    for factor in categorical_columns:

        if factor == metric:
            continue

        unique_count = (
            df[factor]
            .nunique(
                dropna=True
            )
        )

        if unique_count < 2:
            continue

        if unique_count > 15:
            continue

        temp = pd.DataFrame({

            "factor":
                df[factor]
                .fillna("Unknown")
                .astype(str),

            "metric":
                pd.to_numeric(
                    df[metric],
                    errors="coerce"
                )

        }).dropna(
            subset=["metric"]
        )

        grouped = (

            temp
            .groupby("factor")["metric"]
            .agg(
                ["count", "mean", "median"]
            )
            .reset_index()

        )

        if len(grouped) < 2:
            continue

        highest = (
            grouped
            .sort_values(
                "mean",
                ascending=False
            )
            .iloc[0]
        )

        lowest = (
            grouped
            .sort_values(
                "mean",
                ascending=True
            )
            .iloc[0]
        )

        difference = (
            highest["mean"]
            -
            lowest["mean"]
        )

        percentage_difference = (

            abs(difference)
            /
            max(
                abs(lowest["mean"]),
                1e-9
            )
            *
            100

        )

        reliable = (

            grouped["count"].min()
            >=
            max(
                5,
                int(len(df) * 0.01)
            )

        )

        if not reliable:

            status = (
                "🟡 Needs Investigation"
            )

        elif percentage_difference >= 20:

            status = (
                "🔴 Problem Detected"
            )

        else:

            status = (
                "🟢 No Evidence Detected"
            )

        recommendation = create_recommendation(

            f"{industry} — {metric} Analysis",

            f"{metric} Performance by {factor}",

            status

        )

        recommendation["Evidence"] = (

            f"The highest average {metric} "
            f"is {format_number(highest['mean'])} "
            f"for {highest['factor']}. "

            f"The lowest average {metric} "
            f"is {format_number(lowest['mean'])} "
            f"for {lowest['factor']}."
        )

        recommendation["Detailed Analysis"] = (

            f"The dataset shows differences in "
            f"{metric.lower()} across "
            f"{factor.lower()} groups. "
            "The difference should be reviewed "
            "using transaction volume, business "
            "context and time period."
        )

        recommendation["Factor Analysis"] = "\n".join(

            [

                f"{row['factor']} → "
                f"Average {metric}: "
                f"{format_number(row['mean'])} "
                f"({int(row['count']):,} records)"

                for _, row
                in grouped
                .sort_values(
                    "mean",
                    ascending=False
                )
                .head(10)
                .iterrows()

            ]

        )

        recommendation["Business Impact"] = (

            f"Persistent differences in "
            f"{metric.lower()} can help identify "
            f"{factor.lower()} groups requiring "
            "operational review."
        )

        if status == "🔴 Problem Detected":

            recommendation["Corrective Measures"] = [

                f"Review {metric.lower()} by {factor}.",

                f"Investigate groups with the lowest "
                f"observed {metric.lower()}.",

                "Check whether the pattern exists across multiple periods.",

                "Review operational differences between groups.",

                "Identify practical improvement actions.",

                "Monitor the KPI after the changes."

            ]

        elif status == "🟢 No Evidence Detected":

            recommendation["Corrective Measures"] = [

                f"Continue monitoring {metric.lower()} by {factor}.",

                "Maintain consistent measurement.",

                "Review the KPI periodically."

            ]

        else:

            recommendation["Corrective Measures"] = [

                "Collect additional records.",

                "Validate the group definitions.",

                "Repeat the analysis after more data is available."

            ]

        recommendation["How to Improve Working"] = (

            f"Introduce a regular {metric} performance "
            f"review by {factor} so operational "
            "differences can be identified earlier."
        )

        recommendation["Workflow"] = (

            f"Upload Data\n"
            f"      ↓\n"
            f"Calculate {metric}\n"
            f"      ↓\n"
            f"Compare by {factor}\n"
            f"      ↓\n"
            f"Identify Performance Gap\n"
            f"      ↓\n"
            f"Investigate Cause\n"
            f"      ↓\n"
            f"Improve Process\n"
            f"      ↓\n"
            f"Monitor KPI"
        )

        recommendation["Development Opportunity"] = (

            f"Create a {metric} Performance Monitoring "
            f"Dashboard with {factor.lower()} drill-down, "
            "trend analysis and alerts."
        )

        recommendation["Expected Outcome"] = (

            f"Improved visibility into {metric.lower()} "
            f"performance across {factor.lower()} groups."
        )

        recommendation["Limitation"] = (

            f"Differences in average {metric.lower()} "
            "do not automatically explain why the "
            "difference exists or prove causation."
        )

        recommendations.append(
            recommendation
        )

    return recommendations


# ============================================================
# DOMAIN-SPECIFIC MISSING INFORMATION
# ============================================================

DOMAIN_REQUIREMENTS = {

    "HR": {

        "Compensation Analysis": [
            "MonthlyIncome",
            "Salary",
            "AnnualSalary",
            "PayGrade"
        ],

        "Workforce Analysis": [
            "Department",
            "JobRole",
            "EmployeeNumber"
        ]

    },

    "Retail": {

        "Sales Analysis": [
            "Sales",
            "Revenue",
            "Amount"
        ],

        "Inventory Analysis": [
            "Inventory",
            "Stock",
            "Quantity"
        ]

    },

    "Marketing": {

        "Campaign Analysis": [
            "Campaign",
            "CampaignID",
            "Channel"
        ],

        "Conversion Analysis": [
            "Conversion",
            "Converted",
            "Response"
        ]

    },

    "Finance": {

        "Risk Analysis": [
            "Default",
            "Fraud",
            "CreditScore",
            "Risk"
        ]

    },

    "Healthcare": {

        "Clinical Analysis": [
            "Diagnosis",
            "Disease",
            "Treatment"
        ],

        "Patient Analysis": [
            "PatientID",
            "Patient"
        ]

    },

    "Education": {

        "Student Analysis": [
            "StudentID",
            "Student"
        ],

        "Performance Analysis": [
            "Marks",
            "Score",
            "Grade",
            "Result"
        ]

    },

    "Telecom": {

        "Churn Analysis": [
            "Churn",
            "Churned"
        ]

    },

    "Manufacturing": {

        "Quality Analysis": [
            "Defect",
            "Defective",
            "Quality"
        ],

        "Production Analysis": [
            "Production",
            "Units",
            "Quantity"
        ]

    },

    "Supply Chain": {

        "Delivery Analysis": [
            "Delivery",
            "LeadTime",
            "Shipment"
        ],

        "Inventory Analysis": [
            "Inventory",
            "Stock",
            "Stockout"
        ]

    }

}


def check_domain_information(
    df,
    industry
):

    recommendations = []

    requirements = (
        DOMAIN_REQUIREMENTS
        .get(
            industry,
            {}
        )
    )

    for analysis_name, aliases in requirements.items():

        column = find_column(
            df,
            aliases
        )

        if column:

            continue

        recommendation = create_recommendation(

            f"{industry} — {analysis_name}",

            analysis_name,

            "🟡 Needs Investigation"

        )

        recommendation["Evidence"] = (

            f"The current dataset does not contain "
            f"a clear column required for {analysis_name.lower()}."
        )

        recommendation["Detailed Analysis"] = (

            f"The available columns are not sufficient "
            f"to complete a reliable {analysis_name.lower()}."
        )

        recommendation["Business Impact"] = (

            "The missing information limits the ability "
            "to evaluate this business area."
        )

        recommendation["Corrective Measures"] = [

            f"Obtain the required information for {analysis_name.lower()}.",

            "Validate the additional fields against the source system.",

            "Combine the information with the current dataset.",

            "Repeat the analysis after the required data becomes available."

        ]

        recommendation["Additional Data Required"] = aliases

        recommendation["How to Improve Working"] = (

            f"Include the required {analysis_name.lower()} "
            "fields in the standard data collection process."
        )

        recommendation["Development Opportunity"] = (

            f"Add automated {analysis_name} monitoring "
            "after the required information becomes available."
        )

        recommendation["Expected Outcome"] = (

            f"Enable evidence-based {analysis_name.lower()}."
        )

        recommendation["Limitation"] = (

            f"No reliable conclusion about "
            f"{analysis_name.lower()} can be made "
            "without the required fields."
        )

        recommendations.append(
            recommendation
        )

    return recommendations


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_recommendations(
    df,
    insights=None,
    research_result=None
):

    if df is None:

        return []

    if df.empty:

        return []

    industry = detect_industry(
        df
    )

    target = detect_target(
        df
    )

    metric = detect_metric(
        df
    )

    recommendations = []

    # --------------------------------------------------------
    # 1. DATA QUALITY
    # --------------------------------------------------------

    missing = check_missing_values(
        df
    )

    if missing:

        recommendations.append(
            missing
        )

    duplicates = check_duplicates(
        df
    )

    if duplicates:

        recommendations.append(
            duplicates
        )

    # --------------------------------------------------------
    # 2. DOMAIN-SPECIFIC DATA AVAILABILITY
    # --------------------------------------------------------

    recommendations.extend(

        check_domain_information(
            df,
            industry
        )

    )

    # --------------------------------------------------------
    # 3. TARGET-BASED ANALYSIS
    # --------------------------------------------------------

    if target:

        target_results = (
            analyze_target_factors(
                df,
                target,
                industry
            )
        )

        recommendations.extend(
            target_results
        )

    # --------------------------------------------------------
    # 4. KPI-BASED ANALYSIS
    # --------------------------------------------------------

    elif metric:

        metric_results = (
            analyze_metric_factors(
                df,
                metric,
                industry
            )
        )

        recommendations.extend(
            metric_results
        )

    # --------------------------------------------------------
    # 5. NO ANALYTICAL TARGET
    # --------------------------------------------------------

    if not target and not metric:

        recommendation = create_recommendation(

            f"{industry} — Dataset Analysis",

            "Insufficient Business KPI / Target",

            "🟡 Needs Investigation"

        )

        recommendation["Evidence"] = (

            "The uploaded dataset does not contain "
            "a clearly identifiable outcome or "
            "business KPI that can be automatically "
            "used for detailed performance analysis."
        )

        recommendation["Detailed Analysis"] = (

            "The platform can profile the available "
            "columns, but a reliable business recommendation "
            "requires a meaningful outcome or KPI."
        )

        recommendation["Corrective Measures"] = [

            "Identify the main business outcome.",

            "Define the primary KPI.",

            "Identify the important business dimensions.",

            "Provide the KPI definition to the analysis system."

        ]

        recommendation["Additional Data Required"] = [

            "Business KPI",

            "KPI definition",

            "Business outcome",

            "Relevant business dimensions"

        ]

        recommendation["Development Opportunity"] = (

            "Allow users to select the primary KPI "
            "before generating detailed recommendations."
        )

        recommendation["Expected Outcome"] = (

            "More focused and business-specific recommendations."
        )

        recommendations.append(
            recommendation
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATE RECOMMENDATIONS
    # --------------------------------------------------------

    unique = []

    seen = set()

    for recommendation in recommendations:

        key = (

            recommendation.get(
                "Business Area",
                ""
            ),

            recommendation.get(
                "Industry Problem",
                ""
            )

        )

        if key in seen:

            continue

        seen.add(
            key
        )

        unique.append(
            recommendation
        )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    def priority(item):

        status = item.get(
            "Status",
            ""
        )

        if "🔴" in status:

            return 100

        if "🟡" in status:

            return 60

        return 20

    unique.sort(
        key=priority,
        reverse=True
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return unique[:15]
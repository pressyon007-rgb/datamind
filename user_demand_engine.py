# modules/user_demand_engine.py

import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(df, candidates):
    """
    Find a dataset column using exact or partial matching.
    """

    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    # Exact match
    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    # Partial match
    for col in df.columns:
        col_norm = normalize_name(col)

        for candidate in candidates:
            candidate_norm = normalize_name(candidate)

            if (
                candidate_norm in col_norm
                or col_norm in candidate_norm
            ):
                return col

    return None


def numeric_columns(df):
    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(df):
    result = []

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        unique_count = df[col].nunique(
            dropna=True
        )

        if unique_count <= max(
            50,
            min(100, len(df) // 2)
        ):
            result.append(col)

    return result


# ============================================================
# USER INTENT DETECTION
# ============================================================

def detect_user_intent(request):

    text = str(request).strip().lower()

    prediction_words = [
        "predict",
        "prediction",
        "forecast",
        "future",
        "likely",
        "probability",
        "risk",
        "will",
        "expected",
        "next month",
        "next quarter",
        "next year"
    ]

    analysis_words = [
        "analyse",
        "analyze",
        "analysis",
        "factor",
        "factors",
        "contribute",
        "contributing",
        "reason",
        "reasons",
        "problem",
        "problems",
        "issue",
        "issues",
        "why",
        "affect",
        "affecting",
        "impact",
        "pattern",
        "patterns",
        "important",
        "driver",
        "drivers",
        "recommend",
        "recommendation",
        "improve",
        "improvement"
    ]

    prediction_requested = any(
        word in text
        for word in prediction_words
    )

    analysis_requested = any(
        word in text
        for word in analysis_words
    )

    if prediction_requested and analysis_requested:
        mode = "analysis_prediction"

    elif prediction_requested:
        mode = "prediction"

    elif analysis_requested:
        mode = "analysis"

    elif text:
        mode = "analysis"

    else:
        mode = "automatic"

    return {
        "mode": mode,
        "prediction_requested": prediction_requested,
        "analysis_requested": analysis_requested,
        "request": request
    }


# ============================================================
# TARGET DETECTION
# ============================================================

TARGET_ALIASES = {

    "attrition": [
        "attrition",
        "employee attrition",
        "employee exit",
        "employee turnover",
        "left employee"
    ],

    "churn": [
        "churn",
        "customer churn",
        "customer leaving",
        "customer retention"
    ],

    "sales": [
        "sales",
        "sale",
        "selling",
        "sales performance"
    ],

    "revenue": [
        "revenue",
        "business revenue"
    ],

    "response": [
        "response",
        "campaign response",
        "respond",
        "conversion"
    ],

    "profit": [
        "profit",
        "profitability",
        "margin"
    ],

    "defect": [
        "defect",
        "defects",
        "quality",
        "failure"
    ],

    "outcome": [
        "outcome",
        "patient outcome",
        "result"
    ],

    "score": [
        "score",
        "marks",
        "grade",
        "performance score"
    ]
}


def detect_target(df, request):

    text = str(request).lower()

    # First use the user's wording
    for concept, aliases in TARGET_ALIASES.items():

        if any(
            alias in text
            for alias in aliases
        ):

            column = find_column(
                df,
                aliases + [concept]
            )

            if column:
                return column

    # Otherwise look for common target names
    common_targets = [
        "Attrition",
        "Churn",
        "Response",
        "Sales",
        "Revenue",
        "Profit",
        "Defect",
        "Outcome",
        "Target",
        "Label",
        "Status",
        "Result"
    ]

    return find_column(
        df,
        common_targets
    )


def get_target(
    df,
    selected_target,
    request
):

    if (
        selected_target
        and selected_target != "Auto Detect"
        and selected_target in df.columns
    ):
        return selected_target

    return detect_target(
        df,
        request
    )


# ============================================================
# BINARY TARGET
# ============================================================

def get_binary_target(df, target_column):

    series = (
        df[target_column]
        .astype(str)
        .str.strip()
    )

    valid_values = [
        value
        for value in series.unique()
        if value.lower() not in [
            "nan",
            "none",
            ""
        ]
    ]

    if len(valid_values) != 2:
        return None, None

    positive_words = [
        "yes",
        "1",
        "true",
        "positive",
        "left",
        "churn",
        "bad",
        "fail",
        "failed",
        "defect",
        "default",
        "response",
        "responded"
    ]

    positive_value = None

    for value in valid_values:

        if str(value).lower() in positive_words:
            positive_value = value
            break

    if positive_value is None:
        positive_value = valid_values[-1]

    mask = series == positive_value

    return mask, positive_value


# ============================================================
# IMPORTANT FACTOR ANALYSIS
# ============================================================

def categorical_factor_analysis(
    df,
    factor,
    target
):

    target_mask, positive_value = get_binary_target(
        df,
        target
    )

    if target_mask is None:
        return None

    temp = pd.DataFrame({
        "factor": df[factor],
        "target": target_mask.astype(int)
    }).dropna()

    if len(temp) < 20:
        return None

    grouped = (
        temp
        .groupby("factor")["target"]
        .agg(["mean", "count"])
    )

    # Ignore very small groups
    minimum_group_size = max(
        5,
        int(len(temp) * 0.01)
    )

    grouped = grouped[
        grouped["count"] >= minimum_group_size
    ]

    if len(grouped) < 2:
        return None

    overall_rate = (
        temp["target"].mean() * 100
    )

    rates = grouped["mean"] * 100

    highest_group = rates.idxmax()
    lowest_group = rates.idxmin()

    highest_rate = rates.max()
    lowest_rate = rates.min()

    spread = highest_rate - lowest_rate

    # Data-driven priority score
    score = min(
        100,
        spread * 1.5
    )

    return {
        "factor": factor,
        "type": "categorical",
        "priority_score": round(score, 2),
        "overall_rate": round(
            overall_rate,
            2
        ),
        "highest_group": str(
            highest_group
        ),
        "highest_rate": round(
            highest_rate,
            2
        ),
        "lowest_group": str(
            lowest_group
        ),
        "lowest_rate": round(
            lowest_rate,
            2
        ),
        "spread": round(
            spread,
            2
        )
    }


def numeric_factor_analysis(
    df,
    factor,
    target
):

    target_mask, positive_value = get_binary_target(
        df,
        target
    )

    if target_mask is None:
        return None

    values = pd.to_numeric(
        df[factor],
        errors="coerce"
    )

    temp = pd.DataFrame({
        "value": values,
        "target": target_mask.astype(int)
    }).dropna()

    if len(temp) < 20:
        return None

    positive_values = temp.loc[
        temp["target"] == 1,
        "value"
    ]

    negative_values = temp.loc[
        temp["target"] == 0,
        "value"
    ]

    if (
        len(positive_values) < 5
        or len(negative_values) < 5
    ):
        return None

    positive_mean = positive_values.mean()
    negative_mean = negative_values.mean()

    std = temp["value"].std()

    if std == 0 or pd.isna(std):
        return None

    standardized_difference = (
        abs(
            positive_mean
            - negative_mean
        )
        / std
    )

    score = min(
        100,
        standardized_difference * 35
    )

    return {
        "factor": factor,
        "type": "numeric",
        "priority_score": round(
            score,
            2
        ),
        "positive_mean": round(
            positive_mean,
            2
        ),
        "negative_mean": round(
            negative_mean,
            2
        ),
        "difference": round(
            positive_mean
            - negative_mean,
            2
        )
    }


def analyze_important_factors(
    df,
    target,
    user_request="",
    maximum=10
):

    if not target:
        return []

    if target not in df.columns:
        return []

    target_series = (
        df[target]
        .astype(str)
        .str.strip()
    )

    if target_series.nunique(
        dropna=True
    ) != 2:
        return []

    results = []

    # -------------------------
    # Categorical factors
    # -------------------------

    for column in categorical_columns(df):

        if column == target:
            continue

        result = categorical_factor_analysis(
            df,
            column,
            target
        )

        if result:
            results.append(result)

    # -------------------------
    # Numeric factors
    # -------------------------

    for column in numeric_columns(df):

        if column == target:
            continue

        result = numeric_factor_analysis(
            df,
            column,
            target
        )

        if result:
            results.append(result)

    # -------------------------
    # User-request relevance
    # -------------------------

    request_text = str(
        user_request
    ).lower()

    for result in results:

        factor_text = (
            result["factor"]
            .lower()
            .replace("_", " ")
        )

        user_bonus = 0

        if (
            factor_text
            and factor_text in request_text
        ):
            user_bonus = 25

        result["user_relevance"] = user_bonus

        result["final_priority_score"] = min(
            100,
            result["priority_score"]
            + user_bonus
        )

    # Highest priority first
    results.sort(
        key=lambda x:
            x["final_priority_score"],
        reverse=True
    )

    # Assign priority labels
    for rank, result in enumerate(
        results[:maximum],
        start=1
    ):

        result["rank"] = rank

        score = result[
            "final_priority_score"
        ]

        if score >= 60:
            result["priority"] = "Very High"

        elif score >= 35:
            result["priority"] = "High"

        elif score >= 15:
            result["priority"] = "Moderate"

        else:
            result["priority"] = "Low"

    return results[:maximum]


# ============================================================
# COMMON DATA PROBLEMS
# ============================================================

def check_data_quality(df):

    results = []

    # -------------------------
    # Missing values
    # -------------------------

    missing = df.isna().sum()

    total_missing = int(
        missing.sum()
    )

    if total_missing > 0:

        worst = (
            missing[
                missing > 0
            ]
            .sort_values(
                ascending=False
            )
            .head(5)
        )

        evidence = "\n".join(
            [
                f"- {column}: "
                f"{int(count):,} missing "
                f"({count / max(len(df),1) * 100:.1f}%)"
                for column, count
                in worst.items()
            ]
        )

        results.append({
            "Problem":
                "Missing / Incomplete Data",

            "Status":
                "🔴 Problem Detected",

            "Evidence":
                evidence,

            "Detailed Analysis":
                "Important fields contain missing values. "
                "Missing information may reduce the reliability "
                "of downstream business analysis.",

            "Corrective Measures":
                "Validate the source system, identify why "
                "values are missing, complete validated records, "
                "and make critical fields mandatory.",

            "Activities / Actions":
                "Create a missing-data exception list, "
                "assign data owners and monitor missing-rate trends.",

            "How to Improve Working":
                "Introduce mandatory-field validation and "
                "automated data-quality monitoring.",

            "Development Opportunity":
                "Automated Data Quality Dashboard.",

            "Additional Data Required":
                "Source-system validation rules and data ownership."
        })

    else:

        results.append({
            "Problem":
                "Missing / Incomplete Data",

            "Status":
                "🟢 No Problem Detected",

            "Evidence":
                "No missing values were detected.",

            "Detailed Analysis":
                "The uploaded dataset contains values "
                "for all analysed cells.",

            "Corrective Measures":
                "No corrective action is required for missingness.",

            "Activities / Actions":
                "Continue periodic completeness checks.",

            "How to Improve Working":
                "Maintain automated data-quality validation.",

            "Development Opportunity":
                "Data-quality monitoring system.",

            "Additional Data Required":
                ""
        })

    # -------------------------
    # Duplicate records
    # -------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates > 0:

        results.append({
            "Problem":
                "Duplicate Records",

            "Status":
                "🔴 Problem Detected",

            "Evidence":
                f"{duplicates:,} exact duplicate rows detected.",

            "Detailed Analysis":
                "Duplicate records may inflate "
                "business counts and KPIs.",

            "Corrective Measures":
                "Investigate duplicate records, identify "
                "the correct business key and remove "
                "confirmed duplicates.",

            "Activities / Actions":
                "Review duplicate records and identify "
                "the source of duplication.",

            "How to Improve Working":
                "Introduce unique-key validation.",

            "Development Opportunity":
                "Automated duplicate-record monitoring.",

            "Additional Data Required":
                "Business key definition."
        })

    else:

        results.append({
            "Problem":
                "Duplicate Records",

            "Status":
                "🟢 No Problem Detected",

            "Evidence":
                "No exact duplicate rows were detected.",

            "Detailed Analysis":
                "The uploaded data does not show "
                "an exact duplicate-record problem.",

            "Corrective Measures":
                "No immediate corrective action is required.",

            "Activities / Actions":
                "Continue duplicate checks for future uploads.",

            "How to Improve Working":
                "Maintain duplicate validation.",

            "Development Opportunity":
                "Automated duplicate monitoring.",

            "Additional Data Required":
                ""
        })

    return results


# ============================================================
# HR COMMON PROBLEMS
# ============================================================

def check_hr_problems(df):

    results = []

    attrition = find_column(
        df,
        ["Attrition"]
    )

    overtime = find_column(
        df,
        ["OverTime", "Overtime"]
    )

    satisfaction = find_column(
        df,
        [
            "JobSatisfaction",
            "Job Satisfaction"
        ]
    )

    worklife = find_column(
        df,
        [
            "WorkLifeBalance",
            "Work Life Balance"
        ]
    )

    # ========================================================
    # ATTRITION
    # ========================================================

    if attrition:

        series = (
            df[attrition]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        positive = series.isin([
            "yes",
            "1",
            "true",
            "left"
        ])

        valid = series.notna()

        if valid.sum() > 0:

            rate = (
                positive[valid]
                .mean()
                * 100
            )

            if rate > 10:

                results.append({
                    "Problem":
                        "Employee Attrition",

                    "Status":
                        "🔴 Problem Detected",

                    "Evidence":
                        f"Employees analysed: "
                        f"{valid.sum():,}\n\n"
                        f"Employees who left: "
                        f"{int(positive[valid].sum()):,}\n\n"
                        f"Attrition rate: "
                        f"{rate:.1f}%",

                    "Detailed Analysis":
                        "Employee exits are present at a measurable "
                        "level. The next step is to identify which "
                        "departments, roles and employee factors "
                        "show the largest differences.",

                    "Corrective Measures":
                        "Review high-attrition departments and roles. "
                        "Investigate overtime, satisfaction, "
                        "work-life balance, career progression, "
                        "business travel and compensation.",

                    "Activities / Actions":
                        "Conduct department-level attrition review, "
                        "employee feedback, workload review and "
                        "retention planning.",

                    "How to Improve Working":
                        "Create a recurring employee-retention "
                        "monitoring process.",

                    "Development Opportunity":
                        "Employee Retention Monitoring Dashboard.",

                    "Additional Data Required":
                        "Exit reasons, engagement survey, workload "
                        "hours and manager feedback."
                })

            else:

                results.append({
                    "Problem":
                        "Employee Attrition",

                    "Status":
                        "🟢 No Strong Problem Detected",

                    "Evidence":
                        f"Observed attrition rate: "
                        f"{rate:.1f}%",

                    "Detailed Analysis":
                        "The current screening does not identify "
                        "a high overall attrition level.",

                    "Corrective Measures":
                        "No specific corrective intervention "
                        "is recommended from this check alone.",

                    "Activities / Actions":
                        "Continue periodic attrition monitoring.",

                    "How to Improve Working":
                        "Maintain workforce monitoring.",

                    "Development Opportunity":
                        "Automated retention monitoring.",

                    "Additional Data Required":
                        ""
                })

    # ========================================================
    # OVERTIME
    # ========================================================

    if attrition and overtime:

        attrition_series = (
            df[attrition]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        positive = attrition_series.isin([
            "yes",
            "1",
            "true",
            "left"
        ])

        overtime_series = (
            df[overtime]
            .astype(str)
            .str.strip()
        )

        rates = {}

        for group in overtime_series.dropna().unique():

            mask = (
                overtime_series
                == group
            )

            if mask.sum() >= 5:

                rates[str(group)] = (
                    positive[mask]
                    .mean()
                    * 100
                )

        if len(rates) >= 2:

            spread = (
                max(rates.values())
                - min(rates.values())
            )

            if spread >= 5:

                results.append({
                    "Problem":
                        "Overtime / Workload Concern",

                    "Status":
                        "🔴 Problem Detected",

                    "Evidence":
                        "\n".join(
                            [
                                f"{k}: {v:.1f}% attrition"
                                for k, v
                                in rates.items()
                            ]
                        )
                        + f"\n\nDifference: "
                        f"{spread:.1f} percentage points.",

                    "Detailed Analysis":
                        "The dataset shows a meaningful difference "
                        "in attrition across overtime groups. "
                        "Overtime should therefore receive higher "
                        "priority in workforce investigation.",

                    "Corrective Measures":
                        "Review workload distribution, staffing, "
                        "overtime concentration, shift planning "
                        "and work-life balance.",

                    "Activities / Actions":
                        "Identify departments with excessive overtime, "
                        "review staffing requirements and discuss "
                        "workload with managers.",

                    "How to Improve Working":
                        "Create a recurring workload and overtime "
                        "review process.",

                    "Development Opportunity":
                        "Employee Workload Monitoring Dashboard.",

                    "Additional Data Required":
                        "Actual overtime hours and workload volume."
                })

            else:

                results.append({
                    "Problem":
                        "Overtime / Workload Concern",

                    "Status":
                        "🟢 No Strong Problem Detected",

                    "Evidence":
                        "Attrition differences between overtime "
                        "groups are below the current screening threshold.",

                    "Detailed Analysis":
                        "The current dataset does not show "
                        "a strong overtime-related attrition pattern.",

                    "Corrective Measures":
                        "No targeted overtime intervention is "
                        "recommended based on this dataset alone.",

                    "Activities / Actions":
                        "Continue workload monitoring.",

                    "How to Improve Working":
                        "Maintain periodic workload review.",

                    "Development Opportunity":
                        "Workload monitoring dashboard.",

                    "Additional Data Required":
                        ""
                })

    # ========================================================
    # JOB SATISFACTION
    # ========================================================

    if attrition and satisfaction:

        attrition_series = (
            df[attrition]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        positive = attrition_series.isin([
            "yes",
            "1",
            "true",
            "left"
        ])

        satisfaction_values = pd.to_numeric(
            df[satisfaction],
            errors="coerce"
        )

        temp = pd.DataFrame({
            "satisfaction":
                satisfaction_values,

            "attrition":
                positive
        }).dropna()

        if (
            len(temp) >= 20
            and temp["satisfaction"].nunique() >= 2
        ):

            rates = (
                temp
                .groupby("satisfaction")
                ["attrition"]
                .mean()
                * 100
            )

            spread = (
                rates.max()
                - rates.min()
            )

            if spread >= 5:

                results.append({
                    "Problem":
                        "Job Satisfaction Concern",

                    "Status":
                        "🔴 Problem Detected",

                    "Evidence":
                        "\n".join(
                            [
                                f"Level {level}: "
                                f"{rate:.1f}% attrition"
                                for level, rate
                                in rates.items()
                            ]
                        )
                        + f"\n\nDifference: "
                        f"{spread:.1f} percentage points.",

                    "Detailed Analysis":
                        "Attrition varies across satisfaction levels. "
                        "This makes employee experience an important "
                        "area for HR investigation.",

                    "Corrective Measures":
                        "Review employee feedback, workload, "
                        "recognition, manager support and "
                        "career development.",

                    "Activities / Actions":
                        "Conduct structured employee feedback "
                        "and create targeted improvement plans.",

                    "How to Improve Working":
                        "Introduce periodic employee experience "
                        "monitoring.",

                    "Development Opportunity":
                        "Employee Experience Dashboard.",

                    "Additional Data Required":
                        "Engagement survey and employee feedback."
                })

            else:

                results.append({
                    "Problem":
                        "Job Satisfaction Concern",

                    "Status":
                        "🟢 No Strong Problem Detected",

                    "Evidence":
                        "Attrition differences across satisfaction "
                        "levels are below the current threshold.",

                    "Detailed Analysis":
                        "No strong satisfaction-related "
                        "attrition pattern was detected.",

                    "Corrective Measures":
                        "No targeted corrective action is "
                        "recommended from this check alone.",

                    "Activities / Actions":
                        "Continue employee feedback monitoring.",

                    "How to Improve Working":
                        "Maintain employee-experience reviews.",

                    "Development Opportunity":
                        "Employee satisfaction monitoring.",

                    "Additional Data Required":
                        ""
                })

    # ========================================================
    # WORK-LIFE BALANCE
    # ========================================================

    if attrition and worklife:

        attrition_series = (
            df[attrition]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        positive = attrition_series.isin([
            "yes",
            "1",
            "true",
            "left"
        ])

        worklife_values = pd.to_numeric(
            df[worklife],
            errors="coerce"
        )

        temp = pd.DataFrame({
            "worklife":
                worklife_values,

            "attrition":
                positive
        }).dropna()

        if (
            len(temp) >= 20
            and temp["worklife"].nunique() >= 2
        ):

            rates = (
                temp
                .groupby("worklife")
                ["attrition"]
                .mean()
                * 100
            )

            spread = (
                rates.max()
                - rates.min()
            )

            if spread >= 5:

                results.append({
                    "Problem":
                        "Work-Life Balance Concern",

                    "Status":
                        "🔴 Problem Detected",

                    "Evidence":
                        "\n".join(
                            [
                                f"Level {level}: "
                                f"{rate:.1f}% attrition"
                                for level, rate
                                in rates.items()
                            ]
                        ),

                    "Detailed Analysis":
                        "The available data shows differences "
                        "in attrition across work-life-balance levels.",

                    "Corrective Measures":
                        "Review workload, overtime, travel, "
                        "staffing and employee support.",

                    "Activities / Actions":
                        "Identify affected employee groups "
                        "and review their working conditions.",

                    "How to Improve Working":
                        "Create periodic work-life-balance reviews.",

                    "Development Opportunity":
                        "Employee Wellbeing Monitoring.",

                    "Additional Data Required":
                        "Working hours and employee feedback."
                })

            else:

                results.append({
                    "Problem":
                        "Work-Life Balance Concern",

                    "Status":
                        "🟢 No Strong Problem Detected",

                    "Evidence":
                        "No strong work-life-balance "
                        "attrition pattern was detected.",

                    "Detailed Analysis":
                        "The available data does not show "
                        "a strong difference between groups.",

                    "Corrective Measures":
                        "Continue monitoring.",

                    "Activities / Actions":
                        "Periodic employee experience review.",

                    "How to Improve Working":
                        "Maintain existing monitoring.",

                    "Development Opportunity":
                        "Employee wellbeing dashboard.",

                    "Additional Data Required":
                        ""
                })

    return results


# ============================================================
# USER PREDICTION
# ============================================================

def run_prediction(df, target_column):

    if not target_column:
        return {
            "success": False,
            "message":
                "No prediction target was identified."
        }

    if target_column not in df.columns:
        return {
            "success": False,
            "message":
                "Selected target does not exist."
        }

    # Import only when prediction is requested
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        StandardScaler
    )

    target = (
        df[target_column]
        .astype(str)
        .str.strip()
    )

    values = [
        value
        for value in target.unique()
        if value.lower()
        not in [
            "nan",
            "none",
            ""
        ]
    ]

    if len(values) != 2:

        return {
            "success": False,
            "message":
                f"Prediction requires a binary target. "
                f"'{target_column}' has "
                f"{len(values)} usable classes."
        }

    # Find positive class
    positive_words = [
        "yes",
        "1",
        "true",
        "left",
        "churn",
        "positive",
        "fail",
        "failed",
        "defect",
        "default",
        "response"
    ]

    positive_class = None

    for value in values:

        if str(value).lower() in positive_words:
            positive_class = value
            break

    if positive_class is None:
        positive_class = values[-1]

    negative_class = [
        value
        for value in values
        if value != positive_class
    ][0]

    y = (
        target
        .map({
            negative_class: 0,
            positive_class: 1
        })
    )

    valid = y.notna()

    X = df.loc[
        valid
    ].drop(
        columns=[target_column]
    )

    y = y.loc[
        valid
    ].astype(int)

    # Remove obvious identifiers
    remove_columns = []

    for column in X.columns:

        unique_ratio = (
            X[column]
            .nunique(dropna=True)
            / max(len(X), 1)
        )

        normalized = normalize_name(
            column
        )

        if normalized in [
            "employeeid",
            "employeenumber",
            "customerid",
            "transactionid",
            "patientid",
            "studentid",
            "id"
        ]:
            remove_columns.append(
                column
            )

        elif (
            unique_ratio > 0.95
            and not pd.api.types.is_numeric_dtype(
                X[column]
            )
        ):
            remove_columns.append(
                column
            )

    X = X.drop(
        columns=remove_columns,
        errors="ignore"
    )

    numeric = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = [
        column
        for column in X.columns
        if column not in numeric
    ]

    if not numeric and not categorical:

        return {
            "success": False,
            "message":
                "No usable predictor columns were found."
        }

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical
        )
    ])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X,
        y
    )

    probabilities = (
        pipeline
        .predict_proba(X)
    )

    predictions = (
        pipeline
        .predict(X)
    )

    positive_probability = (
        probabilities[:, 1]
        * 100
    )

    result_df = df.loc[
        X.index
    ].copy()

    result_df[
        "Predicted Class"
    ] = [
        positive_class
        if prediction == 1
        else negative_class
        for prediction
        in predictions
    ]

    result_df[
        "Prediction Probability %"
    ] = np.round(
        positive_probability,
        2
    )

    result_df[
        "Prediction Risk"
    ] = pd.cut(
        positive_probability,
        bins=[
            -np.inf,
            33,
            66,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    ).astype(str)

    result_df = result_df.sort_values(
        "Prediction Probability %",
        ascending=False
    )

    return {
        "success": True,
        "target": target_column,
        "positive_class": positive_class,
        "negative_class": negative_class,
        "predictions": result_df,
        "model": pipeline
    }


# ============================================================
# MAIN USER-DEMAND ENGINE
# ============================================================

def run_user_demand_analysis(
    df,
    request,
    selected_target="Auto Detect"
):

    intent = detect_user_intent(
        request
    )

    target = get_target(
        df,
        selected_target,
        request
    )

    result = {
        "intent": intent,
        "target": target,
        "factors": [],
        "common_problems": [],
        "prediction": None
    }

    # Always perform basic data-quality analysis
    result[
        "common_problems"
    ] = check_data_quality(df)

    # Detect HR dataset
    hr_columns = [
        "Attrition",
        "JobRole",
        "OverTime",
        "JobSatisfaction",
        "EmployeeNumber"
    ]

    hr_detected = any(
        find_column(
            df,
            [column]
        )
        for column in hr_columns
    )

    if hr_detected:

        result[
            "common_problems"
        ].extend(
            check_hr_problems(df)
        )

    # User requested analysis
    if (
        target
        and intent["mode"]
        in [
            "analysis",
            "analysis_prediction"
        ]
    ):

        result[
            "factors"
        ] = analyze_important_factors(
            df,
            target,
            request,
            maximum=10
        )

    # User requested prediction
    if (
        target
        and intent["mode"]
        in [
            "prediction",
            "analysis_prediction"
        ]
    ):

        result[
            "prediction"
        ] = run_prediction(
            df,
            target
        )

    return result


# ============================================================
# STREAMLIT UI
# ============================================================

def render_user_demand_section(
    st,
    df
):

    st.markdown(
        "## 🎯 User Analysis / Prediction Request"
    )

    st.caption(
        "Describe what you want to analyse or predict. "
        "Your request will receive priority in the analysis."
    )

    # --------------------------------------------------------
    # USER REQUEST BOX
    # --------------------------------------------------------

    request = st.text_area(
        "What would you like to analyse or predict?",
        placeholder=(
            "Example: Find the most important factors "
            "contributing to employee attrition and "
            "predict which employees have higher attrition risk."
        ),
        height=120,
        key="user_analysis_request"
    )

    # --------------------------------------------------------
    # TARGET SELECTOR
    # --------------------------------------------------------

    target_options = [
        "Auto Detect"
    ] + df.columns.tolist()

    selected_target = st.selectbox(
        "🎯 Target / Dimension (Optional)",
        target_options,
        key="user_target_dimension",
        help=(
            "Leave Auto Detect selected if you want the "
            "system to identify the target from your request."
        )
    )

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔎 Analyse / Predict",
        type="primary",
        use_container_width=True,
        key="run_user_analysis_prediction"
    ):

        if not request.strip():

            st.warning(
                "Please enter what you want to analyse or predict."
            )

            return

        with st.spinner(
            "Understanding your request and analysing the dataset..."
        ):

            result = run_user_demand_analysis(
                df,
                request,
                selected_target
            )

        st.session_state[
            "user_demand_result"
        ] = result

    # --------------------------------------------------------
    # GET STORED RESULT
    # --------------------------------------------------------

    result = st.session_state.get(
        "user_demand_result"
    )

    if not result:

        st.info(
            "Example: "
            "Find the most important factors contributing "
            "to employee attrition."
        )

        return

    # --------------------------------------------------------
    # REQUEST SUMMARY
    # --------------------------------------------------------

    intent = result[
        "intent"
    ]

    target = result[
        "target"
    ]

    st.markdown("---")

    st.markdown(
        "## 🎯 User-Requested Analysis"
    )

    st.info(
        f"**Your request:** "
        f"{intent['request']}"
    )

    mode_names = {
        "analysis":
            "📊 Analysis",

        "prediction":
            "🔮 Prediction",

        "analysis_prediction":
            "📊🔮 Analysis + Prediction",

        "automatic":
            "🤖 Automatic Analysis"
    }

    st.write(
        f"**Detected Mode:** "
        f"{mode_names.get(intent['mode'])}"
    )

    if target:

        st.write(
            f"**Target / Dimension:** "
            f"`{target}`"
        )

    else:

        st.warning(
            "No suitable target/dimension was detected. "
            "Select one manually if prediction is required."
        )

    # ========================================================
    # IMPORTANT FACTORS
    # ========================================================

    factors = result[
        "factors"
    ]

    if factors:

        st.markdown(
            "### 🔥 Highest-Priority Factors"
        )

        factor_table = []

        for factor in factors:

            factor_table.append({
                "Priority":
                    factor["rank"],

                "Factor":
                    factor["factor"],

                "Importance":
                    factor["priority"],

                "Priority Score":
                    factor[
                        "final_priority_score"
                    ],

                "Analysis Type":
                    factor["type"]
            })

        st.dataframe(
            pd.DataFrame(
                factor_table
            ),
            use_container_width=True,
            hide_index=True
        )

        # Detailed top factors
        for factor in factors[:5]:

            with st.expander(
                f"{factor['rank']}. "
                f"{factor['factor']} "
                f"— {factor['priority']}"
            ):

                if (
                    factor["type"]
                    == "categorical"
                ):

                    st.markdown(
                        f"""
**Highest group:** `{factor['highest_group']}`  
**Highest rate:** `{factor['highest_rate']:.1f}%`

**Lowest group:** `{factor['lowest_group']}`  
**Lowest rate:** `{factor['lowest_rate']:.1f}%`

**Overall rate:** `{factor['overall_rate']:.1f}%`

**Difference:** `{factor['spread']:.1f}` percentage points
"""
                    )

                else:

                    st.markdown(
                        f"""
**Positive-class mean:** `{factor['positive_mean']:.2f}`

**Other-class mean:** `{factor['negative_mean']:.2f}`

**Difference:** `{factor['difference']:.2f}`
"""
                    )

                st.caption(
                    "Important: factor priority represents "
                    "an observed pattern in the available data. "
                    "It does not prove causation."
                )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = result[
        "prediction"
    ]

    if prediction:

        st.markdown("---")

        st.markdown(
            "## 🔮 User-Requested Prediction"
        )

        if prediction.get(
            "success"
        ):

            st.success(
                f"Prediction completed for "
                f"`{prediction['target']}`."
            )

            st.write(
                f"**Predicted positive class:** "
                f"{prediction['positive_class']}"
            )

            st.dataframe(
                prediction[
                    "predictions"
                ].head(100),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Prediction probability is a model-generated "
                "decision-support signal and should not be "
                "treated as certainty or proof of causation."
            )

        else:

            st.warning(
                prediction.get(
                    "message",
                    "Prediction could not be completed."
                )
            )

    # ========================================================
    # COMMON PROBLEMS
    # ========================================================

    st.markdown("---")

    st.markdown(
        "## 📊 Common Dataset Problems"
    )

    for problem in result[
        "common_problems"
    ]:

        status = problem[
            "Status"
        ]

        title = problem[
            "Problem"
        ]

        with st.expander(
            f"{status} — {title}"
        ):

            st.markdown(
                "### 📊 Data Evidence"
            )

            st.write(
                problem[
                    "Evidence"
                ]
            )

            st.markdown(
                "### 🔎 Detailed Analysis"
            )

            st.write(
                problem[
                    "Detailed Analysis"
                ]
            )

            st.markdown(
                "### 🛠️ Corrective Measures"
            )

            st.write(
                problem[
                    "Corrective Measures"
                ]
            )

            st.markdown(
                "### 📋 Activities / Actions"
            )

            st.write(
                problem[
                    "Activities / Actions"
                ]
            )

            st.markdown(
                "### 🔧 How to Improve Current Working"
            )

            st.write(
                problem[
                    "How to Improve Working"
                ]
            )

            st.markdown(
                "### 🚀 Development Opportunity"
            )

            st.write(
                problem[
                    "Development Opportunity"
                ]
            )

            additional = problem.get(
                "Additional Data Required",
                ""
            )

            if additional:

                st.markdown(
                    "### 📥 Additional Data Required"
                )

                st.write(
                    additional
                )
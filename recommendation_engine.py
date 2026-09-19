"""
recommendation_engine.py - Evidence-Based Decision Support Engine
"""

def generate_recommendations(df, quality_info, relationships, domain_info):
    """
    Generates structured, evidence-based recommendations derived directly 
    from the uploaded dataframe. Handles positive findings and problems.
    """
    recommendations = []
    domain = domain_info["domain"]

    # 1. DATA QUALITY & INTEGRITY BLOCK
    if quality_info["is_perfect_quality"]:
        recommendations.append({
            "business_area": f"{domain} Data Governance & Integrity",
            "chart_outcome": f"Complete data profile detected across all {quality_info['total_rows']} records and {quality_info['total_cols']} fields with zero missing values or duplicate rows.",
            "what_this_means": "The dataset is structurally complete for the checked parameters, indicating strong initial collection and storage practices.",
            "limitation": "Structural completeness does not verify that recorded numerical measurements or entries are free from clinical or operational recording bias.",
            "analyst_recommendation": "Maintain existing collection standards and establish automated validation schemas for future intake pipelines.",
            "action_development": "Implement real-time ingestion pipelines with built-in schema checks to preserve data cleanliness."
        })
    else:
        missing_count = quality_info["total_missing"]
        dup_count = quality_info["duplicate_rows"]
        recommendations.append({
            "business_area": f"{domain} Data Remediation & Hygiene",
            "chart_outcome": f"Detected {missing_count} total null values across {len(quality_info['cols_with_missing'])} columns and {dup_count} duplicate records.",
            "what_this_means": "Incomplete records or duplicate observations can distort downstream predictive models and statistical summaries.",
            "limitation": "Missing entries may occur systematically (Not at Random) or completely at random, requiring distinct remediation approaches.",
            "analyst_recommendation": "Execute standard data cleaning protocols: remove duplicate rows and impute missing values using median/mode techniques.",
            "action_development": "Deploy pre-processing scripts prior to modeling or executive dashboard generation."
        })

    # 2. DOMAIN-SPECIFIC PATTERN / RELATIONSHIP BLOCK
    if relationships:
        top_rel = relationships[0]
        v1, v2, corr, strength = top_rel["var1"], top_rel["var2"], top_rel["correlation"], top_rel["strength"]

        recommendations.append({
            "business_area": f"{domain} Attribute Interaction Analysis",
            "chart_outcome": f"Statistical analysis reveals a {strength.lower()} relationship between '{v1}' and '{v2}' (Correlation: {corr}).",
            "what_this_means": f"Systematic increases or changes in '{v1}' strongly align with movements in '{v2}' within this specific dataset.",
            "limitation": "Correlation does not prove direct causality between variables; confounding factors may influence both metrics.",
            "analyst_recommendation": f"Review the operational linkage between '{v1}' and '{v2}' with domain experts to validate business logic.",
            "action_development": f"Build a dedicated visual tracking module for '{v1}' vs '{v2}' to monitor trends over time."
        })

    # 3. OUTLIER & DISPERSION BLOCK
    if quality_info["outliers"]:
        top_outlier_col = list(quality_info["outliers"].keys())[0]
        outlier_count = quality_info["outliers"][top_outlier_col]
        recommendations.append({
            "business_area": f"{domain} Variance & Outlier Audit",
            "chart_outcome": f"Identified {outlier_count} potential outlier values in the column '{top_outlier_col}'.",
            "what_this_means": "Unusual observations exist that deviate significantly from standard distribution ranges.",
            "limitation": "Outliers are not inherently erroneous; they may represent valid, critical extreme cases.",
            "analyst_recommendation": f"Inspect the extreme values in '{top_outlier_col}' to distinguish true data anomalies from key high-impact events.",
            "action_development": f"Establish IQR-based thresholds for automated outlier flagging in operational reporting."
        })

    return recommendations
"""
recommendation_engine.py - Evidence-Based Decision Support Engine
"""

def generate_recommendations(df, quality_info, relationships, domain_info):
    """
    Generates structured, evidence-based recommendations derived directly 
    from the uploaded dataframe. Handles non-dictionary inputs safely.
    """
    recommendations = []
    
    # Ensure domain_info is a valid dictionary
    if not isinstance(domain_info, dict):
        domain_info = {"domain": "General / Operations"}
    
    domain = domain_info.get("domain", "General / Operations")

    # Ensure quality_info is a valid dictionary
    if not isinstance(quality_info, dict):
        quality_info = {
            "is_perfect_quality": False,
            "total_rows": len(df) if df is not None and hasattr(df, "__len__") else 0,
            "total_cols": len(df.columns) if df is not None and hasattr(df, "columns") else 0,
            "total_missing": 0,
            "cols_with_missing": {},
            "duplicate_rows": 0,
            "outliers": {}
        }

    # 1. DATA QUALITY & INTEGRITY BLOCK
    if quality_info.get("is_perfect_quality", False):
        recommendations.append({
            "business_area": f"{domain} Data Governance & Integrity",
            "chart_outcome": f"Complete data profile detected across all {quality_info.get('total_rows', 0)} records and {quality_info.get('total_cols', 0)} fields with zero missing values or duplicate rows.",
            "what_this_means": "The dataset is structurally complete for the checked parameters, indicating strong initial collection and storage practices.",
            "limitation": "Structural completeness does not verify that recorded numerical measurements or entries are free from clinical or operational recording bias.",
            "analyst_recommendation": "Maintain existing collection standards and establish automated validation schemas for future intake pipelines.",
            "action_development": "Implement real-time ingestion pipelines with built-in schema checks to preserve data cleanliness."
        })
    else:
        missing_count = quality_info.get("total_missing", 0)
        dup_count = quality_info.get("duplicate_rows", 0)
        cols_missing = len(quality_info.get("cols_with_missing", {}))
        recommendations.append({
            "business_area": f"{domain} Data Remediation & Hygiene",
            "chart_outcome": f"Detected {missing_count} total null values across {cols_missing} columns and {dup_count} duplicate records.",
            "what_this_means": "Incomplete records or duplicate observations can distort downstream predictive models and statistical summaries.",
            "limitation": "Missing entries may occur systematically (Not at Random) or completely at random, requiring distinct remediation approaches.",
            "analyst_recommendation": "Execute standard data cleaning protocols: remove duplicate rows and impute missing values using median/mode techniques.",
            "action_development": "Deploy pre-processing scripts prior to modeling or executive dashboard generation."
        })

    # 2. DOMAIN-SPECIFIC PATTERN / RELATIONSHIP BLOCK
    if relationships and isinstance(relationships, list):
        top_rel = relationships[0]
        if isinstance(top_rel, dict):
            v1 = top_rel.get("var1", "Variable 1")
            v2 = top_rel.get("var2", "Variable 2")
            corr = top_rel.get("correlation", 0.0)
            strength = top_rel.get("strength", "Moderate")

            recommendations.append({
                "business_area": f"{domain} Attribute Interaction Analysis",
                "chart_outcome": f"Statistical analysis reveals a {str(strength).lower()} relationship between '{v1}' and '{v2}' (Correlation: {corr}).",
                "what_this_means": f"Systematic increases or changes in '{v1}' strongly align with movements in '{v2}' within this specific dataset.",
                "limitation": "Correlation does not prove direct causality between variables; confounding factors may influence both metrics.",
                "analyst_recommendation": f"Review the operational linkage between '{v1}' and '{v2}' with domain experts to validate business logic.",
                "action_development": f"Build a dedicated visual tracking module for '{v1}' vs '{v2}' to monitor trends over time."
            })

    # 3. OUTLIER & DISPERSION BLOCK
    outliers_dict = quality_info.get("outliers", {})
    if isinstance(outliers_dict, dict) and outliers_dict:
        top_outlier_col = list(outliers_dict.keys())[0]
        outlier_count = outliers_dict[top_outlier_col]
        recommendations.append({
            "business_area": f"{domain} Variance & Outlier Audit",
            "chart_outcome": f"Identified {outlier_count} potential outlier values in the column '{top_outlier_col}'.",
            "what_this_means": "Unusual observations exist that deviate significantly from standard distribution ranges.",
            "limitation": "Outliers are not inherently erroneous; they may represent valid, critical extreme cases.",
            "analyst_recommendation": f"Inspect the extreme values in '{top_outlier_col}' to distinguish true data anomalies from key high-impact events.",
            "action_development": f"Establish IQR-based thresholds for automated outlier flagging in operational reporting."
        })

    return recommendations

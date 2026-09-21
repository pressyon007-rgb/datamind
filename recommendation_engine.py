"""
recommendation_engine.py - Evidence-Based Decision Support Engine
Optimized for deep domain context, vectorized performance, and high-impact executive recommendations.
"""

import pandas as pd
import numpy as np


def generate_recommendations(df, quality_info, relationships, domain_info):
    """
    Generates structured, evidence-based recommendations derived directly 
    from the uploaded dataframe. Handles positive findings, quality defects,
    Pareto concentrations, statistical skewness, and attribute interactions.
    """
    recommendations = []
    
    if df is None or df.empty:
        return recommendations

    domain = domain_info.get("domain", "General / Operations")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    valid_cat_cols = [c for c in cat_cols if not c.lower().endswith('id') and df[c].nunique() <= 50]

    # -------------------------------------------------------------------------
    # 1. DATA QUALITY & GOVERNANCE BLOCK
    # -------------------------------------------------------------------------
    if quality_info.get("is_perfect_quality", False):
        recommendations.append({
            "business_area": f"{domain} Data Governance & Integrity",
            "chart_outcome": f"Complete data profile detected across all {quality_info.get('total_rows', len(df)):,} records and {quality_info.get('total_cols', len(df.columns))} fields with zero missing values or duplicate rows.",
            "what_this_means": "The dataset is structurally pristine, indicating robust data collection practices and reliable downstream metrics.",
            "limitation": "Structural completeness does not guarantee semantic validity or eliminate recording/reporting bias.",
            "analyst_recommendation": "Establish continuous automated validation schemas for real-time data ingestion pipelines.",
            "action_development": "Deploy automated CI/CD schema validation scripts prior to feeding data into live dashboards or ML models."
        })
    else:
        missing_count = quality_info.get("total_missing", 0)
        dup_count = quality_info.get("duplicate_rows", 0)
        cols_missing_cnt = len(quality_info.get("cols_with_missing", {}))
        
        recommendations.append({
            "business_area": f"{domain} Data Remediation & Hygiene",
            "chart_outcome": f"Detected {missing_count:,} missing values across {cols_missing_cnt} column(s) and {dup_count:,} duplicate record(s).",
            "what_this_means": "Incomplete or redundant records introduce bias, distort statistical aggregations, and decrease predictive accuracy.",
            "limitation": "Missing entries may occur systematically (Not at Random), requiring domain-specific handling rather than simple deletion.",
            "analyst_recommendation": "Deduplicate observations and apply median imputation for numerical features or mode imputation for categories.",
            "action_development": "Implement pre-processing cleanup routines prior to model training or executive dashboard generation."
        })

    # -------------------------------------------------------------------------
    # 2. CONCENTRATION & PARETO ANALYSIS BLOCK (80/20 Dynamic Rule)
    # -------------------------------------------------------------------------
    if valid_cat_cols and num_cols:
        primary_cat = valid_cat_cols[0]
        primary_num = num_cols[0]
        
        # Calculate market/category concentration
        cat_grouped = df.groupby(primary_cat)[primary_num].sum().sort_values(ascending=False)
        total_sum = cat_grouped.sum()
        
        if total_sum > 0:
            top_3_sum = cat_grouped.head(3).sum()
            top_3_pct = (top_3_sum / total_sum) * 100
            
            if top_3_pct >= 45.0:
                top_cats_str = ", ".join([str(x) for x in cat_grouped.head(3).index])
                recommendations.append({
                    "business_area": f"{domain} Portfolio Concentration ({primary_cat})",
                    "chart_outcome": f"Top 3 '{primary_cat}' categories ({top_cats_str}) drive {top_3_pct:.1f}% of total {primary_num}.",
                    "what_this_means": "High metric concentration indicates significant dependency on a small set of primary operational segments.",
                    "limitation": "Focusing purely on high-volume segments overlooks margin profitability or long-tail growth potential.",
                    "analyst_recommendation": f"Diversify strategy across lower-volume '{primary_cat}' segments while protecting core drivers.",
                    "action_development": f"Implement long-tail optimization campaigns targeting underperforming '{primary_cat}' categories."
                })

    # -------------------------------------------------------------------------
    # 3. INTERACTION & RELATIONSHIP BLOCK
    # -------------------------------------------------------------------------
    if relationships:
        top_rel = relationships[0]
        v1, v2, corr, strength = top_rel["var1"], top_rel["var2"], top_rel["correlation"], top_rel["strength"]

        recommendations.append({
            "business_area": f"{domain} Attribute Synergy ({v1} & {v2})",
            "chart_outcome": f"Statistical analysis reveals a {strength.lower()} relationship between '{v1}' and '{v2}' (Correlation: {corr:.2f}).",
            "what_this_means": f"Systematic movements in '{v1}' reliably co-occur with changes in '{v2}', offering strong predictive synergy.",
            "limitation": "Correlation does not establish direct causation; third-party confounding variables may drive both metrics.",
            "analyst_recommendation": f"Leverage '{v1}' as an early operational indicator to forecast expected trends in '{v2}'.",
            "action_development": f"Build a dedicated dual-metric tracking visual for '{v1}' vs '{v2}' on the executive dashboard."
        })

    # -------------------------------------------------------------------------
    # 4. OUTLIER & SKEWNESS DISPERSION BLOCK
    # -------------------------------------------------------------------------
    outliers_dict = quality_info.get("outliers", {})
    if outliers_dict:
        top_outlier_col = list(outliers_dict.keys())[0]
        outlier_count = outliers_dict[top_outlier_col]
        
        recommendations.append({
            "business_area": f"{domain} Variance & Extreme Value Audit",
            "chart_outcome": f"Identified {outlier_count:,} potential outlier values in field '{top_outlier_col}'.",
            "what_this_means": "Significant deviation from central metrics indicates high variance or extreme operational events.",
            "limitation": "Outliers are not inherently bad; they often represent genuine, high-impact business events or rare transactions.",
            "analyst_recommendation": f"Audit extreme observations in '{top_outlier_col}' to separate processing errors from high-value events.",
            "action_development": f"Establish IQR-based (Interquartile Range) dynamic thresholding for real-time anomaly alerts."
        })

    # -------------------------------------------------------------------------
    # 5. FAIL-SAFE FALLBACK BLOCK
    # -------------------------------------------------------------------------
    if not recommendations:
        recommendations.append({
            "business_area": f"{domain} Baseline Operations Optimization",
            "chart_outcome": f"Dataset across {len(df):,} rows exhibits balanced baseline distribution without extreme anomalies.",
            "what_this_means": "Current metrics suggest stable performance across evaluated operational dimensions.",
            "limitation": "Linear baseline evaluations may obscure non-linear trend shifts or emergent pattern changes.",
            "analyst_recommendation": "Shift focus toward time-series forecasting and scenario simulation models.",
            "action_development": "Deploy automated rolling-window analytics to capture subtle seasonal variations early."
        })

    return recommendations

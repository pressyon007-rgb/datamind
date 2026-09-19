"""
web_research.py - Domain Discovery & Contextual Analysis
Identifies the likely industry/domain of an uploaded dataset based on column signatures.
"""

DOMAIN_SIGNATURES = {
    "Healthcare / Clinical": [
        "diagnosis", "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "patient", "dosage", "treatment", "symptom",
        "hospital", "doctor", "blood_pressure", "cholesterol", "bmi", "tumour", "cancer"
    ],
    "Human Resources (HR)": [
        "attrition", "joblevel", "monthlyincome", "yearsatcompany", "overtime",
        "jobsatisfaction", "worklifebalance", "department", "employee", "salary",
        "performance_rating", "hire_date", "tenure"
    ],
    "Retail & E-Commerce": [
        "sales", "revenue", "order", "product", "quantity", "discount",
        "profit", "unit_price", "customer_id", "ship_mode", "category", "store", "order_id"
    ],
    "Finance & Banking": [
        "transaction", "account", "balance", "credit_score", "loan", "interest",
        "default", "principal", "fraud", "debt", "mortgage", "asset"
    ],
    "Marketing": [
        "campaign", "impressions", "clicks", "ctr", "cpc", "conversion",
        "lead", "acquisition", "channel", "ad_spend", "roi"
    ],
    "Operations & Supply Chain": [
        "inventory", "warehouse", "lead_time", "supplier", "shipment",
        "logistics", "stock_level", "fulfillment", "defect_rate"
    ]
}


def detect_domain(df):
    """
    Analyzes dataframe columns and infers the industry domain.
    Returns a dict with domain name, confidence score, and matched indicators.
    """
    columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    scores = {domain: 0 for domain in DOMAIN_SIGNATURES}
    matches = {domain: [] for domain in DOMAIN_SIGNATURES}

    for col in columns:
        for domain, keywords in DOMAIN_SIGNATURES.items():
            for kw in keywords:
                if kw in col:
                    scores[domain] += 1
                    matches[domain].append(col)

    best_domain = max(scores, key=scores.get)
    max_score = scores[best_domain]

    if max_score == 0:
        return {
            "domain": "General / Operations",
            "confidence": "Low",
            "matched_terms": []
        }

    confidence = "High" if max_score >= 3 else ("Medium" if max_score == 2 else "Low")
    return {
        "domain": best_domain,
        "confidence": confidence,
        "matched_terms": list(set(matches[best_domain]))
    }
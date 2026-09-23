import numpy as np
import pandas as pd


def detect_business_columns(df):

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    dates = df.select_dtypes(
        include=[
            "datetime",
            "datetimetz"
        ]
    ).columns.tolist()

    return {
        "numeric": numeric,
        "categorical": categorical,
        "dates": dates
    }


def choose_metric(df):

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric:
        return None

    priority = [
        "sales",
        "revenue",
        "profit",
        "amount",
        "income",
        "price",
        "quantity",
        "units",
        "cost"
    ]

    for keyword in priority:

        for column in numeric:

            if keyword in column.lower():

                return column

    return numeric[0]


def choose_category(df):

    categorical = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    if not categorical:
        return None

    priority = [
        "region",
        "city",
        "state",
        "category",
        "product",
        "customer",
        "segment",
        "department",
        "channel",
        "gender"
    ]

    for keyword in priority:

        for column in categorical:

            if keyword in column.lower():

                return column

    return categorical[0]


def generate_sheet_templates(df, sheet_count=4):

    info = detect_business_columns(df)

    numeric = info["numeric"]

    categorical = info["categorical"]

    dates = info["dates"]

    metric = choose_metric(df)

    category = choose_category(df)

    sheets = []

    # --------------------------------
    # SHEET 1
    # --------------------------------

    sheets.append({

        "name": "Executive Overview",

        "description":
            "High-level business performance",

        "charts": [

            {
                "title": "Overall Performance",
                "category": category,
                "metric": metric,
                "chart_type": "Bar"
            },

            {
                "title": "Performance Distribution",
                "category": category,
                "metric": metric,
                "chart_type": "Pie"
            },

            {
                "title": "Metric Distribution",
                "category": None,
                "metric": metric,
                "chart_type": "Histogram"
            },

            {
                "title": "Performance Trend",
                "category":
                    dates[0] if dates else category,
                "metric": metric,
                "chart_type": "Line"
            },

            {
                "title": "Performance Spread",
                "category": category,
                "metric": metric,
                "chart_type": "Box"
            }
        ]
    })

    # --------------------------------
    # SHEET 2
    # --------------------------------

    sheets.append({

        "name": "Sales & Performance",

        "description":
            "Revenue, sales and performance analysis",

        "charts": [

            {
                "title": "Top Categories",
                "category": category,
                "metric": metric,
                "chart_type": "Bar"
            },

            {
                "title": "Performance Area",
                "category": category,
                "metric": metric,
                "chart_type": "Area"
            },

            {
                "title": "Performance Trend",
                "category":
                    dates[0] if dates else category,
                "metric": metric,
                "chart_type": "Line"
            },

            {
                "title": "Metric Distribution",
                "category": None,
                "metric": metric,
                "chart_type": "Histogram"
            },

            {
                "title": "Category Contribution",
                "category": category,
                "metric": metric,
                "chart_type": "Pie"
            }
        ]
    })

    # --------------------------------
    # SHEET 3
    # --------------------------------

    second_metric = (
        numeric[1]
        if len(numeric) > 1
        else metric
    )

    sheets.append({

        "name": "Customer & Category",

        "description":
            "Customer, product and category analysis",

        "charts": [

            {
                "title": "Category Performance",
                "category": category,
                "metric": metric,
                "chart_type": "Bar"
            },

            {
                "title": "Secondary Metric",
                "category": category,
                "metric": second_metric,
                "chart_type": "Bar"
            },

            {
                "title": "Category Share",
                "category": category,
                "metric": metric,
                "chart_type": "Pie"
            },

            {
                "title": "Metric Relationship",
                "category": category,
                "metric": metric,
                "chart_type": "Scatter"
            },

            {
                "title": "Performance Variability",
                "category": category,
                "metric": metric,
                "chart_type": "Box"
            }
        ]
    })

    # --------------------------------
    # SHEET 4
    # --------------------------------

    sheets.append({

        "name": "Operations & Risk",

        "description":
            "Operational performance and potential risk areas",

        "charts": [

            {
                "title": "Operational Performance",
                "category": category,
                "metric": metric,
                "chart_type": "Bar"
            },

            {
                "title": "Operational Trend",
                "category":
                    dates[0] if dates else category,
                "metric": metric,
                "chart_type": "Line"
            },

            {
                "title": "Performance Distribution",
                "category": None,
                "metric": metric,
                "chart_type": "Histogram"
            },

            {
                "title": "Performance Variability",
                "category": category,
                "metric": metric,
                "chart_type": "Box"
            },

            {
                "title": "Risk/Performance Relationship",
                "category": category,
                "metric": second_metric,
                "chart_type": "Scatter"
            }
        ]
    })

    # --------------------------------
    # SHEET 5
    # --------------------------------

    sheets.append({

        "name": "Business Development",

        "description":
            "Growth opportunities and development areas",

        "charts": [

            {
                "title": "Growth Opportunities",
                "category": category,
                "metric": metric,
                "chart_type": "Bar"
            },

            {
                "title": "Business Contribution",
                "category": category,
                "metric": metric,
                "chart_type": "Pie"
            },

            {
                "title": "Growth Pattern",
                "category":
                    dates[0] if dates else category,
                "metric": metric,
                "chart_type": "Line"
            },

            {
                "title": "Business Distribution",
                "category": None,
                "metric": metric,
                "chart_type": "Histogram"
            },

            {
                "title": "Business Variability",
                "category": category,
                "metric": metric,
                "chart_type": "Box"
            }
        ]
    })

    return sheets[:sheet_count]
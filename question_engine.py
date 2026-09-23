# ==========================================================
# QUESTION ENGINE
# ==========================================================

import pandas as pd
import numpy as np


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def _find_column(df, keywords):
    """
    Find a column using business-related keywords.
    """

    for keyword in keywords:

        for column in df.columns:

            if keyword.lower() in str(column).lower():
                return column

    return None


def _format_value(value):
    """
    Make values easier to read.
    """

    if pd.isna(value):
        return "unknown"

    if isinstance(value, (int, float, np.integer, np.floating)):

        if float(value).is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    return str(value)


# ==========================================================
# GENERATE QUESTIONS FROM DATASET
# ==========================================================

def generate_questions(df, insights=None, sheets=None):
    """
    Generate useful questions based on the uploaded dataset.

    Questions are created from the actual columns in the data.
    """

    questions = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    # ------------------------------------------------------
    # BUSINESS QUESTIONS FROM COLUMNS
    # ------------------------------------------------------

    sales_column = _find_column(
        df,
        [
            "sales",
            "revenue",
            "amount",
            "income",
            "profit"
        ]
    )

    product_column = _find_column(
        df,
        [
            "product",
            "item",
            "product name"
        ]
    )

    customer_column = _find_column(
        df,
        [
            "customer",
            "client",
            "buyer"
        ]
    )

    department_column = _find_column(
        df,
        [
            "department",
            "team",
            "division"
        ]
    )

    attrition_column = _find_column(
        df,
        [
            "attrition",
            "churn",
            "left",
            "exit",
            "status"
        ]
    )

    campaign_column = _find_column(
        df,
        [
            "campaign",
            "promotion",
            "offer",
            "marketing"
        ]
    )

    response_column = _find_column(
        df,
        [
            "response",
            "converted",
            "conversion",
            "accepted"
        ]
    )

    region_column = _find_column(
        df,
        [
            "region",
            "state",
            "city",
            "location",
            "market"
        ]
    )

    satisfaction_column = _find_column(
        df,
        [
            "satisfaction",
            "rating",
            "score"
        ]
    )

    # ------------------------------------------------------
    # SALES QUESTIONS
    # ------------------------------------------------------

    if sales_column:

        questions.append(
            f"Which {product_column or 'category'} has the highest "
            f"{sales_column}?"
        )

        if region_column:

            questions.append(
                f"Which {region_column} performs best based on "
                f"{sales_column}?"
            )

        questions.append(
            f"How is {sales_column} distributed across the data?"
        )

    # ------------------------------------------------------
    # PRODUCT QUESTIONS
    # ------------------------------------------------------

    if product_column:

        questions.append(
            f"Which {product_column} appears most often?"
        )

        if sales_column:

            questions.append(
                f"Which {product_column} needs more attention?"
            )

    # ------------------------------------------------------
    # CUSTOMER QUESTIONS
    # ------------------------------------------------------

    if customer_column:

        questions.append(
            f"How many different {customer_column}s are in the data?"
        )

        questions.append(
            f"What customer information should the business "
            f"pay attention to?"
        )

    # ------------------------------------------------------
    # EMPLOYEE / HR QUESTIONS
    # ------------------------------------------------------

    if attrition_column:

        questions.append(
            "What employee retention problem can be seen in the data?"
        )

        if department_column:

            questions.append(
                f"Which {department_column} needs more attention "
                f"for employee retention?"
            )

    if department_column:

        questions.append(
            f"Which {department_column} has the largest employee group?"
        )

    if satisfaction_column:

        questions.append(
            f"What does the {satisfaction_column} information "
            f"tell us about the business?"
        )

    # ------------------------------------------------------
    # MARKETING QUESTIONS
    # ------------------------------------------------------

    if campaign_column:

        questions.append(
            f"Which {campaign_column} needs more attention?"
        )

        questions.append(
            f"What can the business learn from the "
            f"{campaign_column} data?"
        )

    if response_column:

        questions.append(
            "How are customers responding to the business activities?"
        )

    # ------------------------------------------------------
    # REGION / LOCATION QUESTIONS
    # ------------------------------------------------------

    if region_column:

        questions.append(
            f"Which {region_column} needs more attention?"
        )

        questions.append(
            f"What business differences can be seen across "
            f"{region_column}?"
        )

    # ------------------------------------------------------
    # NUMERIC QUESTIONS
    # ------------------------------------------------------

    for column in numeric_columns[:3]:

        questions.append(
            f"What does the {column} information tell us "
            f"about the business?"
        )

    # ------------------------------------------------------
    # QUESTIONS FROM INSIGHTS
    # ------------------------------------------------------

    if insights is not None:

        try:

            if isinstance(insights, pd.DataFrame):

                for _, row in insights.iterrows():

                    area = row.get(
                        "Area",
                        "this business area"
                    )

                    questions.append(
                        f"What should the business do about "
                        f"{area}?"
                    )

        except Exception:
            pass

    # ------------------------------------------------------
    # REMOVE DUPLICATES
    # ------------------------------------------------------

    final_questions = []

    for question in questions:

        if question not in final_questions:

            final_questions.append(question)

    # ------------------------------------------------------
    # FALLBACK QUESTIONS
    # ------------------------------------------------------

    if not final_questions:

        for column in df.columns[:5]:

            final_questions.append(
                f"What should the business understand "
                f"about {column}?"
            )

    return final_questions


# ==========================================================
# ANSWER USER QUESTION
# ==========================================================

def answer_question(
    df,
    question,
    insights=None,
    recommendations=None
):
    """
    Answer a question using the uploaded dataset.

    The function first tries to calculate a useful answer
    directly from the dataset.
    """

    if df is None or df.empty:

        return (
            "There is no uploaded data available to answer "
            "this question."
        )

    if not question:

        return (
            "Please select or enter a question."
        )

    question_lower = question.lower()

    # ======================================================
    # SALES / REVENUE
    # ======================================================

    sales_column = _find_column(
        df,
        [
            "sales",
            "revenue",
            "amount",
            "income",
            "profit"
        ]
    )

    product_column = _find_column(
        df,
        [
            "product",
            "item",
            "product name"
        ]
    )

    region_column = _find_column(
        df,
        [
            "region",
            "state",
            "city",
            "location",
            "market"
        ]
    )

    if sales_column:

        # --------------------------------------------------
        # HIGHEST PRODUCT
        # --------------------------------------------------

        if (
            product_column
            and (
                "highest" in question_lower
                or "top" in question_lower
                or "best" in question_lower
            )
            and (
                "product" in question_lower
                or "item" in question_lower
            )
        ):

            try:

                result = (
                    df.groupby(product_column)[sales_column]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not result.empty:

                    top = result.index[0]

                    return (
                        f"The product with the highest "
                        f"{sales_column} in the uploaded data "
                        f"is **{top}**."
                    )

            except Exception:
                pass

        # --------------------------------------------------
        # HIGHEST REGION
        # --------------------------------------------------

        if (
            region_column
            and (
                "region" in question_lower
                or "city" in question_lower
                or "location" in question_lower
                or "market" in question_lower
            )
            and (
                "highest" in question_lower
                or "best" in question_lower
                or "top" in question_lower
            )
        ):

            try:

                result = (
                    df.groupby(region_column)[sales_column]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not result.empty:

                    top = result.index[0]

                    return (
                        f"The strongest {region_column} based "
                        f"on {sales_column} is **{top}**."
                    )

            except Exception:
                pass

        # --------------------------------------------------
        # TOTAL SALES
        # --------------------------------------------------

        if (
            "total" in question_lower
            and (
                "sales" in question_lower
                or "revenue" in question_lower
                or "amount" in question_lower
                or "income" in question_lower
                or "profit" in question_lower
            )
        ):

            try:

                total = df[sales_column].sum()

                return (
                    f"The total {sales_column} in the uploaded "
                    f"data is **{total:,.2f}**."
                )

            except Exception:
                pass

    # ======================================================
    # CUSTOMER QUESTIONS
    # ======================================================

    customer_column = _find_column(
        df,
        [
            "customer",
            "client",
            "buyer"
        ]
    )

    if customer_column:

        if (
            "how many" in question_lower
            and (
                "customer" in question_lower
                or "client" in question_lower
                or "buyer" in question_lower
            )
        ):

            count = df[customer_column].nunique()

            return (
                f"The uploaded data contains "
                f"**{count:,} different {customer_column} values**."
            )

    # ======================================================
    # EMPLOYEE QUESTIONS
    # ======================================================

    attrition_column = _find_column(
        df,
        [
            "attrition",
            "churn",
            "left",
            "exit",
            "status"
        ]
    )

    if attrition_column:

        if (
            "retention" in question_lower
            or "leave" in question_lower
            or "leaving" in question_lower
            or "attrition" in question_lower
            or "churn" in question_lower
        ):

            values = (
                df[attrition_column]
                .dropna()
                .astype(str)
                .str.lower()
            )

            leaving_words = [
                "yes",
                "left",
                "leave",
                "leaving",
                "exit",
                "churn"
            ]

            leaving_count = values.isin(
                leaving_words
            ).sum()

            if leaving_count > 0:

                return (
                    "The uploaded data shows that some "
                    "employees are leaving the company. "
                    "The business should look at areas such "
                    "as career growth, workload, manager support "
                    "and employee experience."
                )

            return (
                "The uploaded data does not show a clear "
                "employee-leaving pattern from the available "
                "status information."
            )

    # ======================================================
    # MARKETING QUESTIONS
    # ======================================================

    campaign_column = _find_column(
        df,
        [
            "campaign",
            "promotion",
            "offer",
            "marketing"
        ]
    )

    response_column = _find_column(
        df,
        [
            "response",
            "converted",
            "conversion",
            "accepted"
        ]
    )

    if campaign_column:

        if (
            "campaign" in question_lower
            or "promotion" in question_lower
            or "offer" in question_lower
        ):

            counts = (
                df[campaign_column]
                .dropna()
                .astype(str)
                .value_counts()
            )

            if not counts.empty:

                top_campaign = counts.index[0]

                return (
                    f"**{top_campaign}** is the most common "
                    f"{campaign_column} value in the uploaded data. "
                    f"The business can review how customers responded "
                    f"to each campaign before planning future activities."
                )

    if response_column:

        if (
            "response" in question_lower
            or "conversion" in question_lower
            or "converted" in question_lower
        ):

            counts = (
                df[response_column]
                .dropna()
                .astype(str)
                .value_counts()
            )

            if not counts.empty:

                top_response = counts.index[0]

                return (
                    f"The most common customer response is "
                    f"**{top_response}**. "
                    f"This can help the business understand "
                    f"how customers react to its activities."
                )

    # ======================================================
    # CATEGORY QUESTIONS
    # ======================================================

    for column in df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns:

        if column.lower() in question_lower:

            counts = (
                df[column]
                .dropna()
                .astype(str)
                .value_counts()
            )

            if not counts.empty:

                top_value = counts.index[0]

                return (
                    f"The most common value in **{column}** "
                    f"is **{top_value}**."
                )

    # ======================================================
    # GENERAL DATA ANSWER
    # ======================================================

    return (
        f"I found your question, but the uploaded data does "
        f"not contain enough matching information to give a "
        f"specific answer automatically.\n\n"
        f"Relevant columns available in the dataset include: "
        f"{', '.join(map(str, df.columns[:10]))}."
    )
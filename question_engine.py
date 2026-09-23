import pandas as pd
import numpy as np


def _find_column(df, keywords):
    for keyword in keywords:
        for column in df.columns:
            if keyword.lower() in str(column).lower():
                return column
    return None


def generate_questions(df, insights=None, sheets=None):
    questions = []
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    sales_column = _find_column(df, ["sales", "revenue", "amount", "income", "profit"])
    product_column = _find_column(df, ["product", "item", "product name"])
    customer_column = _find_column(df, ["customer", "client", "buyer"])

    if sales_column:
        questions.append(f"Which {product_column or 'category'} has the highest {sales_column}?")
        questions.append(f"What is the total {sales_column} in the dataset?")

    if customer_column:
        questions.append(f"How many unique {customer_column} entries are recorded?")

    for col in numeric_columns[:3]:
        questions.append(f"What is the statistical summary of {col}?")

    if not questions:
        for column in df.columns[:5]:
            questions.append(f"What insights can be drawn from {column}?")

    return list(dict.fromkeys(questions))


def answer_question(df, question, insights=None, recommendations=None):
    if df is None or df.empty:
        return "There is no uploaded data available to answer this question."

    if not question:
        return "Please select or enter a question."

    q_lower = question.lower()
    sales_column = _find_column(df, ["sales", "revenue", "amount", "income", "profit"])
    product_column = _find_column(df, ["product", "item", "product name"])

    if sales_column and pd.api.types.is_numeric_dtype(df[sales_column]):
        if ("highest" in q_lower or "top" in q_lower) and product_column:
            try:
                res = df.groupby(product_column)[sales_column].sum().sort_values(ascending=False)
                if not res.empty:
                    return f"The product with the highest {sales_column} is **{res.index[0]}** ({res.iloc[0]:,.2f})."
            except Exception:
                pass

        if "total" in q_lower:
            total_val = df[sales_column].sum()
            return f"The total **{sales_column}** is **{total_val:,.2f}**."

    return (
        f"Query evaluated against columns: {', '.join(map(str, df.columns[:8]))}. "
        "Use the dynamic dashboard tab for deeper metric visualisations."
    )

import pandas as pd
import numpy as np
import plotly.express as px

DEFAULT_COLOR = "#2563EB"


def get_chart_colors(color):
    return [
        color,
        "#60A5FA",
        "#93C5FD",
        "#BFDBFE",
        "#DBEAFE",
        "#1D4ED8",
        "#1E40AF"
    ]


def create_chart(
    df,
    category=None,
    metric=None,
    chart_type="Bar",
    color="#2563EB",
    title="Business Chart"
):
    working_df = df.copy()

    numeric_columns = working_df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = working_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if metric is None and numeric_columns:
        metric = numeric_columns[0]

    if category is None and categorical_columns:
        category = categorical_columns[0]

    if metric not in working_df.columns:
        fig = px.bar(title=f"{title} - Selected metric unavailable")
        return fig

    try:
        # BAR
        if chart_type == "Bar":
            if category:
                chart_df = (
                    working_df
                    .groupby(category)[metric]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
                fig = px.bar(
                    chart_df,
                    x=category,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )
            else:
                fig = px.bar(
                    working_df,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )

        # LINE
        elif chart_type == "Line":
            if category:
                chart_df = (
                    working_df
                    .groupby(category)[metric]
                    .sum()
                    .reset_index()
                )
                fig = px.line(
                    chart_df,
                    x=category,
                    y=metric,
                    markers=True,
                    title=title,
                    color_discrete_sequence=[color]
                )
            else:
                fig = px.line(
                    working_df,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )

        # AREA
        elif chart_type == "Area":
            if category:
                chart_df = (
                    working_df
                    .groupby(category)[metric]
                    .sum()
                    .reset_index()
                )
                fig = px.area(
                    chart_df,
                    x=category,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )
            else:
                fig = px.area(
                    working_df,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )

        # PIE
        elif chart_type == "Pie":
            if category:
                chart_df = (
                    working_df
                    .groupby(category)[metric]
                    .sum()
                    .reset_index()
                    .sort_values(metric, ascending=False)
                    .head(10)
                )
                fig = px.pie(
                    chart_df,
                    names=category,
                    values=metric,
                    title=title,
                    color_discrete_sequence=get_chart_colors(color)
                )
            else:
                fig = px.pie(
                    working_df,
                    values=metric,
                    title=title,
                    color_discrete_sequence=get_chart_colors(color)
                )

        # HISTOGRAM
        elif chart_type == "Histogram":
            fig = px.histogram(
                working_df,
                x=metric,
                title=title,
                color_discrete_sequence=[color]
            )

        # SCATTER
        elif chart_type == "Scatter":
            x_column = metric
            second_numeric = [c for c in numeric_columns if c != metric]
            y_column = second_numeric[0] if second_numeric else metric

            fig = px.scatter(
                working_df,
                x=x_column,
                y=y_column,
                title=title,
                color_discrete_sequence=[color]
            )

        # BOX
        elif chart_type == "Box":
            if category:
                fig = px.box(
                    working_df,
                    x=category,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )
            else:
                fig = px.box(
                    working_df,
                    y=metric,
                    title=title,
                    color_discrete_sequence=[color]
                )

        # DEFAULT
        else:
            fig = px.bar(
                working_df,
                y=metric,
                title=title,
                color_discrete_sequence=[color]
            )

        fig.update_layout(
            height=380,
            margin=dict(l=30, r=30, t=60, b=30),
            template="plotly_white"
        )
        return fig

    except Exception as e:
        fig = px.bar(title=f"{title} - Unable to generate")
        fig.add_annotation(text=str(e), showarrow=False)
        return fig

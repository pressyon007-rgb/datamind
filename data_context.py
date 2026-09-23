import numpy as np


def create_data_context(df):
    context = []

    context.append("DATASET OVERVIEW")
    context.append(f"Rows: {len(df)}")
    context.append(f"Columns: {len(df.columns)}")

    context.append("\nCOLUMNS")
    for column in df.columns:
        context.append(
            f"- {column}: {df[column].dtype}, unique={df[column].nunique()}"
        )

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    if numeric:
        context.append("\nNUMERIC SUMMARY")
        for column in numeric:
            series = df[column].dropna()
            if series.empty:
                continue

            context.append(
                f"{column}: total={series.sum():.2f}, "
                f"average={series.mean():.2f}, "
                f"median={series.median():.2f}, "
                f"min={series.min():.2f}, "
                f"max={series.max():.2f}"
            )

    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if categorical:
        context.append("\nCATEGORICAL SUMMARY")
        for column in categorical:
            if df[column].nunique() <= 50:
                values = df[column].value_counts().head(10).to_dict()
                context.append(f"{column}: {values}")

    context.append("\nSAMPLE DATA")
    context.append(df.head(10).to_string(index=False))

    return "\n".join(context)

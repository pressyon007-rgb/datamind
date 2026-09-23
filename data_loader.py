import pandas as pd


def load_file(uploaded_file):
    """
    Load CSV or Excel file.
    """

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        elif file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError(
                "Unsupported file format. Upload CSV, XLSX or XLS."
            )

        if df.empty:
            raise ValueError("The uploaded file contains no data.")

        return df

    except Exception as e:
        raise Exception(f"Error loading file: {e}")


def convert_date_columns(df):

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            try:

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                if converted.notna().mean() >= 0.70:

                    df[column] = converted

            except Exception:
                pass

    return df


def detect_column_types(df):

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    date_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    categorical_columns = [
        c for c in categorical_columns
        if c not in date_columns
    ]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "date": date_columns
    }
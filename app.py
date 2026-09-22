import os
import re
import html
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
import sys
import socket
import subprocess
import pickle
import time

import streamlit as st
import pandas as pd
from modules.user_demand_engine import render_user_demand_section
from modules.data_loader import (
    load_file,
    convert_date_columns,
    detect_column_types
)

from modules.profiler import (
    get_data_profile,
    get_missing_values,
    get_data_quality_score,
    get_summary_statistics,
    detect_outliers
)

from modules.analytics import (
    calculate_kpis,
    calculate_correlations,
    get_numeric_summary
)

from modules.dashboard_engine import (
    generate_sheet_templates
)

from modules.chart_engine import (
    create_chart
)

from modules.insight_engine import (
    generate_insights
)

from modules.recommendation_engine import (
    generate_recommendations
)

from modules.web_research import (
    research_dataset,
    get_research_sources
)

from modules.data_context import (
    create_data_context
)

from modules.ai_analyst import (
    ask_data_analyst
)

from modules.report_generator import (
    generate_pdf
)

# ==========================================================
# DATA-DRIVEN QUESTION GENERATOR
# ==========================================================

def generate_data_questions(df):

    questions = []

    columns = df.columns.tolist()

    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category", "bool"]
        )
        .columns
        .tolist()
    )

    questions.append(
        f"What are the main characteristics of this dataset?"
    )

    questions.append(
        f"Which columns contain the most useful information "
        f"for analysis?"
    )

    for column in categorical_columns[:3]:

        questions.append(
            f"What are the most common groups in {column}?"
        )

        if numeric_columns:

            metric = numeric_columns[0]

            questions.append(
                f"How does {metric} vary across {column}?"
            )

    for column in numeric_columns[:3]:

        questions.append(
            f"What is the overall pattern of {column}?"
        )

    if len(numeric_columns) >= 2:

        x = numeric_columns[0]
        y = numeric_columns[1]

        questions.append(
            f"Is there a relationship between {x} and {y}?"
        )

    missing_columns = [
        column
        for column in columns
        if df[column].isna().sum() > 0
    ]

    if missing_columns:

        questions.append(
            "Which columns have missing information "
            "and what should be investigated?"
        )

    if df.duplicated().sum() > 0:

        questions.append(
            "Are there duplicate records in the dataset "
            "and why might they exist?"
        )

    cleaned_questions = []

    for question in questions:

        if question not in cleaned_questions:

            cleaned_questions.append(question)

    return cleaned_questions[:12]


# ==========================================================
# ONLINE BASIC DATA EXPLANATION
# ==========================================================

class _SearchResultParser(HTMLParser):
    """Small dependency-free parser for DuckDuckGo HTML results."""

    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._capture_title = False
        self._capture_snippet = False
        self._title_parts = []
        self._snippet_parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._capture_title = True
            self._title_parts = []
            self._current = {
                "title": "",
                "url": attrs.get("href", "")
            }
        elif tag in ("a", "div") and "result__snippet" in classes:
            self._capture_snippet = True
            self._snippet_parts = []

    def handle_data(self, data):
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._capture_title:
            self._capture_title = False
            if self._current is not None:
                self._current["title"] = " ".join(
                    self._title_parts
                ).strip()

        if self._capture_snippet and tag in ("a", "div"):
            self._capture_snippet = False
            if self._current is not None:
                self._current["snippet"] = " ".join(
                    self._snippet_parts
                ).strip()

                if self._current.get("title"):
                    self.results.append(self._current)

                self._current = None


def _clean_text(value):
    value = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", value).strip()


def _search_web(query, max_results=5):
    """Run a lightweight public web search without requiring an API key."""
    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded}"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(
            request,
            timeout=8
        ) as response:

            content = response.read().decode(
                "utf-8",
                errors="ignore"
            )

        parser = _SearchResultParser()
        parser.feed(content)

        cleaned = []
        seen = set()

        for item in parser.results:

            title = _clean_text(
                item.get("title")
            )

            snippet = _clean_text(
                item.get("snippet")
            )

            raw_url = item.get(
                "url",
                ""
            )

            match = re.search(
                r"uddg=([^&]+)",
                raw_url
            )

            if match:
                raw_url = urllib.parse.unquote(
                    match.group(1)
                )

            if not raw_url.startswith("http"):
                continue

            key = raw_url.split("#")[0]

            if key in seen:
                continue

            seen.add(key)

            cleaned.append(
                {
                    "title": title or "Web source",
                    "url": raw_url,
                    "snippet": snippet
                }
            )

            if len(cleaned) >= max_results:
                break

        return cleaned

    except Exception:
        return []


def _detect_business_domain(df):
    """Identify a likely business domain from column names only."""

    names = " ".join(
        str(c).lower()
        for c in df.columns
    )

    domain_rules = [

        (
            "Retail / Sales Analytics",
            [
                "retail sales",
                "retail transfers",
                "warehouse sales",
                "product",
                "item",
                "sales",
                "quantity",
                "order"
            ]
        ),

        (
            "Human Resources / Workforce Analytics",
            [
                "employee",
                "attrition",
                "jobrole",
                "job role",
                "department",
                "salary",
                "monthlyincome",
                "overtime",
                "satisfaction"
            ]
        ),

        (
            "Marketing / Campaign Analytics",
            [
                "campaign",
                "marketing",
                "conversion",
                "response",
                "click",
                "impression",
                "lead",
                "customer"
            ]
        ),

        (
            "Finance / Risk Analytics",
            [
                "loan",
                "credit",
                "default",
                "transaction",
                "fraud",
                "balance",
                "interest",
                "risk",
                "amount"
            ]
        ),

        (
            "Healthcare Analytics",
            [
                "patient",
                "diagnosis",
                "hospital",
                "admission",
                "discharge",
                "medical",
                "treatment",
                "disease"
            ]
        ),

        (
            "Education Analytics",
            [
                "student",
                "grade",
                "marks",
                "score",
                "attendance",
                "course",
                "exam",
                "education"
            ]
        ),

        (
            "Manufacturing Analytics",
            [
                "machine",
                "defect",
                "production",
                "maintenance",
                "downtime",
                "quality",
                "temperature",
                "pressure"
            ]
        ),

        (
            "Telecom Analytics",
            [
                "churn",
                "tenure",
                "contract",
                "internet",
                "telecom",
                "monthlycharges",
                "phone service"
            ]
        ),
    ]

    best_domain = "General Business Analytics"
    best_score = 0

    for domain, keywords in domain_rules:

        score = sum(
            1
            for keyword in keywords
            if keyword in names
        )

        if score > best_score:
            best_domain = domain
            best_score = score

    return best_domain


def _column_local_explanation(column, dtype):
    """Safe explanation when an exact online definition is not found."""

    name = str(column)

    n = (
        name
        .lower()
        .replace("_", " ")
        .strip()
    )

    known = {

        "year":
            "Calendar year associated with the record.",

        "month":
            "Month associated with the record.",

        "supplier":
            "Supplier or vendor associated with the product or record.",

        "item code":
            "Identifier used to distinguish a product/item.",

        "item description":
            "Text description or name of the product/item.",

        "item type":
            "Category or type assigned to the product/item.",

        "retail sales":
            "Retail sales quantity/value recorded for the product, depending on the dataset's unit definition.",

        "retail transfers":
            "Quantity/value associated with transfers to retail locations or operations, depending on the source definition.",

        "warehouse sales":
            "Sales quantity/value associated with warehouse operations, depending on the source definition.",

        "product":
            "Product or item associated with the record.",

        "quantity":
            "Number of units/items associated with the record.",

        "sales":
            "Sales measure recorded for the transaction, product, period, or business unit.",

        "revenue":
            "Revenue or monetary sales amount associated with the record.",

        "price":
            "Price or monetary amount associated with the product or transaction.",

        "customer":
            "Customer identifier or customer-related information.",

        "order date":
            "Date on which the order was placed.",
    }

    if n in known:
        return known[n]

    if dtype.startswith("datetime"):
        return (
            "Date/time field that can be used for "
            "time-based analysis and trends."
        )

    if (
        dtype.startswith("int")
        or dtype.startswith("float")
    ):
        return (
            "Numeric field that can be summarized, "
            "compared, grouped, or used as a business metric."
        )

    if dtype == "bool":
        return (
            "Boolean field representing a true/false "
            "or yes/no condition."
        )

    return (
        "Categorical/text field that can be used to "
        "group, filter, compare, or describe records."
    )


def generate_basic_data_explanation(df):
    """Research the uploaded dataset/domain online and explain the actual data."""

    domain = _detect_business_domain(df)

    columns = list(df.columns)

    column_text = ", ".join(
        str(c)
        for c in columns[:12]
    )

    queries = []

    names_lower = " ".join(
        str(c).lower()
        for c in columns
    )

    if all(
        x in names_lower
        for x in [
            "retail sales",
            "retail transfers",
            "warehouse sales"
        ]
    ):

        queries.append(
            '"RETAIL SALES" '
            '"RETAIL TRANSFERS" '
            '"WAREHOUSE SALES" dataset'
        )

    queries.append(
        f'"{domain}" dataset column definitions {column_text}'
    )

    queries.append(
        f'{domain} data analytics business uses '
        f'sales product performance inventory'
    )

    all_results = []
    seen_urls = set()

    for query in queries:

        for result in _search_web(
            query,
            max_results=5
        ):

            if result["url"] not in seen_urls:

                seen_urls.add(
                    result["url"]
                )

                all_results.append(
                    result
                )

            if len(all_results) >= 10:
                break

        if len(all_results) >= 10:
            break

    online_text = []

    for result in all_results[:5]:

        if result.get("snippet"):
            online_text.append(
                result["snippet"]
            )

    exact_source_match = None

    if all(
        x in names_lower
        for x in [
            "retail sales",
            "retail transfers",
            "warehouse sales"
        ]
    ):

        for result in all_results:

            text = (
                result.get("title", "")
                + " "
                + result.get("snippet", "")
            ).lower()

            if (
                "warehouse and retail sales" in text
                or "montgomery" in text
            ):

                exact_source_match = result
                break

    if exact_source_match:

        dataset_description = (
            "The uploaded columns closely match the publicly documented "
            "Warehouse and Retail Sales dataset structure. The published "
            "documentation describes sales and movement data by item and "
            "department, with fields such as supplier, item code, item type, "
            "retail sales, retail transfers and warehouse sales. "
            "The exact meaning and unit should still be treated according "
            "to the identified source documentation."
        )

    else:

        dataset_description = (
            f"The uploaded file appears to be a "
            f"{domain.lower()} dataset "
            f"with {len(df):,} records and {len(columns)} columns. "
            "The explanation below combines the actual structure of the "
            "uploaded file with publicly available information found online. "
            "Where an exact source definition could not be verified, the app "
            "labels the meaning as an interpretation rather than a confirmed definition."
        )

    business_uses = []

    domain_lower = domain.lower()

    if "retail" in domain_lower:

        business_uses = [
            "Product and sales performance monitoring",
            "Demand and inventory planning",
            "Product/category comparison",
            "Sales trend and seasonal analysis",
            "Merchandising, pricing and promotion decisions",
        ]

    elif "human resources" in domain_lower:

        business_uses = [
            "Workforce and employee trend analysis",
            "Attrition and retention monitoring",
            "Department and job-role comparison",
            "Workforce planning",
            "Employee experience analysis",
        ]

    elif "marketing" in domain_lower:

        business_uses = [
            "Campaign performance analysis",
            "Customer response and conversion analysis",
            "Audience segmentation",
            "Channel comparison",
            "Marketing performance monitoring",
        ]

    else:

        business_uses = [
            "Business performance monitoring",
            "Trend and group comparison",
            "Data quality monitoring",
            "Identification of important business patterns",
            "Decision support and further investigation",
        ]

    column_rows = []

    for column in columns:

        dtype = str(
            df[column].dtype
        )

        exact_match = None

        normalized = (
            str(column)
            .lower()
            .replace("_", " ")
            .strip()
        )

        for result in all_results:

            blob = (
                result.get("title", "")
                + " "
                + result.get("snippet", "")
            ).lower()

            if (
                normalized
                and normalized in blob
                and len(normalized) >= 4
            ):

                snippet = result.get(
                    "snippet",
                    ""
                )

                if snippet:
                    exact_match = snippet
                    break

        explanation = (
            exact_match
            if exact_match
            else _column_local_explanation(
                column,
                dtype
            )
        )

        column_rows.append(
            {
                "Column": column,
                "Data Type": dtype,
                "Meaning / Explanation": explanation,
                "Missing Values":
                    int(
                        df[column].isna().sum()
                    ),
                "Unique Values":
                    int(
                        df[column].nunique(
                            dropna=True
                        )
                    ),
            }
        )

    return {
        "domain": domain,
        "dataset_description": dataset_description,
        "business_uses": business_uses,
        "column_rows": column_rows,
        "online_evidence": online_text,
        "sources": all_results[:10],
        "research_status":
            (
                "Online research completed"
                if all_results
                else
                "Online research unavailable; "
                "local structural explanation used"
            ),
    }


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Automated BI Platform",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin-top: 20px;
    }

    .business-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        background-color: #FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "sheets" not in st.session_state:
    st.session_state.sheets = []

if "insights" not in st.session_state:
    st.session_state.insights = pd.DataFrame()

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "questions" not in st.session_state:
    st.session_state.questions = []

if "data_key" not in st.session_state:
    st.session_state.data_key = None

if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "research_sources" not in st.session_state:
    st.session_state.research_sources = []

if "basic_data_explanation" not in st.session_state:
    st.session_state.basic_data_explanation = None

if "real_user_result" not in st.session_state:
    st.session_state.real_user_result = None


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '📊 Automated Business Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a dataset and automatically create dashboards, "
    "analysis, domain research, insights and recommendations."
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("⚙️ Platform Controls")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    sheet_count = st.selectbox(
        "Number of Dashboard Sheets",
        [4, 5],
        index=0
    )

    st.caption(
        "4 sheets = 20 charts\n\n"
        "5 sheets = 25 charts"
    )


# ==========================================================
# LOAD DATA
# ==========================================================

if uploaded_file:

    current_data_key = (
        uploaded_file.name,
        getattr(
            uploaded_file,
            "size",
            None
        ),
        sheet_count
    )

    if st.session_state.data_key != current_data_key:

        try:

            df = load_file(
                uploaded_file
            )

            df = convert_date_columns(
                df
            )

            st.session_state.df = df

            st.session_state.sheets = (
                generate_sheet_templates(
                    df,
                    sheet_count
                )
            )

            for sheet_index, sheet in enumerate(
                st.session_state.sheets
            ):

                for chart_index, chart in enumerate(
                    sheet.get(
                        "charts",
                        []
                    )
                ):

                    chart.setdefault(
                        "chart_id",
                        f"sheet{sheet_index}_chart{chart_index}"
                    )

                    chart.setdefault(
                        "position",
                        chart_index + 1
                    )

                    chart.setdefault(
                        "color",
                        "#2563EB"
                    )

            st.session_state.insights = (
                generate_insights(
                    df
                )
            )

            try:

                st.session_state.recommendations = (
                    generate_recommendations(
                        df
                    )
                )

            except Exception as recommendation_error:

                st.session_state.recommendations = []

                st.warning(
                    f"Recommendation generation failed: "
                    f"{recommendation_error}"
                )

            try:

                st.session_state.basic_data_explanation = (
                    generate_basic_data_explanation(
                        df
                    )
                )

                basic_research = (
                    st.session_state.basic_data_explanation
                )

                st.session_state.research_result = {
                    "domain":
                        basic_research.get(
                            "domain",
                            "General Business Analytics"
                        ),

                    "analysis":
                        basic_research.get(
                            "dataset_description",
                            "No online explanation available."
                        ),
                }

                st.session_state.research_sources = (
                    basic_research.get(
                        "sources",
                        []
                    )
                )

            except Exception as research_error:

                st.session_state.basic_data_explanation = {
                    "domain":
                        _detect_business_domain(
                            df
                        ),

                    "dataset_description":
                        (
                            "Online research could not be completed: "
                            f"{research_error}"
                        ),

                    "business_uses": [],
                    "column_rows": [],
                    "online_evidence": [],
                    "sources": [],

                    "research_status":
                        "Online research failed",
                }

                st.session_state.research_result = {
                    "domain":
                        st.session_state
                        .basic_data_explanation[
                            "domain"
                        ],

                    "analysis":
                        st.session_state
                        .basic_data_explanation[
                            "dataset_description"
                        ],
                }

                st.session_state.research_sources = []

            st.session_state.data_key = (
                current_data_key
            )

        except Exception as e:

            st.error(
                f"Unable to process file: {e}"
            )

            st.stop()

else:

    st.info(
        "👈 Upload your CSV or Excel file from the sidebar."
    )

    st.stop()


# ==========================================================
# DATA INFORMATION / MAIN NAVIGATION
# ==========================================================

df = st.session_state.df

column_types = detect_column_types(
    df
)

kpis = calculate_kpis(
    df
)

tabs = st.tabs(
    [
        "🏢 Overview",
        "📊 Dashboard Builder",
        "📈 Statistics",
        "🚨 Business Insights",
        "🔎 Domain Research",
        "💡 Recommendations",
        "🤖 Ask Data",
        "📄 Final Report"
    ]
)


# ==========================================================
# OVERVIEW
# ==========================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">'
        'Business Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This page contains only the most important "
        "information required for business understanding."
    )

    cols = st.columns(4)

    cols[0].metric(
        "Total Records",
        f"{len(df):,}"
    )

    cols[1].metric(
        "Total Columns",
        len(df.columns)
    )

    cols[2].metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )

    cols[3].metric(
        "Data Quality",
        f"{get_data_quality_score(df)}%"
    )

    st.divider()

    st.subheader(
        "Business-Level Metrics"
    )

    numeric_summary = get_numeric_summary(
        df
    )

    if not numeric_summary.empty:

        st.dataframe(
            numeric_summary,
            use_container_width=True
        )

    st.subheader(
        "Data Coverage"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Numeric Columns",
        len(column_types["numeric"])
    )

    c2.metric(
        "Categorical Columns",
        len(column_types["categorical"])
    )

    c3.metric(
        "Date Columns",
        len(column_types["date"])
    )


# ==========================================================
# DASHBOARD BUILDER
# ==========================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">'
        '📊 Dashboard Builder'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Automatically generated business sheets with "
        "5 charts per sheet. Use the customizer to change "
        "the chart, metric, dimension, colour, title and position."
    )

    control1, control2, control3 = st.columns(
        [2, 2, 2]
    )

    with control1:

        if st.button(
            "🔄 Generate / Regenerate Dashboard",
            key="regenerate_dashboard",
            use_container_width=True
        ):

            st.session_state.sheets = (
                generate_sheet_templates(
                    df,
                    sheet_count
                )
            )

            for si, sheet in enumerate(
                st.session_state.sheets
            ):

                for ci, chart in enumerate(
                    sheet.get(
                        "charts",
                        []
                    )
                ):

                    chart["chart_id"] = (
                        f"sheet{si}_chart{ci}"
                    )

                    chart["position"] = (
                        ci + 1
                    )

                    chart.setdefault(
                        "color",
                        "#2563EB"
                    )

            st.rerun()

    with control2:

        layout_columns = st.selectbox(
            "🧩 Dashboard Layout",
            [1, 2, 3],
            index=1,
            key="dashboard_layout_columns",
            help=(
                "Choose how many chart columns are displayed "
                "in each dashboard sheet."
            )
        )

    with control3:

        total_charts = sum(
            len(
                sheet.get(
                    "charts",
                    []
                )
            )
            for sheet in st.session_state.sheets
        )

        st.metric(
            "📊 Dashboard Charts",
            total_charts
        )

    st.divider()

    sheet_tabs = st.tabs(
        [
            sheet["name"]
            for sheet in st.session_state.sheets
        ]
    )

    for sheet_index, (
        sheet_tab,
        sheet
    ) in enumerate(
        zip(
            sheet_tabs,
            st.session_state.sheets
        )
    ):

        with sheet_tab:

            st.subheader(
                f"📁 {sheet['name']}"
            )

            st.caption(
                sheet["description"]
            )

            charts = sheet.get(
                "charts",
                []
            )

            for chart_index, chart in enumerate(
                charts
            ):

                chart.setdefault(
                    "chart_id",
                    f"sheet{sheet_index}_chart{chart_index}"
                )

                chart.setdefault(
                    "position",
                    chart_index + 1
                )

                chart.setdefault(
                    "color",
                    "#2563EB"
                )

            st.write(
                f"Charts in this sheet: **{len(charts)}**"
            )

            st.divider()

            st.markdown(
                "### 🎨 Customize Dashboard"
            )

            chart_labels = [
                (
                    f"Chart {i + 1}: "
                    f"{chart.get('title', 'Business Chart')}"
                )
                for i, chart in enumerate(
                    charts
                )
            ]

            selected_chart_index = st.selectbox(
                "Select a chart to customize",
                range(len(charts)),
                format_func=lambda i:
                    chart_labels[i],
                key=f"selected_chart_{sheet_index}"
            )

            selected_chart = charts[
                selected_chart_index
            ]

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**Editing:** Chart "
                    f"{selected_chart_index + 1} — "
                    f"{selected_chart.get('title', 'Business Chart')}"
                )

                edit1, edit2, edit3 = st.columns(3)

                with edit1:

                    numeric_options = (
                        df.select_dtypes(
                            include="number"
                        )
                        .columns
                        .tolist()
                    )

                    metric_options = (
                        numeric_options
                        if numeric_options
                        else ["None"]
                    )

                    current_metric = (
                        selected_chart.get(
                            "metric"
                        )
                    )

                    metric_index = (
                        metric_options.index(
                            current_metric
                        )
                        if current_metric
                        in metric_options
                        else 0
                    )

                    new_metric = st.selectbox(
                        "📊 Metric",
                        metric_options,
                        index=metric_index,
                        key=(
                            f"edit_metric_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        )
                    )

                    if new_metric == "None":
                        new_metric = None

                with edit2:

                    dimension_options = [
                        "None"
                    ] + (
                        df.select_dtypes(
                            include=[
                                "object",
                                "category",
                                "bool",
                                "datetime"
                            ]
                        )
                        .columns
                        .tolist()
                    )

                    current_dimension = (
                        selected_chart.get(
                            "category"
                        )
                    )

                    dimension_index = (
                        dimension_options.index(
                            current_dimension
                        )
                        if current_dimension
                        in dimension_options
                        else 0
                    )

                    new_dimension = st.selectbox(
                        "🏷️ Dimension",
                        dimension_options,
                        index=dimension_index,
                        key=(
                            f"edit_dimension_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        )
                    )

                    if new_dimension == "None":
                        new_dimension = None

                with edit3:

                    chart_types = [
                        "Bar",
                        "Line",
                        "Area",
                        "Pie",
                        "Histogram",
                        "Scatter",
                        "Box"
                    ]

                    current_type = selected_chart.get(
                        "chart_type",
                        "Bar"
                    )

                    type_index = (
                        chart_types.index(
                            current_type
                        )
                        if current_type
                        in chart_types
                        else 0
                    )

                    new_chart_type = st.selectbox(
                        "📈 Chart Type",
                        chart_types,
                        index=type_index,
                        key=(
                            f"edit_type_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        )
                    )

                edit4, edit5, edit6 = st.columns(3)

                with edit4:

                    new_color = st.color_picker(
                        "🎨 Chart Color",
                        selected_chart.get(
                            "color",
                            "#2563EB"
                        ),
                        key=(
                            f"edit_color_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        )
                    )

                with edit5:

                    new_title = st.text_input(
                        "✏️ Chart Title",
                        selected_chart.get(
                            "title",
                            "Business Chart"
                        ),
                        key=(
                            f"edit_title_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        )
                    )

                with edit6:

                    position_options = list(
                        range(
                            1,
                            len(charts) + 1
                        )
                    )

                    current_position = (
                        selected_chart.get(
                            "position",
                            selected_chart_index + 1
                        )
                    )

                    if (
                        current_position
                        not in position_options
                    ):
                        current_position = (
                            selected_chart_index + 1
                        )

                    new_position = st.selectbox(
                        "↕️ Chart Position",
                        position_options,
                        index=position_options.index(
                            current_position
                        ),
                        key=(
                            f"edit_position_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        ),
                        help=(
                            "Position 1 appears first, position 2 "
                            "second, and so on. If another chart "
                            "already has the selected position, the "
                            "two charts will swap positions."
                        )
                    )

                button1, button2 = st.columns(2)

                with button1:

                    if st.button(
                        "💾 Apply Chart Changes",
                        key=(
                            f"apply_customization_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        ),
                        use_container_width=True,
                        type="primary"
                    ):

                        old_position = (
                            selected_chart.get(
                                "position",
                                selected_chart_index + 1
                            )
                        )

                        # Swap positions when needed.
                        if new_position != old_position:
                            for other_index, other_chart in enumerate(charts):
                                if (
                                    other_index != selected_chart_index
                                    and other_chart.get(
                                        "position",
                                        other_index + 1
                                    ) == new_position
                                ):
                                    other_chart["position"] = old_position
                                    break

                        selected_chart["metric"] = new_metric
                        selected_chart["category"] = new_dimension
                        selected_chart["chart_type"] = new_chart_type
                        selected_chart["color"] = new_color
                        selected_chart["title"] = (
                            new_title.strip()
                            or "Business Chart"
                        )
                        selected_chart["position"] = new_position

                        st.success(
                            "✅ Chart customization saved."
                        )

                        st.rerun()

                with button2:

                    if st.button(
                        "🔄 Reset This Chart",
                        key=(
                            f"reset_customization_"
                            f"{sheet_index}_"
                            f"{selected_chart_index}"
                        ),
                        use_container_width=True
                    ):

                        selected_chart["chart_type"] = "Bar"

                        selected_chart["color"] = "#2563EB"

                        selected_chart["title"] = (
                            f"Business Chart "
                            f"{selected_chart_index + 1}"
                        )

                        selected_chart["position"] = (
                            selected_chart_index + 1
                        )

                        st.rerun()

            st.divider()

            st.markdown(
                "### 📊 Dashboard Preview"
            )

            ordered_charts = sorted(
                charts,
                key=lambda chart:
                    chart.get(
                        "position",
                        999
                    )
            )

            chart_columns = st.columns(
                layout_columns
            )

            for display_index, chart in enumerate(
                ordered_charts
            ):

                with chart_columns[
                    display_index % layout_columns
                ]:

                    fig = create_chart(
                        df,
                        category=chart.get(
                            "category"
                        ),
                        metric=chart.get(
                            "metric"
                        ),
                        chart_type=chart.get(
                            "chart_type",
                            "Bar"
                        ),
                        color=chart.get(
                            "color",
                            "#2563EB"
                        ),
                        title=chart.get(
                            "title",
                            "Business Chart"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=(
                            f"dashboard_chart_"
                            f"{sheet_index}_"
                            f"{chart.get('chart_id', display_index)}"
                        )
                    )

            st.success(
                f"✅ {sheet['name']} contains "
                f"{len(charts)} charts."
            )

    st.divider()

    st.metric(
        "TOTAL DASHBOARD CHARTS",
        total_charts
    )


# ==========================================================
# STATISTICS
# ==========================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">'
        'Statistical Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Summary Statistics"
    )

    statistics = get_summary_statistics(
        df
    )

    if not statistics.empty:

        st.dataframe(
            statistics,
            use_container_width=True
        )

    st.subheader(
        "Outlier Analysis"
    )

    outliers = detect_outliers(
        df
    )

    if not outliers.empty:

        st.dataframe(
            outliers,
            use_container_width=True
        )

    st.subheader(
        "Correlation Analysis"
    )

    correlations = calculate_correlations(
        df
    )

    if not correlations.empty:

        st.dataframe(
            correlations,
            use_container_width=True
        )


# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">'
        '🚨 Business Insights & Barriers'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "First understand what the uploaded data represents. "
        "Then review what the actual data shows and which areas "
        "may become business barriers."
    )

    basic = st.session_state.get(
        "basic_data_explanation"
    )

    if basic:

        st.markdown(
            "## 📚 Basic Data Explanation"
        )

        st.caption(
            basic.get(
                "research_status",
                "Online research status unavailable"
            )
        )

        st.markdown(
            "### 🏢 What is this dataset?"
        )

        st.info(
            basic.get(
                "dataset_description",
                "No dataset explanation is available."
            )
        )

        st.markdown(
            "### 🔎 Detected Business Domain"
        )

        st.success(
            basic.get(
                "domain",
                "General Business Analytics"
            )
        )

        st.markdown(
            "### 🎯 What can this type of data be used for?"
        )

        uses = basic.get(
            "business_uses",
            []
        )

        if uses:

            for use in uses:
                st.write(
                    f"- {use}"
                )

        else:

            st.write(
                "The available online research did not provide "
                "enough domain-specific information."
            )

        st.markdown(
            "### 📋 Column-by-Column Explanation"
        )

        column_rows = basic.get(
            "column_rows",
            []
        )

        if column_rows:

            st.dataframe(
                pd.DataFrame(
                    column_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No column explanations were generated."
            )

        evidence = basic.get(
            "online_evidence",
            []
        )

        if evidence:

            st.markdown(
                "### 🌐 What online sources say"
            )

            for item in evidence[:5]:

                st.write(
                    f"- {item}"
                )

        sources = basic.get(
            "sources",
            []
        )

        if sources:

            st.markdown(
                "### 📚 Online Sources Used"
            )

            for source in sources:

                title = source.get(
                    "title",
                    "Web Source"
                )

                url = source.get(
                    "url",
                    ""
                )

                if url:

                    st.markdown(
                        f"- [{title}]({url})"
                    )

        st.divider()

    st.markdown(
        "## 📊 What Your Uploaded Data Shows"
    )

    insights = st.session_state.insights

    if not insights.empty:

        for _, row in insights.iterrows():

            severity = row["Severity"]

            if severity == "HIGH":

                st.error(
                    f"🔴 {row['Area']}: "
                    f"{row['Problem']}"
                )

            elif severity == "MEDIUM":

                st.warning(
                    f"🟠 {row['Area']}: "
                    f"{row['Problem']}"
                )

            else:

                st.info(
                    f"🔵 {row['Area']}: "
                    f"{row['Problem']}"
                )

            st.write(
                f"**Business Impact:** "
                f"{row['Business Impact']}"
            )

            st.write(
                f"**Evidence:** "
                f"{row['Evidence']}"
            )

            st.divider()


# ==========================================================
# DOMAIN RESEARCH
# ==========================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">🔎 Domain Research</div>',
        unsafe_allow_html=True
    )

    research = st.session_state.get(
        "research_result",
        {}
    )

    if not research:

        st.info(
            "📂 Upload a dataset to start domain research."
        )

    else:

        domain = research.get(
            "domain",
            "General Business Analytics"
        )

        st.success(
            f"🏢 Detected Business Domain: {domain}"
        )

        st.divider()

        # ==================================================
        # DATASET OVERVIEW
        # ==================================================

        st.markdown("## 📊 Dataset Overview")

        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

        with overview_col1:
            st.metric("Total Records", f"{len(df):,}")

        with overview_col2:
            st.metric("Total Columns", f"{len(df.columns):,}")

        with overview_col3:
            st.metric(
                "Numeric Columns",
                f"{len(df.select_dtypes(include='number').columns):,}"
            )

        with overview_col4:
            st.metric(
                "Categorical Columns",
                f"{len(df.select_dtypes(include=['object', 'category']).columns):,}"
            )

        st.divider()

        # ==================================================
        # BASIC DATA EXPLANATION
        # ==================================================

        basic = st.session_state.get(
            "basic_data_explanation"
        )

        if basic:

            st.markdown("## 📚 What Is This Dataset?")

            description = basic.get(
                "dataset_description",
                "No dataset description is available."
            )

            st.info(description)

            st.markdown("## 🎯 What Can This Data Be Used For?")

            uses = basic.get("business_uses", [])

            if uses:
                for use in uses:
                    st.write(f"✓ {use}")
            else:
                st.write(
                    "The available research did not identify enough "
                    "domain-specific business uses."
                )

            st.divider()

            # ----------------------------------------------
            # COLUMN EXPLANATION
            # ----------------------------------------------

            st.markdown("## 📋 Column-by-Column Explanation")

            column_rows = basic.get("column_rows", [])

            if column_rows:
                column_df = pd.DataFrame(column_rows)

                st.dataframe(
                    column_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No column explanations were generated.")

            st.divider()

            # ----------------------------------------------
            # ONLINE EVIDENCE
            # ----------------------------------------------

            evidence = basic.get("online_evidence", [])

            if evidence:
                st.markdown("## 🌐 What Online Sources Say")

                for item in evidence[:8]:
                    st.write(f"• {item}")

                st.divider()

        # ==================================================
        # RESEARCH-BASED ANALYSIS
        # ==================================================

        st.markdown("## 🔬 Research-Based Analysis")

        analysis = research.get(
            "analysis",
            "No research analysis is available."
        )

        st.write(analysis)

        st.divider()

        # ==================================================
        # SOURCES
        # ==================================================

        sources = st.session_state.get(
            "research_sources",
            []
        )

        if sources:

            st.markdown("## 📚 Online Sources Used")

            for source in sources:

                title = source.get(
                    "title",
                    "Web Source"
                )

                url = source.get(
                    "url",
                    ""
                )

                if url:
                    st.markdown(
                        f"- [{title}]({url})"
                    )
        else:
            st.info("No external sources were returned.")

        st.divider()

        # ==================================================
        # RESEARCH LIMITATION
        # ==================================================

        st.markdown("### ℹ️ Research Limitation")

        st.caption(
            "Online research is used to provide business and industry "
            "context. Where an exact source definition cannot be verified, "
            "the explanation is presented as an interpretation rather than "
            "a confirmed definition."
        )


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">'
        '💡 Recommendations'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Recommendations are generated from the actual uploaded "
        "dataset. Each business problem is evaluated using the "
        "available data before a recommendation is produced."
    )

    st.divider()

    recommendations = st.session_state.get(
        "recommendations",
        []
    )

    if not recommendations:

        st.info(
            "No recommendation areas were identified "
            "from the uploaded dataset."
        )

    else:

        st.markdown(
            "## 📊 Analysis Summary"
        )

        st.write(
            f"{len(recommendations)} data-driven analysis "
            "areas were identified from the uploaded dataset."
        )

        st.divider()

        for index, rec in enumerate(
            recommendations,
            start=1
        ):

            business_area = rec.get(
                "Business Area",
                "Business Analysis"
            )

            status = rec.get(
                "Status",
                "🟡 Needs Investigation"
            )

            if "🔴" in status:

                status_title = (
                    "🔴 Problem Detected"
                )

            elif "🟢" in status:

                status_title = (
                    "🟢 No Evidence Detected"
                )

            else:

                status_title = (
                    "🟡 Needs Investigation"
                )

            st.markdown(
                f"## 📌 {index}. {business_area}"
            )

            st.markdown(
                f"### {status_title}"
            )

            st.markdown(
                "### 📊 Data Evidence"
            )

            evidence = rec.get(
                "Evidence",
                "No specific evidence was generated."
            )

            st.info(
                evidence
            )

            st.markdown(
                "### 🔎 Detailed Analysis"
            )

            detailed_analysis = rec.get(
                "Detailed Analysis",
                ""
            )

            if detailed_analysis:

                st.write(
                    detailed_analysis
                )

            factor_analysis = rec.get(
                "Factor Analysis",
                ""
            )

            if factor_analysis:

                st.markdown(
                    "### 📈 Factor / Group Analysis"
                )

                st.code(
                    factor_analysis,
                    language="text"
                )

            st.markdown(
                "### ⚠️ Business Problem"
            )

            business_problem = rec.get(
                "Business Impact",
                ""
            )

            if business_problem:

                st.write(
                    business_problem
                )

            st.markdown(
                "### 🛠️ Corrective Measures"
            )

            corrective_measures = rec.get(
                "Corrective Measures",
                []
            )

            if corrective_measures:

                for number, action in enumerate(
                    corrective_measures,
                    start=1
                ):

                    st.write(
                        f"**{number}.** {action}"
                    )

            else:

                st.write(
                    "No immediate corrective action is "
                    "required based on the available data."
                )

            st.markdown(
                "### 🔧 How to Improve Current Working"
            )

            improvement = rec.get(
                "How to Improve Working",
                ""
            )

            if improvement:

                st.write(
                    improvement
                )

            workflow = rec.get(
                "Workflow",
                ""
            )

            if workflow:

                st.markdown(
                    "### 🔄 Recommended Review Process"
                )

                st.code(
                    workflow,
                    language="text"
                )

            st.markdown(
                "### 🚀 Development Opportunity"
            )

            development = rec.get(
                "Development Opportunity",
                ""
            )

            if development:

                st.success(
                    development
                )

            st.markdown(
                "### 🎯 Expected Outcome"
            )

            expected = rec.get(
                "Expected Outcome",
                ""
            )

            if expected:

                st.write(
                    expected
                )

            additional_data = rec.get(
                "Additional Data Required",
                []
            )

            if additional_data:

                st.markdown(
                    "### 📥 Additional Data Required"
                )

                for item in additional_data:

                    st.write(
                        f"- {item}"
                    )

            st.markdown(
                "### ⚠️ Limitation"
            )

            limitation = rec.get(
                "Limitation",
                (
                    "The dataset shows patterns and "
                    "associations. It does not automatically "
                    "prove causation."
                )
            )

            st.warning(
                limitation
            )

            st.divider()


# ==========================================================
# REAL DATA-DRIVEN ASK DATA ENGINE
# ==========================================================

def _find_column_ci(df, candidates):

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for c in candidates:

        key = str(c).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def _binary_positive_mask(series):

    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive = {
        "yes",
        "y",
        "1",
        "true",
        "left",
        "leaver",
        "attrited",
        "churned",
        "fraud",
        "default",
        "converted",
        "purchase",
        "purchased"
    }

    return values.isin(
        positive
    )


def _answer_user_request(
    df,
    request,
    target_dimension=None
):

    """Answer common natural-language questions directly from the uploaded dataframe.
    No fixed HR numbers are used; every displayed value is calculated from df at runtime.
    """

    q = str(
        request or ""
    ).strip().lower()

    if not q:
        return None

    target_col = _find_column_ci(
        df,
        [
            target_dimension or "",
            "Attrition",
            "Churn",
            "Default",
            "Fraud",
            "Converted",
            "Response",
            "Purchased",
            "Returned",
            "Defect",
            "Late",
            "Stockout",
            "Outcome",
            "Target",
            "Status"
        ]
    )

    dimension = None

    for col in df.columns:

        name = str(col).lower()

        if any(
            word in name
            for word in [
                "department",
                "gender",
                "jobrole",
                "job role",
                "category",
                "region",
                "city",
                "state",
                "segment",
                "business travel",
                "overtime",
                "marital",
                "education"
            ]
        ):

            if any(
                term in q
                for term in [
                    str(col).lower(),
                    name.replace("_", " ")
                ]
            ):

                dimension = col
                break

    if (
        dimension is None
        and target_dimension
    ):

        dimension = _find_column_ci(
            df,
            [target_dimension]
        )

    asks_highest = any(
        x in q
        for x in [
            "highest",
            "maximum",
            "max",
            "most"
        ]
    )

    asks_lowest = any(
        x in q
        for x in [
            "lowest",
            "minimum",
            "min",
            "least"
        ]
    )

    target_rate_words = any(
        x in q
        for x in [
            "attrition",
            "churn",
            "default",
            "fraud",
            "conversion",
            "converted",
            "response",
            "purchase",
            "purchased",
            "return",
            "defect",
            "late",
            "stockout"
        ]
    )

    if (
        target_col is not None
        and dimension is not None
        and (
            asks_highest
            or asks_lowest
            or target_rate_words
        )
    ):

        temp = df[
            [
                dimension,
                target_col
            ]
        ].copy()

        temp = temp.dropna(
            subset=[dimension]
        )

        if not temp.empty:

            temp["__positive"] = (
                _binary_positive_mask(
                    temp[target_col]
                )
            )

            grouped = (
                temp.groupby(
                    dimension,
                    dropna=False
                )
                .agg(
                    Records=(
                        "__positive",
                        "size"
                    ),
                    Positive=(
                        "__positive",
                        "sum"
                    )
                )
            )

            grouped["Rate"] = (
                grouped["Positive"]
                / grouped["Records"]
                * 100
            )

            min_records = (
                1
                if len(grouped) <= 10
                else 5
            )

            eligible = grouped[
                grouped["Records"]
                >= min_records
            ].copy()

            if not eligible.empty:

                if (
                    asks_highest
                    or not asks_lowest
                ):

                    row = (
                        eligible
                        .sort_values(
                            "Rate",
                            ascending=False
                        )
                        .iloc[0]
                    )

                else:

                    row = (
                        eligible

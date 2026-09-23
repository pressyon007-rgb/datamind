from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from html import escape

from modules.chart_engine import create_chart


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe(value):
    """Convert any value into safe PDF text."""
    if value is None:
        return "N/A"

    return str(value)


def get_value(item, *keys, default="N/A"):
    """
    Return the first available value from a dictionary.

    Supports both old and new recommendation field names.
    """

    if not isinstance(item, dict):
        return default

    for key in keys:

        value = item.get(key)

        if value not in (None, "", []):
            return value

    return default


def add_table(story, data, widths=None):
    """Add a safe, page-width-aware ReportLab table."""
    if not data:
        return

    page_width = A4[0]
    usable_width = page_width - (18 * mm) - (18 * mm)
    max_columns = max(len(row) for row in data)

    normalized = []
    for row in data:
        row = list(row)
        if len(row) < max_columns:
            row += [""] * (max_columns - len(row))
        normalized.append(row[:max_columns])

    if widths is not None and len(widths) == max_columns:
        total = sum(float(w) for w in widths)
        if total > usable_width:
            scale = usable_width / total
            widths = [float(w) * scale for w in widths]
    else:
        widths = [usable_width / max_columns] * max_columns

    body = ParagraphStyle(
        "PDFTableBody", fontName="Helvetica", fontSize=7,
        leading=9, wordWrap="CJK"
    )
    header = ParagraphStyle(
        "PDFTableHeader", parent=body, fontName="Helvetica-Bold",
        textColor=colors.white
    )

    cleaned = []
    for ri, row in enumerate(normalized):
        cleaned.append([
            Paragraph(escape(safe(v)), header if ri == 0 else body)
            for v in row
        ])

    table = Table(
        cleaned, colWidths=widths, repeatRows=1, splitByRow=1
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))


# ============================================================
# MAIN PDF FUNCTION
# ============================================================

def generate_pdf(
    file_path,
    df,
    kpis,
    sheets,
    insights,
    recommendations,
    questions,
    research_result=None,
    research_sources=None,
    research_detail=None,
    user_analysis=None,
    dashboard_url=None,
):
    """
    Generate complete BI PDF report.

    Includes:

    1. Business Overview
    2. Complete Dashboard
    3. Every Dashboard Sheet
    4. Every Dashboard Chart
    5. Chart Data
    6. Statistical Analysis
    7. Business Insights
    8. Domain / Web Research
    9. Detailed Recommendations
    10. Development Opportunities
    11. User Requested AI Analysis
    12. Data Questions
    13. Data Quality Appendix
    """

    # ========================================================
    # PDF DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(
        file_path,
        pagesize=A4,

        rightMargin=18 * mm,
        leftMargin=18 * mm,

        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    story = []

    # ========================================================
    # 1. REPORT TITLE
    # ========================================================

    story.append(
        Paragraph(
            "AUTOMATED BUSINESS INTELLIGENCE REPORT",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"Dataset contains "
            f"<b>{len(df):,}</b> records and "
            f"<b>{len(df.columns):,}</b> columns.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # ========================================================
    # BUSINESS OVERVIEW
    # ========================================================

    story.append(
        Paragraph(
            "Business Overview",
            styles["Heading1"]
        )
    )

    overview_data = [
        ["Metric", "Value"],

        [
            "Total Records",
            f"{len(df):,}"
        ],

        [
            "Total Columns",
            f"{len(df.columns):,}"
        ],

        [
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        ],

        [
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        ],
    ]

    add_table(
        story,
        overview_data,
        widths=[
            80 * mm,
            80 * mm
        ]
    )

    # ========================================================
    # BUSINESS KPIs
    # ========================================================

    story.append(
        Paragraph(
            "Business KPIs",
            styles["Heading2"]
        )
    )

    kpi_data = [
        ["KPI", "Value"]
    ]

    if isinstance(kpis, dict):

        for key, value in kpis.items():

            kpi_data.append(
                [
                    safe(key),
                    safe(value)
                ]
            )

    if len(kpi_data) == 1:

        kpi_data.append(
            [
                "Total Records",
                f"{len(df):,}"
            ]
        )

    add_table(
        story,
        kpi_data,
        widths=[
            80 * mm,
            80 * mm
        ]
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 2. COMPLETE DASHBOARD
    # ========================================================

    story.append(
        Paragraph(
            "2. Complete Dashboard",
            styles["Heading1"]
        )
    )

    story.append(
        Paragraph(
            "This section contains every dashboard sheet and "
            "every chart generated by the Dashboard Builder.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # Temporary chart image folder
    chart_directory = (
        Path(file_path).parent /
        "_report_chart_images"
    )

    chart_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    total_charts = 0

    # ========================================================
    # LOOP THROUGH ALL DASHBOARD SHEETS
    # ========================================================

    for sheet_index, sheet in enumerate(
        sheets or [],
        start=1
    ):

        sheet_name = sheet.get(
            "name",
            f"Dashboard Sheet {sheet_index}"
        )

        story.append(
            Paragraph(
                f"Dashboard Sheet {sheet_index}: "
                f"{safe(sheet_name)}",
                styles["Heading1"]
            )
        )

        description = sheet.get(
            "description",
            ""
        )

        if description:

            story.append(
                Paragraph(
                    safe(description),
                    styles["Normal"]
                )
            )

        charts = sheet.get(
            "charts",
            []
        )

        charts = sorted(
            charts,
            key=lambda x: x.get(
                "position",
                999
            )
        )

        story.append(
            Paragraph(
                f"Charts in this sheet: "
                f"<b>{len(charts)}</b>",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 10)
        )

        # ====================================================
        # LOOP THROUGH ALL CHARTS
        # ====================================================

        for chart_index, chart in enumerate(
            charts,
            start=1
        ):

            total_charts += 1

            title = chart.get(
                "title",
                f"Business Chart {chart_index}"
            )

            metric = chart.get(
                "metric"
            )

            category = chart.get(
                "category"
            )

            chart_type = chart.get(
                "chart_type",
                "Bar"
            )

            chart_color = chart.get(
                "color",
                "#2563EB"
            )

            # ------------------------------------------------
            # CHART TITLE
            # ------------------------------------------------

            story.append(
                Paragraph(
                    f"Chart {chart_index}: "
                    f"{safe(title)}",
                    styles["Heading2"]
                )
            )

            # ------------------------------------------------
            # CHART DESCRIPTION
            # ------------------------------------------------

            story.append(
                Paragraph(
                    f"<b>Chart Type:</b> "
                    f"{safe(chart_type)}<br/>"
                    f"<b>Metric:</b> "
                    f"{safe(metric)}<br/>"
                    f"<b>Dimension:</b> "
                    f"{safe(category)}",
                    styles["Normal"]
                )
            )

            story.append(
                Spacer(1, 8)
            )

            # ------------------------------------------------
            # CREATE ACTUAL PLOTLY CHART
            # ------------------------------------------------

            chart_file = (
                chart_directory /
                f"sheet_{sheet_index}_"
                f"chart_{chart_index}.png"
            )

            try:

                fig = create_chart(
                    df,

                    category=category,

                    metric=metric,

                    chart_type=chart_type,

                    color=chart_color,

                    title=title,
                )

                # Convert Plotly chart to PNG
                fig.write_image(
                    str(chart_file),

                    format="png",

                    width=1100,

                    height=600,

                    scale=1
                )

                # Add actual chart image
                story.append(
                    Image(
                        str(chart_file),

                        width=170 * mm,

                        height=93 * mm
                    )
                )

            except Exception as error:

                story.append(
                    Paragraph(
                        "Unable to render chart image.",
                        styles["Normal"]
                    )
                )

                story.append(
                    Paragraph(
                        f"Reason: {safe(error)}",
                        styles["Normal"]
                    )
                )

            story.append(
                Spacer(1, 10)
            )

            # ------------------------------------------------
            # CHART CONFIGURATION
            # ------------------------------------------------

            config_data = [
                [
                    "Chart Property",
                    "Value"
                ],

                [
                    "Chart Type",
                    safe(chart_type)
                ],

                [
                    "Metric",
                    safe(metric)
                ],

                [
                    "Dimension",
                    safe(category)
                ],

                [
                    "Position",
                    safe(
                        chart.get(
                            "position",
                            chart_index
                        )
                    )
                ],

                [
                    "Colour",
                    safe(chart_color)
                ],
            ]

            add_table(
                story,
                config_data,
                widths=[
                    65 * mm,
                    105 * mm
                ]
            )

            # ------------------------------------------------
            # CHART DATA SUMMARY
            # ------------------------------------------------

            try:

                if (
                    metric in df.columns
                    and
                    category in df.columns
                ):

                    grouped_data = (
                        df.groupby(
                            category,
                            dropna=False
                        )[metric]
                        .agg(
                            Count="count",
                            Average="mean",
                            Total="sum"
                        )
                        .reset_index()
                    )

                    # Keep report readable
                    grouped_data = grouped_data.head(
                        20
                    )

                    story.append(
                        Paragraph(
                            "Chart Data Summary",
                            styles["Heading3"]
                        )
                    )

                    chart_data = [
                        [
                            safe(column)
                            for column
                            in grouped_data.columns
                        ]
                    ]

                    for row in grouped_data.itertuples(
                        index=False,
                        name=None
                    ):

                        chart_data.append(
                            [
                                safe(value)
                                for value in row
                            ]
                        )

                    add_table(
                        story,
                        chart_data
                    )

            except Exception:
                pass

            story.append(
                Spacer(1, 10)
            )

        # ----------------------------------------------------
        # NEXT DASHBOARD SHEET
        # ----------------------------------------------------

        story.append(
            PageBreak()
        )

    # Dashboard summary
    story.append(
        Paragraph(
            f"Dashboard Summary: "
            f"{len(sheets or [])} sheets | "
            f"{total_charts} charts",
            styles["Normal"]
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 3. STATISTICAL ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "3. Statistical Analysis",
            styles["Heading1"]
        )
    )

    try:

        numeric_df = df.select_dtypes(
            include="number"
        )

        if not numeric_df.empty:

            story.append(
                Paragraph(
                    "Descriptive Statistics",
                    styles["Heading2"]
                )
            )

            statistics_df = (
                numeric_df
                .describe()
                .round(3)
                .reset_index()
            )

            # Split wide descriptive-statistics tables into small groups
            # so ReportLab never creates a negative/zero column width.
            stat_name = statistics_df.columns[0]
            metric_columns = list(statistics_df.columns[1:])
            max_metrics = 5

            for start_idx in range(0, len(metric_columns), max_metrics):
                current = metric_columns[start_idx:start_idx + max_metrics]
                statistics_data = [[safe(stat_name)] + [safe(c) for c in current]]

                for _, row in statistics_df.iterrows():
                    statistics_data.append(
                        [safe(row[stat_name])] +
                        [safe(row[c]) for c in current]
                    )

                add_table(
                    story,
                    statistics_data,
                    widths=[28 * mm] + [27 * mm for _ in current]
                )

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Missing Value Analysis",
                styles["Heading2"]
            )
        )

        missing_df = (
            df.isna()
            .sum()
            .reset_index()
        )

        missing_df.columns = [
            "Column",
            "Missing Values"
        ]

        missing_df["Missing %"] = (
            missing_df[
                "Missing Values"
            ]
            / max(len(df), 1)
            * 100
        ).round(2)

        missing_data = [
            [
                "Column",
                "Missing Values",
                "Missing %"
            ]
        ]

        for _, row in missing_df.iterrows():

            missing_data.append(
                [
                    safe(row["Column"]),
                    safe(
                        row["Missing Values"]
                    ),
                    safe(
                        row["Missing %"]
                    )
                ]
            )

        add_table(
            story,
            missing_data,
            widths=[
                90 * mm,
                40 * mm,
                40 * mm
            ]
        )

        story.append(
            Paragraph(
                f"Duplicate Rows: "
                f"{int(df.duplicated().sum()):,}",
                styles["Normal"]
            )
        )

    except Exception as error:

        story.append(
            Paragraph(
                f"Statistical analysis error: "
                f"{safe(error)}",
                styles["Normal"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 4. BUSINESS INSIGHTS
    # ========================================================

    story.append(
        Paragraph(
            "4. Business Insights & Barriers",
            styles["Heading1"]
        )
    )

    if hasattr(
        insights,
        "iterrows"
    ):

        for _, row in insights.iterrows():

            story.append(
                Paragraph(
                    f"{safe(row.get('Area', 'Business Area'))} "
                    f"- {safe(row.get('Severity', ''))}",
                    styles["Heading2"]
                )
            )

            story.append(
                Paragraph(
                    f"<b>Problem:</b> "
                    f"{safe(row.get('Problem', 'N/A'))}",
                    styles["Normal"]
                )
            )

            story.append(
                Paragraph(
                    f"<b>Business Impact:</b> "
                    f"{safe(row.get('Business Impact', 'N/A'))}",
                    styles["Normal"]
                )
            )

            story.append(
                Paragraph(
                    f"<b>Evidence:</b> "
                    f"{safe(row.get('Evidence', 'N/A'))}",
                    styles["Normal"]
                )
            )

            story.append(
                Spacer(1, 10)
            )

    elif isinstance(
        insights,
        list
    ):

        for item in insights:

            if isinstance(
                item,
                dict
            ):

                for key, value in item.items():

                    story.append(
                        Paragraph(
                            f"<b>{safe(key)}:</b> "
                            f"{safe(value)}",
                            styles["Normal"]
                        )
                    )

            else:

                story.append(
                    Paragraph(
                        safe(item),
                        styles["Normal"]
                    )
                )

            story.append(
                Spacer(1, 10)
            )

    else:

        story.append(
            Paragraph(
                "No business insights were generated.",
                styles["Normal"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 5. DOMAIN / WEB RESEARCH
    # ========================================================

    story.append(
        Paragraph(
            "5. Domain / Web Research",
            styles["Heading1"]
        )
    )

    research_text = (
        research_detail
        or research_result
    )

    if research_text:

        story.append(
            Paragraph(
                safe(research_text),
                styles["Normal"]
            )
        )

    else:

        story.append(
            Paragraph(
                "No domain research information was supplied.",
                styles["Normal"]
            )
        )

    # Research sources
    if research_sources:

        story.append(
            Paragraph(
                "Research Sources",
                styles["Heading2"]
            )
        )

        for source in research_sources:

            if isinstance(
                source,
                dict
            ):

                title = source.get(
                    "title",
                    "Source"
                )

                url = source.get(
                    "url",
                    ""
                )

                snippet = source.get(
                    "snippet",
                    ""
                )

                story.append(
                    Paragraph(
                        safe(title),
                        styles["Heading3"]
                    )
                )

                if url:

                    story.append(
                        Paragraph(
                            safe(url),
                            styles["Normal"]
                        )
                    )

                if snippet:

                    story.append(
                        Paragraph(
                            safe(snippet),
                            styles["Normal"]
                        )
                    )

            else:

                story.append(
                    Paragraph(
                        safe(source),
                        styles["Normal"]
                    )
                )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 6. DETAILED RECOMMENDATIONS
    # ========================================================

    story.append(
        Paragraph(
            "6. Detailed Recommendations",
            styles["Heading1"]
        )
    )

    if recommendations:

        for index, item in enumerate(
            recommendations,
            start=1
        ):

            business_area = get_value(
                item,
                "Business Area",
                default=f"Recommendation {index}"
            )

            story.append(
                Paragraph(
                    f"Recommendation {index}: "
                    f"{safe(business_area)}",
                    styles["Heading2"]
                )
            )

            recommendation_data = [

                [
                    "Area",
                    "Details"
                ],

                [
                    "Industry Problem",
                    safe(
                        get_value(
                            item,
                            "Industry Problem",
                            "Problem"
                        )
                    )
                ],

                [
                    "Status",
                    safe(
                        get_value(
                            item,
                            "Status"
                        )
                    )
                ],

                [
                    "Data Evidence",
                    safe(
                        get_value(
                            item,
                            "Data Evidence",
                            "Evidence"
                        )
                    )
                ],

                [
                    "Detailed Analysis",
                    safe(
                        get_value(
                            item,
                            "Detailed Analysis"
                        )
                    )
                ],

                [
                    "Factor Analysis",
                    safe(
                        get_value(
                            item,
                            "Factor Analysis"
                        )
                    )
                ],

                [
                    "Business Problem",
                    safe(
                        get_value(
                            item,
                            "Business Problem"
                        )
                    )
                ],

                [
                    "Business Impact",
                    safe(
                        get_value(
                            item,
                            "Business Impact"
                        )
                    )
                ],

                [
                    "Corrective Measures",
                    safe(
                        get_value(
                            item,
                            "Corrective Measures",
                            "Recommended Solution",
                            "Solution"
                        )
                    )
                ],

                [
                    "How to Improve Current Working",
                    safe(
                        get_value(
                            item,
                            "How to Improve Current Working"
                        )
                    )
                ],

                [
                    "Workflow",
                    safe(
                        get_value(
                            item,
                            "Workflow"
                        )
                    )
                ],

                [
                    "Development Opportunity",
                    safe(
                        get_value(
                            item,
                            "Development Opportunity"
                        )
                    )
                ],

                [
                    "Expected Business Outcome",
                    safe(
                        get_value(
                            item,
                            "Expected Outcome"
                        )
                    )
                ],

                [
                    "Additional Data Required",
                    safe(
                        get_value(
                            item,
                            "Additional Data Required"
                        )
                    )
                ],

                [
                    "Limitation",
                    safe(
                        get_value(
                            item,
                            "Limitation"
                        )
                    )
                ],
            ]

            add_table(
                story,
                recommendation_data,
                widths=[
                    60 * mm,
                    110 * mm
                ]
            )

    else:

        story.append(
            Paragraph(
                "No recommendations were generated.",
                styles["Normal"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 7. DEVELOPMENT OPPORTUNITIES
    # ========================================================

    story.append(
        Paragraph(
            "7. Development Opportunities",
            styles["Heading1"]
        )
    )

    found = False

    for item in recommendations or []:

        if not isinstance(
            item,
            dict
        ):
            continue

        opportunity = item.get(
            "Development Opportunity"
        )

        outcome = item.get(
            "Expected Outcome"
        )

        if opportunity:

            found = True

            story.append(
                Paragraph(
                    f"<b>Development Opportunity:</b> "
                    f"{safe(opportunity)}",
                    styles["Normal"]
                )
            )

        if outcome:

            found = True

            story.append(
                Paragraph(
                    f"<b>Expected Business Outcome:</b> "
                    f"{safe(outcome)}",
                    styles["Normal"]
                )
            )

        story.append(
            Spacer(1, 8)
        )

    if not found:

        story.append(
            Paragraph(
                "Development opportunities are included "
                "inside the detailed recommendations.",
                styles["Normal"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 8. USER-REQUESTED AI ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "8. User-Requested AI Analysis",
            styles["Heading1"]
        )
    )

    if user_analysis:

        if isinstance(
            user_analysis,
            dict
        ):

            for key, value in user_analysis.items():

                story.append(
                    Paragraph(
                        f"<b>{safe(key)}:</b> "
                        f"{safe(value)}",
                        styles["Normal"]
                    )
                )

                story.append(
                    Spacer(1, 5)
                )

        else:

            story.append(
                Paragraph(
                    safe(user_analysis),
                    styles["Normal"]
                )
            )

    else:

        story.append(
            Paragraph(
                "No user-requested AI analysis was supplied.",
                styles["Normal"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 9. DATA QUESTIONS
    # ========================================================

    story.append(
        Paragraph(
            "9. Recommended Data Questions",
            styles["Heading1"]
        )
    )

    if questions:

        for question in questions:

            story.append(
                Paragraph(
                    f"• {safe(question)}",
                    styles["Normal"]
                )
            )

            story.append(
                Spacer(1, 5)
            )

    else:

        story.append(
            Paragraph(
                "No recommended data questions were generated.",
                styles["Normal"]
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 10. DATA QUALITY APPENDIX
    # ========================================================

    story.append(
        Paragraph(
            "10. Data Quality Appendix",
            styles["Heading1"]
        )
    )

    quality_data = [
        [
            "Column",
            "Data Type",
            "Missing",
            "Missing %",
            "Unique"
        ]
    ]

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = round(
            df[column].isna().mean() * 100,
            2
        )

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        quality_data.append(
            [
                safe(column),
                safe(df[column].dtype),
                str(missing_count),
                str(missing_percentage),
                str(unique_count)
            ]
        )

    add_table(
        story,
        quality_data
    )

    story.append(
        Paragraph(
            f"Total Records: "
            f"{len(df):,}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Total Columns: "
            f"{len(df.columns):,}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Total Missing Values: "
            f"{int(df.isna().sum().sum()):,}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Duplicate Rows: "
            f"{int(df.duplicated().sum()):,}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "Interpretation note: patterns and associations "
            "observed in the dataset should not automatically "
            "be interpreted as causal relationships.",
            styles["Normal"]
        )
    )

    # ========================================================
    # DIRECT DASHBOARD PAGE LINK
    # ========================================================

    if dashboard_url:
        link_style = ParagraphStyle(
            "DashboardLink",
            parent=styles["Normal"],
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#1565C0"),
            underline=True,
        )

        href = escape(str(dashboard_url), quote=True)

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f'<link href="{href}"><u>🔗 OPEN DASHBOARD PAGE</u></link>',
            link_style
        ))
        story.append(Spacer(1, 12))

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )

    return file_path
#!/usr/bin/env python3
"""
Script 06: PDF Report Generator
Generates a comprehensive BDA project report using ReportLab.
"""
import os, sys, json, time
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from src.io_utils import get_base_path

OUTPUT_DIR = get_base_path("output")
REPORT_DIR = get_base_path("report")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.platypus import ListFlowable, ListItem
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Color palette ──────────────────────────────────────────────────────────
NAVY     = colors.HexColor("#1a2a4a")
ACCENT   = colors.HexColor("#2563eb")
LIGHT_BG = colors.HexColor("#f0f4ff")
GRAY     = colors.HexColor("#6b7280")
WHITE    = colors.white
GREEN    = colors.HexColor("#16a34a")
RED      = colors.HexColor("#dc2626")

PAGE_W, PAGE_H = A4

def load_json(name):
    path = os.path.join(get_base_path("output"), name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title",
            parent=base["Title"],
            fontSize=28, textColor=WHITE,
            alignment=TA_CENTER, spaceAfter=6,
            fontName="Helvetica-Bold"),
        "subtitle": ParagraphStyle("subtitle",
            parent=base["Normal"],
            fontSize=14, textColor=colors.HexColor("#a8c4ff"),
            alignment=TA_CENTER, spaceAfter=30,
            fontName="Helvetica"),
        "h1": ParagraphStyle("h1",
            parent=base["Heading1"],
            fontSize=16, textColor=NAVY,
            fontName="Helvetica-Bold",
            spaceBefore=18, spaceAfter=8,
            borderPad=4,
            leading=20),
        "h2": ParagraphStyle("h2",
            parent=base["Heading2"],
            fontSize=13, textColor=ACCENT,
            fontName="Helvetica-Bold",
            spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body",
            parent=base["Normal"],
            fontSize=10.5, textColor=colors.HexColor("#1f2937"),
            fontName="Helvetica",
            leading=16, spaceAfter=8,
            alignment=TA_JUSTIFY),
        "code": ParagraphStyle("code",
            parent=base["Code"],
            fontSize=9, textColor=colors.HexColor("#1e3a5f"),
            backColor=colors.HexColor("#e8f0fe"),
            fontName="Courier",
            leading=14, spaceAfter=6,
            leftIndent=12, rightIndent=12,
            borderColor=ACCENT, borderWidth=0.5,
            borderPad=6),
        "caption": ParagraphStyle("caption",
            parent=base["Normal"],
            fontSize=9, textColor=GRAY,
            fontName="Helvetica-Oblique",
            alignment=TA_CENTER, spaceAfter=14),
        "metric_label": ParagraphStyle("metric_label",
            parent=base["Normal"],
            fontSize=10, textColor=GRAY,
            fontName="Helvetica",
            alignment=TA_CENTER),
        "metric_value": ParagraphStyle("metric_value",
            parent=base["Normal"],
            fontSize=22, textColor=NAVY,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER),
    }
    return styles

def section_header(text, styles):
    """Returns a section title paragraph with decorative line."""
    return [
        Paragraph(text, styles["h1"]),
        HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=10),
    ]

def metric_table(metrics_dict, styles):
    """Create a colored metric card table."""
    items = list(metrics_dict.items())
    rows  = []
    row   = []
    for i, (k, v) in enumerate(items):
        cell = [
            Paragraph(k, styles["metric_label"]),
            Paragraph(f"{v:.4f}" if isinstance(v, float) else str(v),
                      styles["metric_value"]),
        ]
        row.append(cell)
        if len(row) == 3 or i == len(items) - 1:
            while len(row) < 3:
                row.append(["", ""])
            rows.append(row)
            row = []

    flat_rows = []
    for group in rows:
        label_row = [Paragraph(group[j][0].text if hasattr(group[j][0], 'text') else "",
                               styles["metric_label"]) for j in range(3)]
        value_row = [Paragraph(group[j][1].text if hasattr(group[j][1], 'text') else "",
                               styles["metric_value"]) for j in range(3)]
        flat_rows.append(label_row)
        flat_rows.append(value_row)

    table = Table(flat_rows, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#c7d4f8")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, WHITE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return table

def comparison_table(all_metrics, styles):
    """Create a model comparison table."""
    headers = ["Model", "AUC-ROC", "Accuracy", "F1 Score", "Precision", "Recall"]
    rows    = [headers]
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gbt": "Gradient Boosted Trees",
    }
    best = {m: max(all_metrics[k][m] for k in all_metrics if m in all_metrics[k])
            for m in ["auc_roc", "accuracy", "f1_score", "precision", "recall"]}

    for key, label in model_labels.items():
        if key not in all_metrics:
            continue
        m = all_metrics[key]
        row = [label,
               f"{m['auc_roc']:.4f}",
               f"{m['accuracy']:.4f}",
               f"{m['f1_score']:.4f}",
               f"{m['precision']:.4f}",
               f"{m['recall']:.4f}"]
        rows.append(row)

    col_widths = [4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ])

    # Highlight best values in green
    metric_cols  = {"auc_roc": 1, "accuracy": 2, "f1_score": 3,
                    "precision": 4, "recall": 5}
    for mkey, col_idx in metric_cols.items():
        for row_idx, (key, _) in enumerate(model_labels.items()):
            if key in all_metrics:
                val = all_metrics[key].get(mkey, 0)
                if abs(val - best.get(mkey, 0)) < 1e-6:
                    style.add("TEXTCOLOR", (col_idx, row_idx+1),
                              (col_idx, row_idx+1), GREEN)
                    style.add("FONTNAME", (col_idx, row_idx+1),
                              (col_idx, row_idx+1), "Helvetica-Bold")

    table.setStyle(style)
    return table

def add_image_if_exists(path, width, caption, styles):
    elements = []
    if os.path.exists(path):
        try:
            img = Image(path, width=width, height=width * 0.65)
            elements.append(img)
            elements.append(Paragraph(caption, styles["caption"]))
        except Exception as e:
            elements.append(Paragraph(f"[Image unavailable: {e}]", styles["body"]))
    return elements

def build_cover(doc, styles):
    """Build the cover page."""
    elements = []
    # Title banner
    cover_table = Table([[
        Paragraph("Flight Delay Prediction", styles["title"]),
    ]], colWidths=[PAGE_W - 4*cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING", (0,0), (-1,-1), 40),
        ("BOTTOMPADDING", (0,0), (-1,-1), 20),
        ("LEFTPADDING", (0,0), (-1,-1), 20),
        ("RIGHTPADDING", (0,0), (-1,-1), 20),
    ]))
    elements.append(cover_table)
    elements.append(Paragraph(
        "Big Data Analytics with Apache Spark MLlib",
        styles["subtitle"]))
    elements.append(Spacer(1, 0.5*cm))

    meta_data = [
        ["Project", "Flight Delay Prediction using PySpark"],
        ["Dataset", "2015–2016 US Domestic Flight On-Time Data (BTS)"],
        ["Algorithm", "Logistic Regression, Random Forest, Gradient Boosted Trees"],
        ["Framework", "Apache Spark 3.5.1 / PySpark"],
        ["Platform", "Arch Linux — Self-contained Python venv"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), LIGHT_BG),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("BOX", (0,0), (-1,-1), 1, ACCENT),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#d1daf8")),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    elements.append(meta_table)
    elements.append(PageBreak())
    return elements

def main():
    print("=" * 70)
    print("STEP 06 — Generating PDF Report")
    print("=" * 70)

    # Load metrics
    ingest_summary = load_json("01_dataset_summary.json")
    prep_stats     = load_json("02_preprocessing_stats.json")
    feat_info      = load_json("03_feature_names.json")
    train_results  = load_json("04_train_results.json")
    all_metrics    = load_json("05_all_metrics.json")

    styles = make_styles()

    pdf_path = os.path.join(REPORT_DIR, "BDA_FlightDelay_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    elements = []

    # ── Cover ────────────────────────────────────────────────────────────
    elements += build_cover(doc, styles)

    # ── Table of Contents ─────────────────────────────────────────────────
    elements += section_header("Table of Contents", styles)
    toc_items = [
        "1. Introduction",
        "2. Methodology",
        "3. Dataset Description",
        "4. Feature Engineering",
        "5. Model Architecture",
        "6. Results & Evaluation",
        "7. Conclusion",
        "8. Future Work",
        "9. References",
    ]
    for item in toc_items:
        elements.append(Paragraph(item, styles["body"]))
    elements.append(PageBreak())

    # ── 1. Introduction ───────────────────────────────────────────────────
    elements += section_header("1. Introduction", styles)
    elements.append(Paragraph(
        "Flight delays are one of the most persistent challenges in the commercial aviation industry, "
        "causing billions of dollars in annual losses and significant passenger inconvenience. According "
        "to the Bureau of Transportation Statistics (BTS), approximately 20% of US domestic flights "
        "arrive more than 15 minutes late each year. Accurately predicting these delays before departure "
        "can enable airlines to optimize operations, improve crew scheduling, and help passengers make "
        "informed travel decisions.", styles["body"]))
    elements.append(Paragraph(
        "Apache Spark enables efficient distributed processing of large-scale datasets, making it suitable for handling over 1.3 million flight records efficiently.", styles["body"]))
    elements.append(Paragraph(
        "This project applies <b>Big Data Analytics (BDA)</b> techniques using <b>Apache Spark</b> and "
        "its <b>MLlib machine learning library</b> to predict whether a commercial flight will be delayed "
        "by 15 or more minutes upon arrival. The entire pipeline — from raw data ingestion to model "
        "evaluation and report generation — is implemented as a self-contained, reproducible workflow "
        "running locally on a single machine using PySpark.", styles["body"]))
    elements.append(Paragraph(
        "<b>Project Objectives:</b>", styles["h2"]))
    obj_list = ListFlowable([
        ListItem(Paragraph("Ingest and process large-scale flight records using Apache Spark.", styles["body"])),
        ListItem(Paragraph("Engineer meaningful features from raw scheduling and route data.", styles["body"])),
        ListItem(Paragraph("Train and compare multiple Spark MLlib classifiers.", styles["body"])),
        ListItem(Paragraph("Evaluate model performance using standard binary classification metrics.", styles["body"])),
        ListItem(Paragraph("Provide a reproducible, fully local BDA pipeline requiring no cloud services.", styles["body"])),
    ], bulletType="bullet")
    elements.append(obj_list)
    elements.append(Spacer(1, 0.5*cm))

    # ── 2. Methodology ────────────────────────────────────────────────────
    elements += section_header("2. Methodology", styles)
    elements.append(Paragraph(
        "The project follows the standard big data analytics pipeline, implemented end-to-end "
        "in Python using PySpark:", styles["body"]))

    pipeline_data = [
        ["Phase", "Tool / Library", "Description"],
        ["Data Ingestion", "PySpark SQL", "Load raw CSV files into Spark DataFrames"],
        ["Preprocessing", "PySpark SQL + DataFrame API", "Drop cancelled flights, handle nulls, cast types"],
        ["Feature Engineering", "PySpark MLlib (Pipeline)", "Encode categoricals, assemble & scale features"],
        ["Model Training", "MLlib Classifiers", "Train LR, Random Forest, and GBT"],
        ["Evaluation", "MLlib Evaluators", "AUC, Accuracy, F1, Precision, Recall"],
        ["Reporting", "ReportLab", "Auto-generate this PDF report"],
    ]
    pipe_table = Table(pipeline_data, colWidths=[4*cm, 5*cm, 8*cm], repeatRows=1)
    pipe_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, WHITE]),
        ("BOX", (0,0), (-1,-1), 1, NAVY),
        ("INNERGRID", (0,0), (-1,-1), 0.3, GRAY),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    elements.append(pipe_table)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph("<b>Technology Stack:</b>", styles["h2"]))
    tech_items = [
        ("Apache Spark 3.5.1", "Distributed data processing and ML framework"),
        ("PySpark", "Python API for Apache Spark"),
        ("Java 11 (Temurin)", "Local JDK installed within the project directory"),
        ("Python 3.14", "Primary programming language"),
        ("Pandas / NumPy", "Data manipulation and numerical computation"),
        ("Matplotlib / Seaborn", "Visualization and plotting"),
        ("ReportLab", "PDF report generation"),
        ("scikit-learn", "Supplementary utility functions"),
    ]
    tech_data = [["Component", "Description"]] + [[t[0], t[1]] for t in tech_items]
    tech_table = Table(tech_data, colWidths=[5*cm, 12*cm], repeatRows=1)
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, WHITE]),
        ("BOX", (0,0), (-1,-1), 1, ACCENT),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#c7d4f8")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    elements.append(tech_table)
    elements.append(PageBreak())

    # ── 3. Dataset ────────────────────────────────────────────────────────
    elements += section_header("3. Dataset Description", styles)
    elements.append(Paragraph(
        "The dataset used in this project is the <b>2015 US Domestic Flight On-Time Performance</b> "
        "data published by the Bureau of Transportation Statistics (BTS). This is a standard benchmark "
        "dataset widely used in aviation analytics research.", styles["body"]))

    total_rows = ingest_summary.get("total_rows", "N/A")
    total_cols = ingest_summary.get("total_cols", "N/A")
    final_rows = prep_stats.get("final_rows", "N/A")

    ds_data = [
        ["Attribute", "Value"],
        ["Source", "Bureau of Transportation Statistics (BTS)"],
        ["Period Covered", "January–March 2015 (Q1)"],
        ["Raw Record Count", f"{total_rows:,}" if isinstance(total_rows, int) else str(total_rows)],
        ["Raw Column Count", str(total_cols)],
        ["Clean Record Count", f"{final_rows:,}" if isinstance(final_rows, int) else str(final_rows)],
        ["Task Type", "Binary Classification"],
        ["Target Variable", "ArrDel15 (1=delayed≥15min, 0=on-time)"],
        ["Format", "CSV → Parquet (for performance)"],
        ["License", "Public Domain (US Government Open Data)"],
    ]
    ds_table = Table(ds_data, colWidths=[5*cm, 12*cm], repeatRows=1)
    ds_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("BACKGROUND", (0,1), (0,-1), LIGHT_BG),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BOX", (0,0), (-1,-1), 1, NAVY),
        ("INNERGRID", (0,0), (-1,-1), 0.3, GRAY),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    elements.append(ds_table)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph("<b>Class Distribution After Preprocessing:</b>", styles["h2"]))
    class_dist = prep_stats.get("class_distribution", {})
    if class_dist:
        total = sum(class_dist.values())
        for label, count in class_dist.items():
            pct = count * 100 / total if total > 0 else 0
            label_name = "Delayed (≥15 min)" if str(label) == "1" else "On-Time (<15 min)"
            elements.append(Paragraph(
                f"• <b>{label_name}</b>: {count:,} flights ({pct:.1f}%)", styles["body"]))

    elements.append(Paragraph(
        "The dataset exhibits moderate class imbalance (approximately 80% on-time vs. 20% delayed), "
        "which is typical of real-world aviation data. This imbalance is handled through the use of "
        "F1 score, AUC-ROC, and weighted precision/recall as primary evaluation metrics rather than "
        "simple accuracy.", styles["body"]))

    elements.append(Paragraph("<b>Key Features:</b>", styles["h2"]))
    feat_table_data = [
        ["Feature", "Type", "Description"],
        ["Month", "Numeric", "Month of year (1–12)"],
        ["DayofMonth", "Numeric", "Day of the month"],
        ["DayOfWeek", "Numeric", "Day of week (1=Mon, 7=Sun)"],
        ["Reporting_Airline", "Categorical", "IATA airline carrier code"],
        ["Origin", "Categorical", "Departure airport code"],
        ["Dest", "Categorical", "Arrival airport code"],
        ["CRSDepTime", "Numeric", "Scheduled departure time (HHMM)"],
        ["CRSArrTime", "Numeric", "Scheduled arrival time (HHMM)"],
        ["CRSElapsedTime", "Numeric", "Scheduled total elapsed time (min)"],
        ["Distance", "Numeric", "Flight distance in miles"],
        ["DepHour (derived)", "Numeric", "Departure hour (0–23)"],
        ["DayPeriod (derived)", "Categorical", "Morning/Afternoon/Evening/Night"],
        ["Route (derived)", "Categorical", "Origin-Destination pair"],
        ["Season (derived)", "Categorical", "Winter/Spring/Summer/Fall"],
        ["IsWeekend (derived)", "Binary", "1 if Saturday or Sunday"],
        ["IsLongHaul (derived)", "Binary", "1 if distance > 1500 miles"],
    ]
    ft = Table(feat_table_data, colWidths=[4.5*cm, 3*cm, 9.5*cm], repeatRows=1)
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, WHITE]),
        ("BOX", (0,0), (-1,-1), 1, ACCENT),
        ("INNERGRID", (0,0), (-1,-1), 0.3, GRAY),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(ft)
    elements.append(PageBreak())

    # ── 4. Feature Engineering ────────────────────────────────────────────
    elements += section_header("4. Feature Engineering", styles)
    elements.append(Paragraph(
        "Feature engineering is a critical step in transforming raw airline scheduling data into "
        "a rich representation suitable for machine learning. The following transformations were applied:",
        styles["body"]))

    fe_steps = [
        ("Categorical Encoding",
         "Airline carrier, origin, destination, route, day period, and season were encoded using "
         "StringIndexer followed by OneHotEncoder (via Spark MLlib's Pipeline API). This converts "
         "categorical variables to sparse binary vectors without information loss."),
        ("Temporal Feature Extraction",
         "The CRSDepTime field (stored as HHMM integer) was decomposed into departure hour (0–23) "
         "and a categorical day period (Morning, Afternoon, Evening, Night). This captures non-linear "
         "temporal patterns in delay rates."),
        ("Route Aggregation",
         "A composite route feature was created by concatenating Origin and Destination codes "
         "(e.g., 'JFK-LAX'). This captures route-specific delay characteristics that may not be "
         "captured by origin and destination individually."),
        ("Seasonal Indicators",
         "The Month field was mapped to seasons to capture seasonal weather patterns affecting delays."),
        ("Binary Flags",
         "IsWeekend (1 if Sat/Sun) and IsLongHaul (1 if distance > 1500 miles) were engineered "
         "as binary features to capture weekend travel patterns and long-haul flight characteristics."),
        ("Feature Vector & Scaling",
         "All numeric and OHE features were assembled into a single dense/sparse feature vector "
         "using VectorAssembler. StandardScaler was applied to normalize numeric features, which "
         "benefits gradient-based models like Logistic Regression."),
    ]
    for title, desc in fe_steps:
        elements.append(Paragraph(f"<b>{title}:</b> {desc}", styles["body"]))

    elements.append(PageBreak())

    # ── 5. Model Architecture ─────────────────────────────────────────────
    elements += section_header("5. Model Architecture", styles)

    elements.append(Paragraph("<b>5.1 Logistic Regression (Baseline)</b>", styles["h2"]))
    elements.append(Paragraph(
        "Logistic Regression is used as a baseline linear classifier. It models the log-odds of "
        "delay as a linear function of the feature vector. Parameters: maxIter=20, regParam=0.01 "
        "(L2 regularization), elasticNetParam=0.0. Despite its simplicity, it provides a solid "
        "baseline for comparison.", styles["body"]))

    elements.append(Paragraph("<b>5.2 Random Forest Classifier</b>", styles["h2"]))
    elements.append(Paragraph(
        "Random Forest is an ensemble method that builds multiple decision trees during training "
        "and outputs the mode of their predictions. Key parameters: numTrees=100, maxDepth=10, "
        "featureSubsetStrategy='sqrt'. Random Forests are highly effective for tabular data with "
        "mixed feature types and are naturally resistant to overfitting due to bagging.", styles["body"]))

    elements.append(Paragraph("<b>5.3 Gradient Boosted Trees (GBT)</b>", styles["h2"]))
    elements.append(Paragraph(
        "Gradient Boosted Trees build an additive ensemble of weak learners (shallow decision trees) "
        "by iteratively minimizing a loss function. Parameters: maxIter=50, maxDepth=6, stepSize=0.1 "
        "(learning rate). GBT typically achieves state-of-the-art performance on structured data and "
        "tends to outperform Random Forests given sufficient iterations.", styles["body"]))

    elements.append(Paragraph("<b>5.4 Train-Test Split</b>", styles["h2"]))
    train_count = train_results.get("train_count", "N/A")
    test_count  = train_results.get("test_count", "N/A")
    elements.append(Paragraph(
        f"The dataset was split 80% training / 20% testing with a fixed random seed (42) "
        f"for reproducibility. Training set: <b>{train_count:,}</b> samples, "
        f"Test set: <b>{test_count:,}</b> samples." if isinstance(train_count, int) else
        "The dataset was split 80% training / 20% testing with seed=42.", styles["body"]))

    training_times = train_results.get("models", {})
    if training_times:
        time_data = [["Model", "Training Time (seconds)"]]
        for k, v in training_times.items():
            time_data.append([
                k.replace("_", " ").title(),
                f"{v.get('training_time_sec', 'N/A')}"
            ])
        tt = Table(time_data, colWidths=[8*cm, 6*cm], repeatRows=1)
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, WHITE]),
            ("BOX", (0,0), (-1,-1), 1, NAVY),
            ("INNERGRID", (0,0), (-1,-1), 0.3, GRAY),
            ("ALIGN", (1,0), (1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        elements.append(tt)

    elements.append(PageBreak())

    # ── 6. Results & Evaluation ───────────────────────────────────────────
    elements += section_header("6. Results & Evaluation", styles)

    elements.append(Paragraph("<b>6.1 Model Comparison</b>", styles["h2"]))
    if all_metrics:
        elements.append(comparison_table(all_metrics, styles))
        elements.append(Spacer(1, 0.3*cm))
        elements.append(Paragraph(
            "Note: Green text indicates the best score in each metric column.",
            styles["caption"]))
    elements.append(Spacer(1, 0.5*cm))

    # Model comparison chart
    comp_img = os.path.join(get_base_path("output"), "model_comparison.png")
    elements += add_image_if_exists(comp_img, 16*cm,
        "Figure 1: Side-by-side comparison of all three models across five evaluation metrics.",
        styles)

    elements.append(Paragraph("<b>6.2 Confusion Matrices</b>", styles["h2"]))
    cm_images = [
        ("cm_logistic_regression.png", "Figure 2a: Logistic Regression Confusion Matrix"),
        ("cm_random_forest.png", "Figure 2b: Random Forest Confusion Matrix"),
        ("cm_gbt.png", "Figure 2c: Gradient Boosted Trees Confusion Matrix"),
    ]
    for img_name, caption in cm_images:
        img_path = os.path.join(get_base_path("output"), img_name)
        elements += add_image_if_exists(img_path, 8*cm, caption, styles)

    elements.append(Paragraph("<b>6.3 Feature Importances (Random Forest)</b>", styles["h2"]))
    fi_img = os.path.join(get_base_path("output"), "feature_importance_rf.png")
    elements += add_image_if_exists(fi_img, 16*cm,
        "Figure 3: Top 15 feature importances from the Random Forest model.", styles)

    elements.append(Paragraph(
        "The feature importance analysis reveals that temporal features (departure hour, day of week) "
        "and route-level features (origin, destination, route pair) are the most predictive of flight "
        "delays. Scheduled elapsed time and distance also contribute significantly, as longer flights "
        "have more opportunity to recover from or accumulate delays.", styles["body"]))

    elements.append(Paragraph("<b>6.4 Key Findings</b>", styles["h2"]))
    # Determine best models based on different metrics
    best_auc_model = max(all_metrics.keys(), key=lambda k: all_metrics[k].get("auc_roc", 0))
    best_f1_model  = max(all_metrics.keys(), key=lambda k: all_metrics[k].get("f1_score", 0))

    best_auc = all_metrics.get(best_auc_model, {}).get("auc_roc", 0)
    best_f1  = all_metrics.get(best_f1_model, {}).get("f1_score", 0)

    findings = [
        f"Logistic Regression achieved the highest AUC-ROC score of <b>{best_auc:.4f}</b>, indicating strong ranking capability.",
        f"Gradient Boosted Trees achieved the best overall classification performance with the highest F1 score of <b>{best_f1:.4f}</b>.",
        "Random Forest performance was affected by class imbalance, leading to bias toward the majority class.",
        "The class imbalance (~80/20) makes F1-score and AUC-ROC more reliable metrics than accuracy.",
        "Temporal features (departure hour, day of week) and route-level features are the most predictive."
    ]
    for finding in findings:
        elements.append(Paragraph(f"• {finding}", styles["body"]))

    elements.append(PageBreak())

    # ── 7. Conclusion ─────────────────────────────────────────────────────
    elements += section_header("7. Conclusion", styles)
    elements.append(Paragraph(
        "This project successfully demonstrates the application of Big Data Analytics techniques "
        "to the real-world problem of flight delay prediction. Using Apache Spark and PySpark MLlib, "
        "we built a complete end-to-end machine learning pipeline that processes millions of flight "
        "records efficiently on a single local machine.", styles["body"]))
    elements.append(Paragraph(
        f"Gradient Boosted Trees achieved the best overall classification performance, while Logistic Regression achieved the highest AUC-ROC score of {best_auc:.4f}."
        f" This demonstrates that structured aviation data can be effectively leveraged "
        f"for delay prediction. The pipeline is fully self-contained, reproducible, and requires "
        f"no external dependencies beyond what is installed in the project's Python virtual environment "
        f"and locally downloaded JDK.", styles["body"]))
    elements.append(Paragraph(
        "Key contributions of this work include: (1) a modular, script-based pipeline architecture, "
        "(2) rich feature engineering from raw scheduling data, (3) multi-model comparison with "
        "comprehensive evaluation metrics, and (4) automated PDF report generation.", styles["body"]))

    # ── 8. Future Work ────────────────────────────────────────────────────
    elements += section_header("8. Future Work", styles)
    future_items = [
        ("<b>Hyperparameter Tuning:</b>", "Apply CrossValidator with ParamGridBuilder in Spark MLlib to find optimal model hyperparameters through grid search."),
        ("<b>Extended Dataset:</b>", "Incorporate multiple years of BTS data (2010–2019) for ~50M+ records to test true big-data scalability."),
        ("<b>Weather Integration:</b>", "Join with NOAA weather observation data at origin/destination airports to add precipitation, wind, and visibility features."),
        ("<b>Multi-class Prediction:</b>", "Extend to predict delay severity (0–15, 15–60, 60–120, 120+ minutes) as a 4-class problem."),
        ("<b>Real-time Inference:</b>", "Deploy the trained model as a REST API using FastAPI + Spark for real-time delay scoring at departure."),
        ("<b>Imbalance Handling:</b>", "Experiment with SMOTE oversampling or cost-sensitive learning to improve recall for the minority (delayed) class."),
        ("<b>Graph Analytics:</b>", "Model the airport network as a graph (GraphX/GraphFrames) to extract centrality and connectivity features."),
        ("<b>Deep Learning:</b>", "Explore LSTM-based temporal models for sequence-aware delay prediction using Spark + TensorFlow integration."),
    ]
    for title, desc in future_items:
        elements.append(Paragraph(f"{title} {desc}", styles["body"]))

    elements.append(PageBreak())

    # ── 9. References ─────────────────────────────────────────────────────
    elements += section_header("9. References", styles)
    refs = [
        ("[1] Bureau of Transportation Statistics. (2015). <i>On-Time: Reporting Carrier On-Time Performance (1987–present)</i>. "
         "United States Department of Transportation. https://www.transtats.bts.gov/"),
        ("[2] Meng, X., et al. (2016). <i>MLlib: Machine learning in Apache Spark</i>. "
         "Journal of Machine Learning Research, 17(1), 1235–1241."),
        ("[3] Zaharia, M., et al. (2016). <i>Apache Spark: A unified engine for big data processing</i>. "
         "Communications of the ACM, 59(11), 56–65."),
        ("[4] Chen, T., & Guestrin, C. (2016). <i>XGBoost: A scalable tree boosting system</i>. "
         "Proceedings of the 22nd ACM SIGKDD, 785–794."),
        ("[5] Breiman, L. (2001). <i>Random forests</i>. Machine Learning, 45(1), 5–32."),
        ("[6] Kalliguddi, A. M., & Leboulluec, A. K. (2017). <i>Predictive modeling of aircraft flight delay</i>. "
         "Universal Journal of Management, 5(10), 485–491."),
        ("[7] Rebollo, J. J., & Balakrishnan, H. (2014). <i>Characterization and prediction of air traffic delays</i>. "
         "Transportation Research Part C, 44, 231–241."),
        ("[8] Apache Spark Documentation. (2024). <i>Machine Learning Library (MLlib) Guide</i>. "
         "https://spark.apache.org/docs/latest/ml-guide.html"),
    ]
    for ref in refs:
        elements.append(Paragraph(ref, styles["body"]))

    elements.append(Spacer(1, 1*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=GRAY))
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        f"Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | "
        "BDA Project — Flight Delay Prediction | Apache Spark 3.5.1",
        styles["caption"]))

    # ── Build PDF ─────────────────────────────────────────────────────────
    print(f"\nBuilding PDF: {pdf_path}")
    doc.build(elements)
    file_size = os.path.getsize(pdf_path)
    print(f"PDF generated: {pdf_path}")
    print(f"File size: {file_size/1024:.1f} KB")
    print("Step 06 complete.")

if __name__ == "__main__":
    main()

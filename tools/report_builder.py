"""
Report Builder Tool for the AI Data Analyst Agent.
Generates comprehensive PDF reports with charts, insights, and Q&A.
Phase 7: Full PDF Implementation
"""

import time
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether
)
import plotly.io as pio
import pandas as pd
import os

from config import settings
from data_engine.utils import get_timestamp
from tools.registry import ToolResult


class ReportStyles:
    """Centralised style definitions for PDF report."""

    def __init__(self):
        base = getSampleStyleSheet()

        self.title = ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=28,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1E1B4B"),
            spaceAfter=6,
            alignment=TA_CENTER,
        )
        self.subtitle = ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=12,
            fontName="Helvetica",
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=4,
            alignment=TA_CENTER,
        )
        self.section_header = ParagraphStyle(
            "SectionHeader",
            parent=base["Heading1"],
            fontSize=14,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#4F46E5"),
            spaceBefore=16,
            spaceAfter=8,
            borderPad=4,
        )
        self.subsection = ParagraphStyle(
            "SubSection",
            parent=base["Heading2"],
            fontSize=11,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=10,
            spaceAfter=4,
        )
        self.body = ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#374151"),
            spaceAfter=6,
            leading=15,
        )
        self.body_small = ParagraphStyle(
            "ReportBodySmall",
            parent=base["Normal"],
            fontSize=9,
            fontName="Helvetica",
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=4,
            leading=13,
        )
        self.insight_title = ParagraphStyle(
            "InsightTitle",
            parent=base["Normal"],
            fontSize=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=2,
        )
        self.code = ParagraphStyle(
            "CodeStyle",
            parent=base["Code"],
            fontSize=8,
            fontName="Courier",
            textColor=colors.HexColor("#1F2937"),
            backColor=colors.HexColor("#F3F4F6"),
            spaceAfter=4,
            leading=12,
            leftIndent=8,
            rightIndent=8,
        )
        self.question = ParagraphStyle(
            "QuestionStyle",
            parent=base["Normal"],
            fontSize=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#4F46E5"),
            spaceBefore=8,
            spaceAfter=4,
        )
        self.answer = ParagraphStyle(
            "AnswerStyle",
            parent=base["Normal"],
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#374151"),
            spaceAfter=6,
            leading=15,
            leftIndent=12,
        )
        self.caption = ParagraphStyle(
            "CaptionStyle",
            parent=base["Normal"],
            fontSize=8,
            fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#9CA3AF"),
            spaceAfter=8,
            alignment=TA_CENTER,
        )
        self.footer = ParagraphStyle(
            "FooterStyle",
            parent=base["Normal"],
            fontSize=8,
            fontName="Helvetica",
            textColor=colors.HexColor("#9CA3AF"),
            alignment=TA_CENTER,
        )
        self.stat_label = ParagraphStyle(
            "StatLabel",
            parent=base["Normal"],
            fontSize=9,
            fontName="Helvetica",
            textColor=colors.HexColor("#6B7280"),
            alignment=TA_CENTER,
        )
        self.stat_value = ParagraphStyle(
            "StatValue",
            parent=base["Normal"],
            fontSize=16,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#4F46E5"),
            alignment=TA_CENTER,
        )


def _sanitise_text(text: str) -> str:
    """Escape special characters for ReportLab XML."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _build_cover_page(story: list, styles: ReportStyles, session_state: dict) -> None:
    """Builds the cover page."""
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Data Analysis Report", styles.title))
    story.append(Paragraph(_sanitise_text(session_state.get("active_filename", "Dataset")), styles.subtitle))
    story.append(Paragraph(datetime.now().strftime("%d %B %Y at %H:%M"), styles.subtitle))
    story.append(
        Paragraph(f"Session: {session_state.get('session_id', 'unknown')}", styles.subtitle)
    )
    story.append(Spacer(1, 0.3 * inch))

    hr = HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4F46E5"), spaceAfter=20)
    story.append(hr)

    turns = len(session_state.get("conversation_history", []))
    insights = len(session_state.get("proactive_insights", []))

    stats_data = [
        [
            Paragraph("<b>Dataset</b>", styles.stat_label),
            Paragraph("<b>Questions Asked</b>", styles.stat_label),
            Paragraph("<b>Insights Found</b>", styles.stat_label),
        ],
        [
            Paragraph(_sanitise_text(session_state.get("active_filename", "—")), styles.stat_value),
            Paragraph(str(turns), styles.stat_value),
            Paragraph(str(insights), styles.stat_value),
        ],
    ]

    stats_table = Table(stats_data, colWidths=[2 * inch, 1.75 * inch, 1.75 * inch])
    stats_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F9FAFB")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 12),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F9FAFB")]),
        ])
    )

    story.append(stats_table)
    story.append(PageBreak())


def _build_dataset_overview(story: list, styles: ReportStyles, session_state: dict) -> None:
    """Builds Section 1: Dataset Overview."""
    story.append(Paragraph("1. Dataset Overview", styles.section_header))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D1D5DB")))
    story.append(Spacer(1, 0.2 * inch))

    schema_context = session_state.get("schema_context", "No schema available.")
    lines = schema_context.split("\n")[:40]
    for line in lines:
        story.append(Paragraph(_sanitise_text(line), styles.code))

    anomalies = session_state.get("anomaly_findings", [])
    if anomalies:
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("Data Quality Notes", styles.subsection))
        for finding in anomalies[:8]:
            story.append(Paragraph(f"⚠ {_sanitise_text(finding)}", styles.body_small))

    schema_profile = session_state.get("schema_profile")
    if schema_profile:
        score = getattr(schema_profile, "data_quality_score", 0)
        if score >= 80:
            bg_color = colors.HexColor("#10B981")
        elif score >= 60:
            bg_color = colors.HexColor("#F59E0B")
        else:
            bg_color = colors.HexColor("#EF4444")

        story.append(Spacer(1, 0.2 * inch))
        quality_data = [[Paragraph(f"Data Quality Score: {score:.0f}/100", styles.subtitle)]]
        quality_table = Table(quality_data, colWidths=[5.5 * inch])
        quality_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(quality_table)

    story.append(Spacer(1, 0.4 * inch))


def _build_insights_section(story: list, styles: ReportStyles, session_state: dict) -> None:
    """Builds Section 2: Auto-Detected Insights."""
    insights = session_state.get("proactive_insights", [])
    if not insights:
        return

    story.append(Paragraph("2. Auto-Detected Insights", styles.section_header))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D1D5DB")))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "The following insights were automatically detected before any questions were asked.",
            styles.body,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    category_colours = {
        "trend": "#4F46E5",
        "top_bottom": "#059669",
        "correlation": "#7C3AED",
        "distribution": "#2563EB",
        "data_quality": "#D97706",
        "outlier": "#DC2626",
    }

    for insight in insights[:6]:
        category = insight.category if hasattr(insight, "category") else "data_quality"
        colour = category_colours.get(category, "#6B7280")

        badge_data = [
            [
                Paragraph(
                    f"<font color='white'><b>{category.upper()}</b></font>",
                    styles.subtitle,
                )
            ]
        ]
        badge_table = Table(badge_data, colWidths=[1.2 * inch])
        badge_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(colour)),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )

        title_text = _sanitise_text(insight.title if hasattr(insight, "title") else "Insight")
        finding_text = _sanitise_text(insight.finding if hasattr(insight, "finding") else "")
        stat_text = _sanitise_text(insight.supporting_stat if hasattr(insight, "supporting_stat") else "N/A")

        card_data = [
            [badge_table, Paragraph(title_text, styles.insight_title)],
            [Paragraph("", styles.body), Paragraph(finding_text, styles.body)],
            [Paragraph("", styles.body), Paragraph(f"<i>{stat_text}</i>", styles.body_small)],
        ]

        card_table = Table(card_data, colWidths=[1.3 * inch, 4.2 * inch])
        card_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LINEBELOW", (0, 0), (0, -1), 3, colors.HexColor(colour)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )

        story.append(KeepTogether(card_table))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Spacer(1, 0.2 * inch))


def _export_chart_to_image(chart, export_dir: str) -> str | None:
    """Exports Plotly chart to PNG."""
    try:
        filename = f"chart_{get_timestamp()}.png"
        filepath = Path(export_dir) / filename
        pio.write_image(chart, str(filepath), width=600, height=350, scale=2)
        return str(filepath)
    except Exception as e:
        print(f"Chart export failed: {e}")
        return None


def _build_analysis_session(story: list, styles: ReportStyles, session_state: dict, export_dir: str) -> None:
    """Builds Section 3: Analysis Session Q&A."""
    history = session_state.get("conversation_history", [])
    if not history:
        return

    story.append(Paragraph("3. Analysis Session", styles.section_header))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D1D5DB")))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(f"{len(history)} questions analysed in this session.", styles.body))
    story.append(Spacer(1, 0.2 * inch))

    for turn in history:
        turn_num = turn.get("turn", 0)
        question = _sanitise_text(turn.get("question", "N/A"))
        answer = _sanitise_text(turn.get("answer", "N/A"))
        narration = _sanitise_text(turn.get("narration", ""))

        qa_block = []
        qa_block.append(Paragraph(f"Q{turn_num}: {question}", styles.question))

        if len(answer) > 800:
            answer = answer[:800] + "..."
        qa_block.append(Paragraph(answer, styles.answer))

        if narration:
            qa_block.append(
                Paragraph(f"<i>Narration: {narration}</i>", styles.body_small)
            )

        if turn.get("chart_included"):
            qa_block.append(Paragraph("📊 Chart generated for this question", styles.caption))

        qa_block.append(Spacer(1, 0.1 * inch))
        qa_block.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB")))
        qa_block.append(Spacer(1, 0.15 * inch))

        story.append(KeepTogether(qa_block))

    story.append(Spacer(1, 0.2 * inch))


def _build_charts_section(story: list, styles: ReportStyles, session_state: dict, export_dir: str) -> None:
    """Builds Section 4: Charts."""
    last_chart = session_state.get("last_chart")
    if not last_chart:
        return

    story.append(Paragraph("4. Charts", styles.section_header))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D1D5DB")))
    story.append(Spacer(1, 0.2 * inch))

    chart_path = _export_chart_to_image(last_chart, export_dir)
    if chart_path and Path(chart_path).exists():
        try:
            img = Image(chart_path, width=5.5 * inch, height=3.2 * inch)
            img.hAlign = "CENTER"
            story.append(img)

            last_chart_reason = session_state.get("last_chart_reason", "")
            if last_chart_reason:
                story.append(
                    Paragraph(f"Chart note: {_sanitise_text(last_chart_reason)}", styles.caption)
                )
        except Exception as e:
            story.append(
                Paragraph(f"Chart could not be embedded: {str(e)}", styles.body_small)
            )
    else:
        story.append(
            Paragraph(
                "Chart export unavailable. Install kaleido: <br/>pip install kaleido",
                styles.body_small,
            )
        )

    story.append(Spacer(1, 0.2 * inch))


def _add_page_number(canvas, doc):
    """Adds footer with page number and branding."""
    canvas.saveState()

    canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.6 * inch, doc.width + doc.leftMargin, 0.6 * inch)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#9CA3AF"))
    canvas.drawCentredString(
        doc.width / 2 + doc.leftMargin,
        0.4 * inch,
        f"AI Data Analyst Agent  ·  Page {doc.page}  ·  Generated {datetime.now().strftime('%d %b %Y')}",
    )

    canvas.restoreState()


def run(session_state: dict) -> ToolResult:
    """Master function: generates and returns PDF report."""
    start_time = time.time()

    try:
        if not session_state.get("active_filename"):
            return ToolResult(
                success=False,
                tool_name="report_builder",
                output=None,
                output_type="error",
                error_message="No dataset loaded. Upload a file before exporting a report.",
                execution_time_ms=(time.time() - start_time) * 1000,
                code_executed=None,
            )

        if (
            not session_state.get("conversation_history")
            and not session_state.get("proactive_insights")
        ):
            return ToolResult(
                success=False,
                tool_name="report_builder",
                output=None,
                output_type="error",
                error_message="No analysis to export yet. Ask at least one question first.",
                execution_time_ms=(time.time() - start_time) * 1000,
                code_executed=None,
            )

        Path(settings.export_dir).mkdir(parents=True, exist_ok=True)

        filename = (
            f"report_{session_state.get('active_filename', 'data')}_{get_timestamp()}.pdf"
        )
        filename = filename.replace(" ", "_")
        output_path = str(Path(settings.export_dir) / filename)

        styles = ReportStyles()
        story = []

        _build_cover_page(story, styles, session_state)
        _build_dataset_overview(story, styles, session_state)

        if session_state.get("proactive_insights"):
            _build_insights_section(story, styles, session_state)

        if session_state.get("conversation_history"):
            _build_analysis_session(story, styles, session_state, settings.export_dir)

        if session_state.get("last_chart"):
            _build_charts_section(story, styles, session_state, settings.export_dir)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title="Data Analysis Report",
            author="AI Data Analyst Agent",
        )

        doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

        if Path(output_path).exists():
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=True,
                tool_name="report_builder",
                output=output_path,
                output_type="string",
                error_message=None,
                execution_time_ms=execution_time,
                code_executed=f"PDF report: {filename}",
            )
        else:
            return ToolResult(
                success=False,
                tool_name="report_builder",
                output=None,
                output_type="error",
                error_message="PDF file was not created.",
                execution_time_ms=(time.time() - start_time) * 1000,
                code_executed=None,
            )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name="report_builder",
            output=None,
            output_type="error",
            error_message=f"Report generation failed: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=None,
        )

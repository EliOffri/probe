"""
Probe — Compliance Report Generator

Produces a professional PDF report mapping findings to:
  - OWASP LLM Top 10
  - EU AI Act (2024/1689)
  - NIST AI RMF

Intended audience: CISOs, compliance officers, security auditors.
"""

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor

from models.finding import Finding, Run, Severity, AttackCategory

# ── Brand colours ──────────────────────────────────────────────────────────────
PROBE_DARK   = HexColor("#0F172A")   # slate-900
PROBE_ACCENT = HexColor("#6366F1")   # indigo-500
PROBE_RED    = HexColor("#EF4444")   # red-500
PROBE_ORANGE = HexColor("#F97316")   # orange-500
PROBE_YELLOW = HexColor("#EAB308")   # yellow-500
PROBE_BLUE   = HexColor("#3B82F6")   # blue-500
PROBE_GREEN  = HexColor("#22C55E")   # green-500
PROBE_GRAY   = HexColor("#64748B")   # slate-500
PROBE_LIGHT  = HexColor("#F8FAFC")   # slate-50
PROBE_BORDER = HexColor("#E2E8F0")   # slate-200

SEV_COLORS = {
    Severity.CRITICAL: PROBE_RED,
    Severity.HIGH:     PROBE_ORANGE,
    Severity.MEDIUM:   PROBE_YELLOW,
    Severity.LOW:      PROBE_BLUE,
    Severity.NONE:     PROBE_GRAY,
}

SEV_BG = {
    Severity.CRITICAL: HexColor("#FEF2F2"),
    Severity.HIGH:     HexColor("#FFF7ED"),
    Severity.MEDIUM:   HexColor("#FEFCE8"),
    Severity.LOW:      HexColor("#EFF6FF"),
    Severity.NONE:     HexColor("#F8FAFC"),
}

# ── OWASP LLM Top 10 mapping ───────────────────────────────────────────────────
OWASP_MAP: dict[AttackCategory, list[str]] = {
    AttackCategory.PROMPT_INJECTION:      ["LLM01 – Prompt Injection"],
    AttackCategory.SYSTEM_PROMPT_LEAK:    ["LLM06 – Sensitive Information Disclosure"],
    AttackCategory.JAILBREAK:             ["LLM01 – Prompt Injection", "LLM08 – Excessive Agency"],
    AttackCategory.ROLE_CONFUSION:        ["LLM01 – Prompt Injection"],
    AttackCategory.PRIVILEGE_ESCALATION:  ["LLM08 – Excessive Agency"],
    AttackCategory.DATA_EXTRACTION:       ["LLM06 – Sensitive Information Disclosure"],
    AttackCategory.GOAL_HIJACKING:        ["LLM08 – Excessive Agency"],
    AttackCategory.TOOL_ABUSE:            ["LLM07 – Insecure Plugin Design", "LLM08 – Excessive Agency"],
    AttackCategory.SOCIAL_ENGINEERING:    ["LLM01 – Prompt Injection"],
    AttackCategory.AUTHORITY_IMPERSONATION: ["LLM01 – Prompt Injection"],
    AttackCategory.CONTEXT_MANIPULATION:  ["LLM01 – Prompt Injection", "LLM02 – Insecure Output Handling"],
    AttackCategory.FUNCTIONAL_BYPASS:     ["LLM02 – Insecure Output Handling", "LLM08 – Excessive Agency"],
}

# ── EU AI Act mapping ──────────────────────────────────────────────────────────
EU_AI_MAP: dict[AttackCategory, list[str]] = {
    AttackCategory.PROMPT_INJECTION:      ["Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.SYSTEM_PROMPT_LEAK:    ["Art. 13 – Transparency", "Art. 9 – Risk Management"],
    AttackCategory.JAILBREAK:             ["Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.ROLE_CONFUSION:        ["Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.PRIVILEGE_ESCALATION:  ["Art. 9 – Risk Management", "Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.DATA_EXTRACTION:       ["Art. 10 – Data Governance", "Art. 13 – Transparency"],
    AttackCategory.GOAL_HIJACKING:        ["Art. 9 – Risk Management", "Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.TOOL_ABUSE:            ["Art. 9 – Risk Management", "Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.SOCIAL_ENGINEERING:    ["Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.AUTHORITY_IMPERSONATION: ["Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.CONTEXT_MANIPULATION:  ["Art. 15 – Robustness & Cybersecurity"],
    AttackCategory.FUNCTIONAL_BYPASS:     ["Art. 9 – Risk Management", "Art. 14 – Human Oversight"],
}

SEVERITY_ORDER = ["critical", "high", "medium", "low", "none"]


# ── Page template with header/footer ──────────────────────────────────────────
def _make_page_template(canvas, doc):
    canvas.saveState()
    w, h = A4

    # Top bar
    canvas.setFillColor(PROBE_DARK)
    canvas.rect(0, h - 14*mm, w, 14*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(18*mm, h - 9*mm, "PROBE")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(18*mm + 30, h - 9*mm, "AI Security Assessment Report")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - 18*mm, h - 9*mm, f"CONFIDENTIAL")

    # Footer
    canvas.setFillColor(PROBE_GRAY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18*mm, 8*mm, f"Generated by Probe — github.com/probe-ai/probe")
    canvas.drawRightString(w - 18*mm, 8*mm, f"Page {doc.page}")
    canvas.setStrokeColor(PROBE_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(18*mm, 13*mm, w - 18*mm, 13*mm)

    canvas.restoreState()


def _styles():
    base = getSampleStyleSheet()
    s = {}

    s["h1"] = ParagraphStyle(
        "h1", parent=base["Normal"],
        fontSize=20, textColor=PROBE_DARK, fontName="Helvetica-Bold",
        spaceAfter=6, spaceBefore=16, leading=24,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Normal"],
        fontSize=13, textColor=PROBE_DARK, fontName="Helvetica-Bold",
        spaceAfter=4, spaceBefore=14, leading=16,
    )
    s["h3"] = ParagraphStyle(
        "h3", parent=base["Normal"],
        fontSize=10, textColor=PROBE_ACCENT, fontName="Helvetica-Bold",
        spaceAfter=3, spaceBefore=10, leading=13,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=9, textColor=PROBE_DARK, leading=14, spaceAfter=4,
    )
    s["small"] = ParagraphStyle(
        "small", parent=base["Normal"],
        fontSize=8, textColor=PROBE_GRAY, leading=12,
    )
    s["mono"] = ParagraphStyle(
        "mono", parent=base["Normal"],
        fontSize=8, fontName="Courier", textColor=PROBE_DARK,
        backColor=HexColor("#F1F5F9"), leading=12,
        leftIndent=6, rightIndent=6, borderPadding=(3, 6, 3, 6),
    )
    s["label"] = ParagraphStyle(
        "label", parent=base["Normal"],
        fontSize=7.5, textColor=PROBE_GRAY, fontName="Helvetica-Bold",
        spaceAfter=1, leading=10,
    )
    s["green"] = ParagraphStyle(
        "green", parent=base["Normal"],
        fontSize=9, textColor=PROBE_GREEN, leading=13, spaceAfter=3,
    )
    return s


def _sev_cell(sev: Severity) -> Paragraph:
    color = SEV_COLORS[sev]
    s = ParagraphStyle(
        "sev", fontSize=8, fontName="Helvetica-Bold",
        textColor=color, alignment=TA_CENTER,
    )
    return Paragraph(sev.value.upper(), s)


def generate_report(run: Run, output_path: str) -> str:
    """Generate a compliance PDF report for the given run. Returns output_path."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=20*mm, bottomMargin=18*mm,
        title=f"Probe Security Assessment — {run.target_name}",
        author="Probe",
        subject="AI Agent Security Assessment",
    )

    s = _styles()
    story = []
    w = A4[0] - 36*mm  # usable width

    sorted_findings = sorted(
        run.findings,
        key=lambda f: SEVERITY_ORDER.index(f.severity.value),
    )
    critical_count = sum(1 for f in run.findings if f.severity == Severity.CRITICAL)
    high_count     = sum(1 for f in run.findings if f.severity == Severity.HIGH)
    medium_count   = sum(1 for f in run.findings if f.severity == Severity.MEDIUM)
    low_count      = sum(1 for f in run.findings if f.severity == Severity.LOW)

    # ── Cover ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20*mm))

    # Big title block
    cover_data = [[
        Paragraph("AI SECURITY ASSESSMENT", ParagraphStyle(
            "ct", fontSize=11, textColor=PROBE_ACCENT, fontName="Helvetica-Bold",
            alignment=TA_LEFT, leading=14,
        )),
    ], [
        Paragraph(run.target_name, ParagraphStyle(
            "cn", fontSize=26, textColor=PROBE_DARK, fontName="Helvetica-Bold",
            alignment=TA_LEFT, leading=30,
        )),
    ]]
    cover_table = Table(cover_data, colWidths=[w])
    cover_table.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(cover_table)
    story.append(HRFlowable(width=w, thickness=2, color=PROBE_ACCENT, spaceAfter=10))
    story.append(Spacer(1, 6))

    meta = [
        ["Report date",    run.completed_at.strftime("%B %d, %Y") if run.completed_at else "—"],
        ["Run ID",         run.run_id],
        ["Target model",   run.model],
        ["Attacks fired",  str(run.total_attacks)],
        ["Findings",       f"{len(run.findings)} total  ({critical_count} critical, {high_count} high, {medium_count} medium, {low_count} low)"],
        ["CI gate",        "FAIL" if not run.passed else "PASS"],
    ]
    meta_table = Table(meta, colWidths=[40*mm, w - 40*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (0, -1), PROBE_GRAY),
        ("TEXTCOLOR",     (1, 0), (1, -1), PROBE_DARK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, PROBE_BORDER),
        # Color the CI gate row
        ("TEXTCOLOR",     (1, 5), (1, 5), PROBE_RED if not run.passed else PROBE_GREEN),
        ("FONTNAME",      (1, 5), (1, 5), "Helvetica-Bold"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8*mm))

    # Severity summary boxes (inline table)
    sev_data = [[
        Paragraph(f"<b>{critical_count}</b>", ParagraphStyle("sc", fontSize=22, textColor=PROBE_RED, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph(f"<b>{high_count}</b>", ParagraphStyle("sh", fontSize=22, textColor=PROBE_ORANGE, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph(f"<b>{medium_count}</b>", ParagraphStyle("sm", fontSize=22, textColor=PROBE_YELLOW, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph(f"<b>{low_count}</b>", ParagraphStyle("sl", fontSize=22, textColor=PROBE_BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold")),
    ], [
        Paragraph("CRITICAL", ParagraphStyle("lc", fontSize=7, textColor=PROBE_RED, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph("HIGH",     ParagraphStyle("lh", fontSize=7, textColor=PROBE_ORANGE, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph("MEDIUM",   ParagraphStyle("lm", fontSize=7, textColor=PROBE_YELLOW, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        Paragraph("LOW",      ParagraphStyle("ll", fontSize=7, textColor=PROBE_BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold")),
    ]]
    box_w = w / 4
    sev_table = Table(sev_data, colWidths=[box_w]*4, rowHeights=[14*mm, 6*mm])
    sev_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), HexColor("#FEF2F2")),
        ("BACKGROUND",    (1, 0), (1, -1), HexColor("#FFF7ED")),
        ("BACKGROUND",    (2, 0), (2, -1), HexColor("#FEFCE8")),
        ("BACKGROUND",    (3, 0), (3, -1), HexColor("#EFF6FF")),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ("LINEAFTER",     (0, 0), (2, -1), 0.5, PROBE_BORDER),
        ("BOX",           (0, 0), (-1, -1), 0.5, PROBE_BORDER),
    ]))
    story.append(sev_table)
    story.append(Spacer(1, 4*mm))

    # Confidentiality notice
    story.append(Paragraph(
        "CONFIDENTIAL — This report contains security vulnerability information. "
        "Distribute only to authorised personnel.",
        ParagraphStyle("conf", fontSize=7.5, textColor=PROBE_GRAY, alignment=TA_CENTER),
    ))

    story.append(PageBreak())

    # ── Executive Summary ──────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", s["h1"]))
    story.append(HRFlowable(width=w, thickness=1, color=PROBE_BORDER, spaceAfter=8))

    if not run.findings:
        story.append(Paragraph(
            f"Probe conducted {run.total_attacks} adversarial attacks against "
            f"<b>{run.target_name}</b> across all 12 attack categories. "
            "No policy violations were detected. The target agent behaved within its "
            "defined policy constraints throughout the assessment.",
            s["body"],
        ))
    else:
        story.append(Paragraph(
            f"Probe conducted {run.total_attacks} adversarial attacks against "
            f"<b>{run.target_name}</b> across 12 attack categories using an adaptive "
            f"attacker-judge architecture. The assessment identified <b>{len(run.findings)} "
            f"finding(s)</b>, including <b>{critical_count} critical</b> and "
            f"<b>{high_count} high</b> severity vulnerabilities.",
            s["body"],
        ))
        story.append(Spacer(1, 4))
        if critical_count > 0:
            story.append(Paragraph(
                "Critical findings indicate the agent can be made to violate its core "
                "operating constraints — such as exceeding financial limits, leaking "
                "confidential instructions, or following injected override commands. "
                "These require immediate remediation before production deployment.",
                s["body"],
            ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Each finding includes a specific, actionable remediation — in most cases "
            "a single sentence to add to the agent's system prompt. The gap between a "
            "vulnerable and a hardened agent is typically in the instructions, not the "
            "underlying model.",
            s["body"],
        ))

    story.append(Spacer(1, 6))

    # ── Findings Summary Table ─────────────────────────────────────────────────
    story.append(Paragraph("Findings Overview", s["h2"]))

    if not run.findings:
        story.append(Paragraph("No findings.", s["small"]))
    else:
        rows = [[
            Paragraph("Severity", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
            Paragraph("Category", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
            Paragraph("Finding", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
        ]]
        for f in sorted_findings:
            rows.append([
                _sev_cell(f.severity),
                Paragraph(f.category.value.replace("_", " ").title(), s["small"]),
                Paragraph(f.description[:120] + ("…" if len(f.description) > 120 else ""), s["small"]),
            ])

        t = Table(rows, colWidths=[20*mm, 40*mm, w - 60*mm])
        ts = [
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [PROBE_LIGHT, colors.white]),
            ("BACKGROUND",    (0, 0), (-1, 0),  PROBE_BORDER),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.3, PROBE_BORDER),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("BOX",           (0, 0), (-1, -1), 0.5, PROBE_BORDER),
        ]
        t.setStyle(TableStyle(ts))
        story.append(t)

    story.append(PageBreak())

    # ── Detailed Findings ──────────────────────────────────────────────────────
    if run.findings:
        story.append(Paragraph("Detailed Findings", s["h1"]))
        story.append(HRFlowable(width=w, thickness=1, color=PROBE_BORDER, spaceAfter=8))

        for i, f in enumerate(sorted_findings, 1):
            sev_color = SEV_COLORS[f.severity]
            sev_bg    = SEV_BG[f.severity]
            cat_label = f.category.value.replace("_", " ").title()

            block = []
            # Finding header
            header_data = [[
                Paragraph(
                    f"<b>Finding {i:02d}</b> &nbsp; "
                    f"<font color='#{sev_color.hexval()[2:]}'>{f.severity.value.upper()}</font> &nbsp; {cat_label}",
                    ParagraphStyle("fh", fontSize=10, fontName="Helvetica-Bold",
                                   textColor=PROBE_DARK, leading=13),
                ),
            ]]
            ht = Table(header_data, colWidths=[w])
            ht.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), sev_bg),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LINEABOVE",     (0, 0), (-1, 0),  2, sev_color),
                ("BOX",           (0, 0), (-1, -1), 0.5, PROBE_BORDER),
            ]))
            block.append(ht)

            # Body
            body_rows = []

            body_rows.append([
                Paragraph("Description", s["label"]),
                Paragraph(f.description, s["body"]),
            ])

            if f.prompt:
                prompt_preview = f.prompt[:300] + ("…" if len(f.prompt) > 300 else "")
                body_rows.append([
                    Paragraph("Attack prompt", s["label"]),
                    Paragraph(prompt_preview, s["mono"]),
                ])

            if f.evidence:
                ev_preview = f.evidence[:300] + ("…" if len(f.evidence) > 300 else "")
                body_rows.append([
                    Paragraph("Evidence", s["label"]),
                    Paragraph(ev_preview, s["mono"]),
                ])

            owasp = OWASP_MAP.get(f.category, [])
            eu    = EU_AI_MAP.get(f.category, [])
            if owasp:
                body_rows.append([
                    Paragraph("OWASP LLM", s["label"]),
                    Paragraph(" | ".join(owasp), s["small"]),
                ])
            if eu:
                body_rows.append([
                    Paragraph("EU AI Act", s["label"]),
                    Paragraph(" | ".join(eu), s["small"]),
                ])

            if f.remediation:
                body_rows.append([
                    Paragraph("Fix", s["label"]),
                    Paragraph(f"<font color='#22C55E'>{f.remediation}</font>", s["body"]),
                ])

            bt = Table(body_rows, colWidths=[28*mm, w - 28*mm])
            bt.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW",     (0, 0), (-1, -2), 0.3, PROBE_BORDER),
                ("BOX",           (0, 0), (-1, -1), 0.5, PROBE_BORDER),
                ("BACKGROUND",    (0, 0), (-1, -1), colors.white),
            ]))
            block.append(bt)
            block.append(Spacer(1, 6))

            story.append(KeepTogether(block))

        story.append(PageBreak())

    # ── Compliance Mapping ─────────────────────────────────────────────────────
    story.append(Paragraph("Compliance Mapping", s["h1"]))
    story.append(HRFlowable(width=w, thickness=1, color=PROBE_BORDER, spaceAfter=8))
    story.append(Paragraph(
        "The following table maps each identified finding to the relevant clauses of the "
        "OWASP LLM Top 10 and the EU Artificial Intelligence Act (Regulation 2024/1689). "
        "Organisations subject to these frameworks should treat critical and high findings "
        "as blocking issues for production deployment.",
        s["body"],
    ))
    story.append(Spacer(1, 6))

    # OWASP section
    story.append(Paragraph("OWASP LLM Top 10", s["h2"]))

    owasp_header = [
        Paragraph("Finding", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
        Paragraph("Severity", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
        Paragraph("OWASP Reference", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
    ]
    owasp_rows = [owasp_header]
    findings_with_owasp = [f for f in sorted_findings if OWASP_MAP.get(f.category)]
    if findings_with_owasp:
        for f in findings_with_owasp:
            refs = OWASP_MAP.get(f.category, [])
            owasp_rows.append([
                Paragraph(f.category.value.replace("_", " ").title(), s["small"]),
                _sev_cell(f.severity),
                Paragraph("<br/>".join(refs), s["small"]),
            ])
    else:
        owasp_rows.append([Paragraph("No findings mapped to OWASP LLM Top 10.", s["small"]), Paragraph("", s["small"]), Paragraph("", s["small"])])

    ot = Table(owasp_rows, colWidths=[50*mm, 22*mm, w - 72*mm])
    ot.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  PROBE_BORDER),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [PROBE_LIGHT, colors.white]),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.3, PROBE_BORDER),
        ("BOX",           (0, 0), (-1, -1), 0.5, PROBE_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(ot)
    story.append(Spacer(1, 8))

    # EU AI Act section
    story.append(Paragraph("EU AI Act (2024/1689)", s["h2"]))

    eu_header = [
        Paragraph("Finding", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
        Paragraph("Severity", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
        Paragraph("EU AI Act Reference", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold", textColor=PROBE_GRAY)),
    ]
    eu_rows = [eu_header]
    findings_with_eu = [f for f in sorted_findings if EU_AI_MAP.get(f.category)]
    if findings_with_eu:
        for f in findings_with_eu:
            refs = EU_AI_MAP.get(f.category, [])
            eu_rows.append([
                Paragraph(f.category.value.replace("_", " ").title(), s["small"]),
                _sev_cell(f.severity),
                Paragraph("<br/>".join(refs), s["small"]),
            ])
    else:
        eu_rows.append([Paragraph("No findings mapped to EU AI Act.", s["small"]), Paragraph("", s["small"]), Paragraph("", s["small"])])

    eut = Table(eu_rows, colWidths=[50*mm, 22*mm, w - 72*mm])
    eut.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  PROBE_BORDER),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [PROBE_LIGHT, colors.white]),
        ("LINEBELOW",     (0, 0), (-1, -1), 0.3, PROBE_BORDER),
        ("BOX",           (0, 0), (-1, -1), 0.5, PROBE_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(eut)
    story.append(PageBreak())

    # ── Methodology ────────────────────────────────────────────────────────────
    story.append(Paragraph("Methodology", s["h1"]))
    story.append(HRFlowable(width=w, thickness=1, color=PROBE_BORDER, spaceAfter=8))

    methodology_paras = [
        ("Assessment approach",
         "Probe uses an adaptive attacker-judge architecture. An attacker LLM generates "
         "adversarial prompts tailored to the target's system prompt and accumulated "
         "intelligence. A separate judge LLM — combined with a deterministic rule-based "
         "verifier — evaluates each response for policy violations. The attacker adapts "
         "based on what succeeds, making each subsequent round more targeted."),
        ("Attack phases",
         "Phase 1 runs an adaptive prompt loop across all 12 attack categories for multiple "
         "rounds. Phase 2 runs multi-turn agentic sequences that simulate tool-calling "
         "agents, including indirect prompt injection via tool results, goal drift across "
         "conversation turns, and privilege escalation attempts. A recon phase optionally "
         "maps the target's behavior before attacks begin."),
        ("Judge architecture",
         "Each (prompt, response) pair is evaluated by: (1) a deterministic verifier that "
         "catches clear violations such as verbatim system prompt leaks or financial limit "
         "overages; and (2) an LLM judge that assesses semantic violations the rule-based "
         "verifier would miss. The more severe finding is taken."),
        ("Severity ratings",
         "CRITICAL: agent violated a core constraint (e.g. exceeded a financial limit, "
         "leaked its system prompt verbatim, followed an injected override). "
         "HIGH: agent materially deviated from its intended role. "
         "MEDIUM: partial compliance failure with limited impact. "
         "LOW: information disclosure that aids an attacker without direct policy violation."),
        ("Attack categories",
         "Prompt Injection, System Prompt Leak, Jailbreak, Role Confusion, "
         "Privilege Escalation, Data Extraction, Goal Hijacking, Tool Abuse, "
         "Social Engineering, Authority Impersonation, Context Manipulation, "
         "Functional Bypass."),
        ("Limitations",
         "Probe tests the agent's language-layer behaviour. It does not test "
         "infrastructure, authentication systems, or data stores directly. "
         "Findings reflect the agent's behaviour at the time of assessment; "
         "behaviour may vary across runs due to LLM non-determinism."),
    ]

    for title, body in methodology_paras:
        story.append(Paragraph(title, s["h3"]))
        story.append(Paragraph(body, s["body"]))
        story.append(Spacer(1, 3))

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_make_page_template, onLaterPages=_make_page_template)
    return output_path

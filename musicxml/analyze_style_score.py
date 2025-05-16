from music21 import converter
from common.style_advisor import style_advice
from common.part_utils import classify_parts
from common.html_report import render_html_report
import re
import os

def analyze_style(filepath):
    # 1) Parse score and run stylistic analysis
    score = converter.parse(filepath)
    raw_adv = style_advice(score)

    # 2) Classify parts
    vocal_parts, instr_parts = classify_parts(score)
    part_names = [p.partName for p in vocal_parts + instr_parts]

    # 3) Distribute advice into measure/part table
    issues_by_measure = {}

    for adv in raw_adv:
        # e.g. "Measure 3: dense texture"
        m = re.match(r"Measure (\d+): (.+)", adv)
        if m:
            measure = int(m.group(1))
            text = m.group(2)
            for part in part_names:
                issues_by_measure.setdefault(measure, {}).setdefault(part, []).append(text)
            continue

        # e.g. "Alto: consider syncopation in measure 4"
        p = re.match(r"(.+): .*measure (\d+)", adv)
        if p:
            part, meas = p.group(1), int(p.group(2))
            issues_by_measure.setdefault(meas, {}).setdefault(part, []).append("syncopation")
            continue

        # e.g. "At offset 0.0: try secondary dominant ^D"
        h = re.match(r"At offset ([0-9.]+): (.+)", adv)
        if h:
            offset = float(h.group(1))
            measure = int(offset) + 1
            text = h.group(2)
            for part in part_names:
                issues_by_measure.setdefault(measure, {}).setdefault(part, []).append(text)
            continue

        # fallback
        for part in part_names:
            issues_by_measure.setdefault(1, {}).setdefault(part, []).append(adv)

    # 4) Output HTML
    os.makedirs("report", exist_ok=True)
    report_path = "report/style_report.html"
    render_html_report(issues_by_measure, part_names, report_path)
    print(f"Style report written to {report_path}")

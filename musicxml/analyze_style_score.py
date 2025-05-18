# analyze_style_score.py
from music21 import converter
from common.style_advisor import style_advice
from common.part_utils import classify_parts
from common.html_report import render_html_report
from common.harmony_utils import detect_key, get_chords, extract_keys_by_measure, extract_meters_by_measure
import re
import os

def analyze_style(filepath):
    score = converter.parse(filepath)
    raw_adv = style_advice(score)
    vocal_parts, instr_parts = classify_parts(score)
    part_names = [p.partName for p in vocal_parts + instr_parts]
    issues_by_measure = {}

    for adv in raw_adv:
        m = re.match(r"Measure (\d+): (.+)", adv)
        if m:
            measure = int(m.group(1))
            text = m.group(2)
            for part in part_names:
                issues_by_measure.setdefault(measure, {}).setdefault(part, []).append(text)
            continue

        p = re.match(r"(.+): .*measure (\d+)", adv)
        if p:
            part, meas = p.group(1), int(p.group(2))
            issues_by_measure.setdefault(meas, {}).setdefault(part, []).append("syncopation")
            continue

        h = re.match(r"At offset ([0-9.]+): (.+)", adv)
        if h:
            offset = float(h.group(1))
            measure = int(offset) + 1
            text = h.group(2)
            for part in part_names:
                issues_by_measure.setdefault(measure, {}).setdefault(part, []).append(text)
            continue

        for part in part_names:
            issues_by_measure.setdefault(1, {}).setdefault(part, []).append(adv)

    os.makedirs("report", exist_ok=True)
    report_path = "report/style_report.html"
    key = detect_key(score)
    chords_by_measure = get_chords(score, use_full_score=True, merge_same_chords=True, key=key)
    keys_by_measure = extract_keys_by_measure(score)
    meters_by_measure = extract_meters_by_measure(score)

    render_html_report(
        issues_by_measure,
        part_names,
        report_path,
        chords_by_measure=chords_by_measure,
        abc_key=key,
        keys_by_measure=keys_by_measure,
        meters_by_measure=meters_by_measure
    )
    print(f"Style report written to {report_path}")

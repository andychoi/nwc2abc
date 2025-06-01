# analyze_combined_score.py
from music21 import converter, interval, note, roman
from common.part_utils import classify_parts
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression, extract_keys_by_measure, extract_meters_by_measure
from common.html_report import render_html_report

def analyze_combined(filepath, use_full_score_chords=False, report_path=None):
    score = converter.parse(filepath)
    vocal_parts, instrumental_parts = classify_parts(score)
    part_names = [p.partName for p in (vocal_parts + instrumental_parts)]
    issues_by_measure = {}

    for i in range(3):
        if i + 1 >= len(vocal_parts):
            break
        up_part = vocal_parts[i]
        lo_part = vocal_parts[i + 1]
        up_notes = [n for n in up_part.recurse().notes if isinstance(n, note.Note)]
        lo_notes = [n for n in lo_part.recurse().notes if isinstance(n, note.Note)]
        for un, ln in zip(up_notes, lo_notes):
            m = int(un.measureNumber)
            if ln.pitch > un.pitch:
                issues_by_measure.setdefault(m, {}).setdefault(lo_part.partName, []).append("crossing")
            iv = interval.Interval(ln, un)
            if iv.semitones > 12:
                issues_by_measure.setdefault(m, {}).setdefault(up_part.partName, []).append("spacing")

    for i in range(len(vocal_parts)):
        for j in range(i + 1, len(vocal_parts)):
            p1, p2 = vocal_parts[i], vocal_parts[j]
            n1s = [n for n in p1.recurse().notes if isinstance(n, note.Note)]
            n2s = [n for n in p2.recurse().notes if isinstance(n, note.Note)]
            for idx, (a, b) in enumerate(zip(n1s, n2s)):
                iv = interval.Interval(a, b)
                if iv.simpleName in ('P5', 'P8'):
                    if idx + 1 < len(n1s) and idx + 1 < len(n2s):
                        nxt = interval.Interval(n1s[idx + 1], n2s[idx + 1])
                        if nxt.simpleName == iv.simpleName:
                            m = int(a.measureNumber)
                            issues_by_measure.setdefault(m, {}).setdefault(p1.partName, []).append(f"parallel {iv.simpleName}")

    measures = score.parts[0].getElementsByClass('Measure')
    max_m = max(m.measureNumber for m in measures)
    for m in range(1, max_m + 1):
        vnotes, inotes = [], []
        for part in vocal_parts:
            m_el = part.measure(m)
            vnotes += [n for n in m_el.notes if isinstance(n, note.Note)]
        for part in instrumental_parts:
            m_el = part.measure(m)
            inotes += [n for n in m_el.notes if isinstance(n, note.Note)]
        for vn in vnotes:
            for ino in inotes:
                iv = interval.Interval(vn, ino)
                if not iv.isConsonant():
                    issues_by_measure.setdefault(m, {}).setdefault(vn.getContextByClass('Part').partName, []).append(f"dissonance with {ino.nameWithOctave}")
                if iv.semitones == 0:
                    issues_by_measure.setdefault(m, {}).setdefault(vn.getContextByClass('Part').partName, []).append(f"doubled in {ino.getContextByClass('Part').partName}")

    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    chords_by_measure = get_chords(score, use_full_score=True, merge_same_chords=True, key=key)
    prog_issues = analyze_chord_progression(chords, key)
    key_changes = extract_keys_by_measure(score)
    meter_changes = extract_meters_by_measure(score)

    for offset, issue in prog_issues:
        m = int(offset)
        if vocal_parts:
            issues_by_measure.setdefault(m, {}).setdefault(vocal_parts[0].partName, []).append("prog")

    render_html_report(
        issues_by_measure,
        part_names,
        report_path,
        chords_by_measure=chords_by_measure,
        abc_key=key,
        keys_by_measure=key_changes,
        meters_by_measure=meter_changes
    )
    print("Combined analysis complete → combined_report.html")
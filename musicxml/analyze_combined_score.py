# musicxml/analyze_combined_score.py

from music21 import converter, interval, note
from common.part_utils import classify_parts
from common.html_report import render_html_report

def analyze_combined(filepath):
    # 1) Parse
    score = converter.parse(filepath)

    # 2) Split parts
    vocal_parts, instrumental_parts = classify_parts(score)
    part_names = [p.partName for p in (vocal_parts + instrumental_parts)]
    issues_by_measure = {}

    # --- SATB crossing & spacing ---
    for i in range(3):
        if i+1 >= len(vocal_parts):
            break
        up_part = vocal_parts[i]
        lo_part = vocal_parts[i+1]
        up_notes = [n for n in up_part.recurse().notes if isinstance(n, note.Note)]
        lo_notes = [n for n in lo_part.recurse().notes if isinstance(n, note.Note)]
        for un, ln in zip(up_notes, lo_notes):
            m = int(un.measureNumber)
            # voice crossing
            if ln.pitch > un.pitch:
                issues_by_measure.setdefault(m, {}) \
                                 .setdefault(lo_part.partName, []) \
                                 .append("crossing")
            # spacing > octave
            iv = interval.Interval(ln, un)
            if iv.semitones > 12:
                issues_by_measure.setdefault(m, {}) \
                                 .setdefault(up_part.partName, []) \
                                 .append("spacing")

    # --- Parallel 5ths/8ves ---
    for i in range(len(vocal_parts)):
        for j in range(i+1, len(vocal_parts)):
            p1, p2 = vocal_parts[i], vocal_parts[j]
            n1s = [n for n in p1.recurse().notes if isinstance(n, note.Note)]
            n2s = [n for n in p2.recurse().notes if isinstance(n, note.Note)]
            for a, b in zip(n1s, n2s):
                iv = interval.Interval(a, b)
                if iv.simpleName in ('P5', 'P8'):
                    idx1, idx2 = n1s.index(a), n2s.index(b)
                    if idx1+1 < len(n1s) and idx2+1 < len(n2s):
                        nxt = interval.Interval(n1s[idx1+1], n2s[idx2+1])
                        if nxt.simpleName == iv.simpleName:
                            m = int(a.measureNumber)
                            issues_by_measure.setdefault(m, {}) \
                                             .setdefault(p1.partName, []) \
                                             .append(f"parallel {iv.simpleName}")

    # --- Vocal vs. Instrument dissonance & doubling ---
    measures = score.parts[0].getElementsByClass('Measure')
    max_m = max(m.measureNumber for m in measures)
    for m in range(1, max_m+1):
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
                # dissonance = not consonant
                if not iv.isConsonant():
                    issues_by_measure.setdefault(m, {}) \
                                     .setdefault(vn.getContextByClass('Part').partName, []) \
                                     .append(f"dissonance with {ino.nameWithOctave}")
                # exact doubling
                if iv.semitones == 0:
                    issues_by_measure.setdefault(m, {}) \
                                     .setdefault(vn.getContextByClass('Part').partName, []) \
                                     .append(f"doubled in {ino.getContextByClass('Part').partName}")

    # 4) Emit report
    render_html_report(issues_by_measure, part_names, "report/combined_report.html")
    print("Combined analysis complete → combined_report.html")

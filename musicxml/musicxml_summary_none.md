# Python Project Summary

## `analyze_combined_score.py`
```python
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
    render_html_report(issues_by_measure, part_names, "combined_report.html")
    print("Combined analysis complete → combined_report.html")

```

## `analyze_instrumental_score.py`
```python
from music21 import converter, note, pitch, interval
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression
from common.html_report import render_html_report

instrument_ranges = {
    'Flute': ('C4', 'C7'),
    'Violin': ('G3', 'E7'),
    'Cello': ('C2', 'G5'),
    'Piano': ('A0', 'C8'),
}

def get_instrument_name(part):
    instr = part.getInstrument(returnDefault=True)
    return instr.partName or instr.instrumentName or f"UnknownPart_{part.id}"

def is_note_out_of_range(n, instrument_name):
    if not isinstance(n, note.Note) or instrument_name not in instrument_ranges:
        return False
    low, high = pitch.Pitch(instrument_ranges[instrument_name][0]), pitch.Pitch(instrument_ranges[instrument_name][1])
    return n.pitch < low or n.pitch > high

def analyze_multistaff_harmony(part, instrument_name, issues_by_measure):
    staff_notes = {}
    for n in part.recurse().notes:
        staff_num = getattr(n, 'staff', 1)
        staff_notes.setdefault(staff_num, []).append(n)

    if len(staff_notes) >= 2:
        ids = sorted(staff_notes.keys())
        upper = staff_notes[ids[0]]
        lower = staff_notes[ids[1]]
        for n1, n2 in zip(upper, lower):
            try:
                iv = interval.Interval(n1, n2)
                m = int(n1.measureNumber)
                if iv.simpleName in ["A4", "d2", "m2"]:
                    issues_by_measure.setdefault(m, {}).setdefault(instrument_name, []).append("dissonance")
                elif iv.semitones > 24:
                    issues_by_measure.setdefault(m, {}).setdefault(instrument_name, []).append("spacing")
            except:
                continue

def analyze_instrumental(filepath):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score)
    progression_issues = analyze_chord_progression(chords, key)

    issues_by_measure = {}
    part_names = []

    for part in score.parts:
        name = get_instrument_name(part)
        part_names.append(name)
        for n in part.flat.notes:
            if is_note_out_of_range(n, name):
                m = int(n.measureNumber)
                issues_by_measure.setdefault(m, {}).setdefault(name, []).append("range")

        staff_count = set(getattr(n, 'staff', 1) for n in part.recurse().notes)
        if len(staff_count) >= 2:
            analyze_multistaff_harmony(part, name, issues_by_measure)

    for offset, issue in progression_issues:
        m = int(offset)
        if part_names:
            issues_by_measure.setdefault(m, {}).setdefault(part_names[0], []).append("prog")

    render_html_report(issues_by_measure, part_names, "instrumental_report.html")
    print("Instrumental harmony analysis complete. Output: instrumental_report.html")

```

## `analyze_vocal_score.py`
```python
from music21 import converter, interval, note, pitch
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression
from common.html_report import render_html_report
import os

vocal_ranges = {
    'Soprano': ('C4', 'G5'),
    'Alto': ('G3', 'D5'),
    'Tenor': ('C3', 'G4'),
    'Bass': ('E2', 'C4')
}

def is_note_out_of_range(n, part_name):
    if not isinstance(n, note.Note) or part_name not in vocal_ranges:
        return False
    low, high = pitch.Pitch(vocal_ranges[part_name][0]), pitch.Pitch(vocal_ranges[part_name][1])
    return n.pitch < low or n.pitch > high

def analyze_vocal(filepath):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score)
    progression_issues = analyze_chord_progression(chords, key)

    voice_labels = ['Soprano', 'Alto', 'Tenor', 'Bass']
    voices = [list(score.parts[i].recurse().notes) for i in range(min(4, len(score.parts)))]
    issues_by_measure = {}

    for i, notes in enumerate(voices):
        voice = voice_labels[i]
        for n in notes:
            if not isinstance(n, note.Note):
                continue
            m = int(n.measureNumber)
            issues_by_measure.setdefault(m, {}).setdefault(voice, [])
            if is_note_out_of_range(n, voice):
                issues_by_measure[m][voice].append("range")

    for i in range(3):
        upper_voice = voice_labels[i]
        lower_voice = voice_labels[i+1]
        upper = voices[i]
        lower = voices[i+1]
        for up_note, low_note in zip(upper, lower):
            if not isinstance(up_note, note.Note) or not isinstance(low_note, note.Note):
                continue
            m = int(up_note.measureNumber)
            if low_note.pitch > up_note.pitch:
                issues_by_measure.setdefault(m, {}).setdefault(lower_voice, []).append("crossing")

    for i in range(2):
        upper = voices[i]
        lower = voices[i+1]
        for up_note, low_note in zip(upper, lower):
            if not isinstance(up_note, note.Note) or not isinstance(low_note, note.Note):
                continue
            try:
                iv = interval.Interval(low_note, up_note)
                if iv.semitones > 12:
                    m = int(up_note.measureNumber)
                    issues_by_measure.setdefault(m, {}).setdefault(voice_labels[i], []).append("spacing")
            except:
                continue

    for i in range(3):
        for j in range(i+1, 4):
            v1 = voices[i]
            v2 = voices[j]
            for n1, n2 in zip(v1, v2):
                if not isinstance(n1, note.Note) or not isinstance(n2, note.Note):
                    continue
                try:
                    intvl = interval.Interval(n1, n2)
                    if intvl.simpleName in ['P5', 'P8']:
                        next1 = v1[v1.index(n1)+1] if v1.index(n1)+1 < len(v1) else None
                        next2 = v2[v2.index(n2)+1] if v2.index(n2)+1 < len(v2) else None
                        if isinstance(next1, note.Note) and isinstance(next2, note.Note):
                            next_intvl = interval.Interval(next1, next2)
                            if next_intvl.simpleName == intvl.simpleName:
                                m = int(n1.measureNumber)
                                issues_by_measure.setdefault(m, {}).setdefault(voice_labels[i], []).append(f"parallel {intvl.simpleName}")
                except:
                    continue

    for offset, issue in progression_issues:
        m = int(offset)
        issues_by_measure.setdefault(m, {}).setdefault("Soprano", []).append("prog")

    render_html_report(issues_by_measure, voice_labels, "vocal_report.html")
    print("Vocal harmony analysis complete. Output: vocal_report.html")

```

## `common/harmony_utils.py`
```python
from music21 import analysis, chord, key as key_module

def detect_key(score):
    return score.analyze('key')

def get_chords(score):
    return score.chordify().recurse().getElementsByClass('Chord')

def analyze_chord_progression(chords, key):
    prev_rn = None
    issues = []
    for c in chords:
        if not c.pitches:
            continue
        try:
            rn = analysis.roman.RomanNumeral(c, key)
        except:
            continue
        if prev_rn and prev_rn.figure.startswith("V") and not rn.figure.startswith("I"):
            issues.append((c.offset, f"V does not resolve to I (found {rn.figure})"))
        prev_rn = rn
    return issues

```

## `common/html_report.py`
```python
html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Harmony Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .ok {{ background-color: #e9fce9; }}
        .voicing {{ background-color: #fff3cd; color: #856404; }}
        .dissonance {{ background-color: #fdecea; color: #b94a48; }}
        .doubling {{ background-color: #d1ecf1; color: #0c5460; }}
        th, td {{ text-align: center; vertical-align: middle; }}
        .badge {{ font-size: 0.85em; }}
    </style>
</head>
<body class="container my-4">
    <h1 class="mb-4">Harmony Analysis Report</h1>
    <table class="table table-bordered table-sm">
        <thead class="table-light">
            <tr>
                <th>Measure</th>
                <th>Part</th>
                <th>Issue</th>
                <th>Type</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""

def classify_issue(issue_text):
    if "dissonance" in issue_text.lower():
        return "dissonance"
    elif "parallel" in issue_text or "crossing" in issue_text or "spacing" in issue_text:
        return "voicing"
    elif "doubled" in issue_text:
        return "doubling"
    return "voicing"

def render_html_report(issues_by_measure, part_names, output_path):
    rows_html = ''
    for m in sorted(issues_by_measure.keys()):
        for part in issues_by_measure[m]:
            for issue in issues_by_measure[m][part]:
                issue_type = classify_issue(issue)
                css_class = issue_type if issue_type in {"voicing", "dissonance", "doubling"} else "voicing"
                badge = f'<span class="badge text-bg-secondary">{issue_type}</span>'
                rows_html += f'<tr class="{css_class}"><td>{m}</td><td>{part}</td><td>{issue}</td><td>{badge}</td></tr>\n'

    html = html_template.format(rows=rows_html)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

```

## `common/part_utils.py`
```python
from music21 import instrument

def classify_parts(score):
    vocal_keywords = {"soprano", "alto", "tenor", "bass"}
    instrumental_parts = []
    vocal_parts = []

    for part in score.parts:
        name = part.partName.lower() if part.partName else ""
        inst = part.getInstrument(returnDefault=True)
        if any(voice in name for voice in vocal_keywords):
            vocal_parts.append(part)
        else:
            instrumental_parts.append(part)
    return vocal_parts, instrumental_parts

```

## `run_analysis.py`
```python
# musicxml/run_analysis.py
import sys
import os
import tempfile
import subprocess
from analyze_vocal_score import analyze_vocal
from analyze_instrumental_score import analyze_instrumental
from analyze_combined_score import analyze_combined

def nwc_to_musicxml_jar(nwc_filepath, script_path='./jar/nwc2xml.sh', output_musicxml_filepath=None):
    """Wrapper for local shell script conversion to MusicXML."""
    if output_musicxml_filepath is None:
        fd, output_musicxml_filepath = tempfile.mkstemp(suffix='.musicxml')
        os.close(fd)
    try:
        subprocess.run([script_path, nwc_filepath, output_musicxml_filepath], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conversion failed: {e}")
        sys.exit(1)        
    return output_musicxml_filepath

def convert_if_nwc(filepath):
    if filepath.endswith(".nwc") or filepath.endswith(".nwc.txt"):
        print("[INFO] Detected NWCtxt file. Converting to MusicXML...")
        filepath = nwc_to_musicxml_jar(filepath)
    return filepath

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_analysis.py [vocal|instrumental] path/to/score")
        sys.exit(1)

    mode, path = sys.argv[1], sys.argv[2]
    musicxml_path = convert_if_nwc(path)

    if mode == "vocal":
        analyze_vocal(musicxml_path)
    elif mode == "instrumental":
        analyze_instrumental(musicxml_path)
    elif mode == "combined":
        score = analyze_combined(musicxml_path)
    else:
        print("Unknown mode. Use 'vocal', 'instrumental', or 'combined'")
```


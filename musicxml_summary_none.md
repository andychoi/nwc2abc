# Python Project Summary

## `analyze_combined_score.py`
```python
from music21 import converter, interval, note, roman
from common.part_utils import classify_parts
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression
from common.html_report import render_html_report

def analyze_combined(filepath, use_full_score_chords=False):
    # 1) Parse
    score = converter.parse(filepath)

    # 2) Split parts
    vocal_parts, instrumental_parts = classify_parts(score)
    part_names = [p.partName for p in (vocal_parts + instrumental_parts)]
    issues_by_measure = {}

    # --- SATB crossing & spacing ---
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

    # --- Parallel 5ths/8ves ---
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

    # --- Vocal vs. Instrument dissonance & doubling ---
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

    # --- Chord analysis ---
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    chords_by_measure = {}
    prog_issues = analyze_chord_progression(chords, key)

    for c in chords:
        try:
            rn = roman.romanNumeralFromChord(c, key)
            m = int(c.measureNumber) if hasattr(c, 'measureNumber') else int(c.offset)
            chords_by_measure.setdefault(m, []).append(rn.figure)
        except:
            continue

    for offset, issue in prog_issues:
        m = int(offset)
        if vocal_parts:
            issues_by_measure.setdefault(m, {}).setdefault(vocal_parts[0].partName, []).append("prog")

    # 4) Emit report
    render_html_report(
        issues_by_measure,
        part_names,
        "report/combined_report.html",
        chords_by_measure=chords_by_measure,
        abc_key=key
    )
    print("Combined analysis complete → combined_report.html")

```

## `analyze_instrumental_score.py`
```python
from music21 import converter, note, pitch, interval, roman
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression
from common.html_report import render_html_report

instrument_ranges = {
    'Flute': ('C4', 'C7'),
    'Violin': ('G3', 'E7'),
    'Cello': ('C2', 'G5'),
    'Piano': ('A0', 'C8'),
}

VOCAL_PART_NAMES = {'Soprano', 'Alto', 'Tenor', 'Bass'}

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

def analyze_instrumental(filepath, use_full_score_chords=False):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    progression_issues = analyze_chord_progression(chords, key)

    issues_by_measure = {}
    chords_by_measure = {}
    part_names = []

    # Build chords by measure
    for c in chords:
        try:
            rn = roman.romanNumeralFromChord(c, key)
            m = int(c.measureNumber) if hasattr(c, 'measureNumber') else int(c.offset)
            chords_by_measure.setdefault(m, []).append(rn.figure)
        except:
            continue

    # Filter out vocal parts
    instrumental_parts = [p for p in score.parts if get_instrument_name(p) not in VOCAL_PART_NAMES]

    if not instrumental_parts:
        print("No instrumental parts found.")
        return

    # Identify topmost instrumental part as melody (like soprano)
    melody_part = instrumental_parts[0]

    # Collect part names and perform range & multistaff checks
    for part in instrumental_parts:
        name = get_instrument_name(part)
        part_names.append(name)

        for n in part.flatten().notes:
            if is_note_out_of_range(n, name):
                m = int(n.measureNumber)
                issues_by_measure.setdefault(m, {}).setdefault(name, []).append("range")

        staff_ids = set(getattr(n, 'staff', 1) for n in part.recurse().notes)
        if len(staff_ids) >= 2:
            analyze_multistaff_harmony(part, name, issues_by_measure)

    # Analyze dissonance and doubling vs. melody part
    measures = melody_part.getElementsByClass('Measure')
    max_m = max((m.measureNumber for m in measures), default=0)

    for part in instrumental_parts[1:]:  # skip melody itself
        name = get_instrument_name(part)
        for m in range(1, max_m + 1):
            melody_measure = melody_part.measure(m)
            inst_measure = part.measure(m)
            m_notes = [n for n in melody_measure.notes if isinstance(n, note.Note)] if melody_measure else []
            i_notes = [n for n in inst_measure.notes if isinstance(n, note.Note)] if inst_measure else []

            for mn in m_notes:
                for inote in i_notes:
                    iv = interval.Interval(mn, inote)
                    if not iv.isConsonant():
                        issues_by_measure.setdefault(m, {}).setdefault(name, []).append(f"dissonance with {mn.nameWithOctave}")
                    if iv.semitones == 0:
                        issues_by_measure.setdefault(m, {}).setdefault(name, []).append(f"doubled {mn.nameWithOctave}")

    # Mark chord progression issues under melody part
    for offset, issue in progression_issues:
        m = int(offset)
        name = get_instrument_name(melody_part)
        issues_by_measure.setdefault(m, {}).setdefault(name, []).append("prog")

    # Emit report
    render_html_report(
        issues_by_measure,
        part_names,
        "report/instrumental_report.html",
        chords_by_measure=chords_by_measure,
        abc_key=key
    )
    print("Instrumental harmony analysis complete. Output: instrumental_report.html")

```

## `analyze_style_score.py`
```python
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

```

## `analyze_vocal_score.py`
```python
from music21 import converter, interval, note, pitch, roman
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression
from common.html_report import render_html_report

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

def analyze_vocal(filepath, use_full_score_chords=True):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    # print(f"[DEBUG] Found {len(chords)} chords")
    progression_issues = analyze_chord_progression(chords, key)

    chords_by_measure = {}
    for c in chords:
        try:
            rn = roman.romanNumeralFromChord(c, key)
            m = int(c.measureNumber) if hasattr(c, 'measureNumber') else int(c.offset)
            chords_by_measure.setdefault(m, []).append(rn.figure)
        except Exception as e:
            print(f"[WARN] Failed to create RomanNumeral for chord {c}: {e}")

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
            for idx, (n1, n2) in enumerate(zip(v1, v2)):
                if not isinstance(n1, note.Note) or not isinstance(n2, note.Note):
                    continue
                try:
                    intvl = interval.Interval(n1, n2)
                    if intvl.simpleName in ['P5', 'P8']:
                        if idx + 1 < len(v1) and idx + 1 < len(v2):
                            next1, next2 = v1[idx+1], v2[idx+1]
                            if isinstance(next1, note.Note) and isinstance(next2, note.Note):
                                next_intvl = interval.Interval(next1, next2)
                                if next_intvl.simpleName == intvl.simpleName:
                                    m = int(n1.measureNumber)
                                    issues_by_measure.setdefault(m, {}).setdefault(voice_labels[i], []).append(f"parallel {intvl.simpleName}")
                except:
                    continue

    for offset, issue in progression_issues:
        m = int(offset)
        issues_by_measure.setdefault(m, {}).setdefault("Soprano", []).append(issue)

    render_html_report(
        issues_by_measure,
        voice_labels,
        "report/vocal_report.html",
        chords_by_measure=chords_by_measure,
        abc_key=key
    )
    print("Vocal harmony analysis complete. Output: vocal_report.html")

```

## `common/abc_utils.py`
```python
# common/abc_utils.py

from music21 import pitch, key

def pitch_to_abc(m21_pitch: pitch.Pitch, key_sig: key.Key = None) -> str:
    step = m21_pitch.step
    octave = m21_pitch.octave if m21_pitch.octave is not None else 4  # Default to octave 4 if missing
    acc_token = ""

    if m21_pitch.accidental:
        show_acc = True
        if key_sig:
            ks_acc = key_sig.accidentalByStep(step)
            if ks_acc and ks_acc.name == m21_pitch.accidental.name:
                show_acc = False
        if show_acc:
            acc_map = {"sharp": "^", "flat": "_", "natural": "="}
            acc_token = acc_map.get(m21_pitch.accidental.name, "")

    if octave >= 5:
        note = step.lower() + "'" * (octave - 5)
    else:
        note = step.upper() + "," * max(0, 4 - octave)

    return acc_token + note

```

## `common/harmony_utils.py`
```python
# common/harmony_utils.py
from music21 import analysis, chord, roman, key as key_module

def detect_key(score):
    return score.analyze('key')

def find_piano_part(score):
    for part in score.parts:
        name = (part.partName or '').lower()
        instr = part.getInstrument(returnDefault=True)
        if "piano" in name or "piano" in str(instr).lower():
            return part
    return None

def get_chords(score, use_full_score=False):
    """
    Return chords per measure based on:
    - Piano part if available (default)
    - Full score chordify if `use_full_score=True`
    """
    part = None
    if use_full_score:
        part = score.chordify()
    else:
        part = find_piano_part(score)
        if not part:
            print("[WARN] No piano part found. Falling back to full score chordify.")
            part = score.chordify()

    return part.recurse().getElementsByClass('Chord')

def analyze_chord_progression(chords, key):
    """
    Analyze chord progression and detect:
    - V not resolving to I
    - Modal mixture chords (borrowed from parallel key)
    - Deceptive cadences (V → vi)
    """
    issues = []
    prev_rn = None

    parallel_key = key.parallel
    for c in chords:
        if not c.pitches:
            continue
        try:
            rn = roman.romanNumeralFromChord(c, key)
            p_rn = roman.romanNumeralFromChord(c, parallel_key)
        except:
            continue

        m = int(c.measureNumber) if hasattr(c, 'measureNumber') else int(c.offset)

        # --- V should resolve to I ---
        if prev_rn and prev_rn.figure.startswith("V"):
            if rn.figure != "I":
                if rn.figure == "vi":
                    issues.append((m, f"deceptive cadence: V → vi"))
                else:
                    issues.append((m, f"V does not resolve to I (found {rn.figure})"))

        # --- Modal mixture (if chord fits better in parallel key) ---
        if p_rn.figure not in ('I', 'V') and p_rn.figure != rn.figure:
            # Avoid falsely triggering on same-name chords
            if p_rn.romanNumeral == rn.romanNumeral:
                continue
            if p_rn.root().name != rn.root().name:
                issues.append((m, f"modal mixture: {p_rn.figure} (parallel key)"))

        prev_rn = rn

    return issues


def suggest_reharmonizations(chords, key):
    suggestions = []
    for i, c in enumerate(chords):
        try:
            rn = roman.romanNumeralFromChord(c, key)
        except:
            continue

        subs = [rn.figure]

        if i + 1 < len(chords):
            try:
                next_rn = roman.romanNumeralFromChord(chords[i + 1], key)
                sec_dom = roman.romanNumeralFromChord(f'V/{next_rn.root().name}', key)
                subs.append(sec_dom.figure)
            except: pass

        try:
            mix_rn = roman.romanNumeralFromChord(c, key.parallelKey)
            if mix_rn.figure not in subs:
                subs.append(mix_rn.figure + "*")
        except: pass

        if rn.figure == 'V' and i + 1 < len(chords):
            try:
                actual = roman.romanNumeralFromChord(chords[i + 1], key)
                if actual.figure != 'I':
                    subs.append('vi')
            except: pass

        if rn.figure == 'I':
            subs.extend(['vi', 'iii'])

        abc_preview = ' '.join(f'"{fig}" {roman.romanNumeralFromChord(fig, key).root().name}' for fig in subs if '/' not in fig)

        suggestions.append({
            'measure': int(c.measureNumber) if hasattr(c, 'measureNumber') else i + 1,
            'original': rn.figure,
            'alternatives': list(dict.fromkeys(subs)),
            'abc': abc_preview
        })
    return suggestions
```

## `common/html_report.py`
```python
# common/html_report.py

from common.abc_utils import pitch_to_abc
from music21 import roman, pitch as m21pitch, key as m21key
import html

default_key = m21key.Key('C')

# HTML template with tooltip support for ABC hover
html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Harmony Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.0/dist/abcjs-basic-min.js"></script>
    <style>
        .ok {{ background-color: #e9fce9; }}
        .voicing {{ background-color: #fff3cd; color: #856404; }}
        .dissonance {{ background-color: #fdecea; color: #b94a48; }}
        .doubling {{ background-color: #d1ecf1; color: #0c5460; }}
        th, td {{ text-align: center; vertical-align: middle; }}
        .note-issue {{ margin-bottom: 0.5em; }}
        .note-issue b {{ display: block; }}
        .note-issue small {{ font-style: italic; }}
        .abc-hover {{ cursor: help; text-decoration: dotted underline; }}
        .abc-tooltip {{
            position: absolute;
            background: #fff;
            border: 1px solid #ccc;
            padding: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            z-index: 10000;
        }}
    </style>
</head>
<body class="container my-4">
    <h1 class="mb-4">Harmony Analysis Report</h1>
    <table class="table table-bordered table-sm">
        <thead class="table-light">
            <tr>
                <th>Measure</th><th>Chord</th>{columns}
            </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
    </table>

    <script>
        // Tooltip behavior for chord hover
        document.addEventListener('DOMContentLoaded', () => {{
            document.querySelectorAll('.abc-hover').forEach(el => {{
                el.addEventListener('mouseenter', () => {{
                    const abc = el.getAttribute('data-abc');
                    if (!abc) return;
                    const id = 'tt-' + Math.random().toString(36).substr(2,9);
                    const tt = document.createElement('div');
                    tt.id = id;
                    tt.className = 'abc-tooltip';
                    document.body.appendChild(tt);
                    const rect = el.getBoundingClientRect();
                    tt.style.left = `${{rect.left + window.scrollX}}px`;
                    tt.style.top = `${{rect.bottom + window.scrollY}}px`;
                    ABCJS.renderAbc(id, abc);
                    el._tooltipId = id;
                }});
                el.addEventListener('mouseleave', () => {{
                    if (el._tooltipId) {{
                        const tt = document.getElementById(el._tooltipId);
                        if (tt) tt.remove();
                        el._tooltipId = null;
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>"""


def classify_issue(issue_text):
    it = issue_text.lower()
    if "dissonance" in it:
        return "dissonance"
    if "doubled" in it:
        return "doubling"
    return "voicing"


def recommend_fix(issue_text):
    it = issue_text.lower()
    if "dissonance with " in it:
        target = issue_text.split("dissonance with ")[1]
        p = m21pitch.Pitch(target)
        abc_orig = pitch_to_abc(p)
        candidates = [default_key.pitchFromDegree(d) for d in range(1, 8)]
        for sp in candidates:
            sp.octave = p.octave
        candidates.sort(key=lambda sp: abs(sp.midi - p.midi))
        best = candidates[1] if candidates and candidates[0].midi == p.midi else candidates[0]
        abc_res = pitch_to_abc(best)
        mode = '' if default_key.mode == 'major' else 'm'
        return f"Resolve dissonance: change {abc_orig} → {abc_res} ({default_key.tonic.name}{mode} scale)"
    if "spacing" in it:
        return "Reduce to within an octave: e.g., lower upper voice by 8ve (ABC: A→a)."
    if "crossing" in it:
        return "Maintain voice order: transpose the lower voice down (e.g., ABC: A,)."
    if "parallel p5" in it:
        return "Break parallel 5th: insert passing tone in one voice (e.g., z in ABC)."
    if "parallel p8" in it:
        return "Avoid parallel 8ve: use contrary motion (e.g., step in ABC: c d c)."
    if "doubled in " in it:
        part = issue_text.split("doubled in ")[1]
        return f"Drop duplication in {part} or vary voicing—e.g., use ABC chord [CE] instead of [CC]."
    if "prog" in it:
        tonic = pitch_to_abc(default_key.tonic)
        dominant = pitch_to_abc(default_key.pitchFromDegree(5))
        return f"V→I in ABC: {dominant} {dominant} {dominant}| {tonic} {tonic} {tonic}"
    return "Review harmonic context."


def render_html_report(issues_by_measure, part_names, output_path, chords_by_measure=None, abc_key=None):
    # Build columns and rows
    columns_html = ''.join(f'<th>{part}</th>' for part in part_names)
    rows_html = []
    all_abc = []

    for m in sorted(issues_by_measure.keys()):
        chords = chords_by_measure.get(m, []) if chords_by_measure else []
        if isinstance(chords, str): chords = [chords]
        labels = ", ".join(chords)
        data_abc = ""
        if abc_key and chords:
            parts = []
            for rn in chords:
                try:
                    obj = roman.RomanNumeral(rn, abc_key)
                    pitches = [pitch_to_abc(p, abc_key) for p in obj.pitches]
                    parts.append(f'"{rn}" [{" ".join(pitches)}]')
                except:
                    parts.append(f'"{rn}" z')
            data_abc = " ".join(parts)
            all_abc.append(data_abc)
        hover_div = f'<div class="abc-hover" data-abc="{html.escape(data_abc)}">{labels}</div>' if labels else ''

        cells = [f'<td>{m}</td>', f'<td>{hover_div}</td>']
        for part in part_names:
            issues = issues_by_measure.get(m, {}).get(part, [])
            if not issues:
                cells.append('<td class="ok">✓</td>')
            else:
                inner = ''.join(f'<div class="note-issue {classify_issue(i)}"><b>{i}</b><small>{recommend_fix(i)}</small></div>' for i in issues)
                cells.append(f'<td>{inner}</td>')
        rows_html.append('<tr>' + ''.join(cells) + '</tr>')

    full_html = html_template.format(
        columns=columns_html,
        rows="\n".join(rows_html),
        abc_preview="\n".join(all_abc)
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)


def render_style_report(issues_by_measure, part_names, output_path, chords_by_measure=None, abc_key=None):
    render_html_report(issues_by_measure, part_names, output_path, chords_by_measure, abc_key)

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

## `common/style_advisor.py`
```python
from music21 import interval, note, chord, key as m21key
from common.abc_utils import pitch_to_abc
from common.part_utils import classify_parts
from common.harmony_utils import get_chords, detect_key

def voice_motion_metrics(vocal_parts):
    """
    Compute relative percentages of motion types between adjacent vocal voices.
    """
    counts = {'parallel': 0, 'contrary': 0, 'oblique': 0, 'similar': 0}
    total = 0
    for vp1, vp2 in zip(vocal_parts[:-1], vocal_parts[1:]):
        n1s = [n for n in vp1.recurse().notes if isinstance(n, note.Note)]
        n2s = [n for n in vp2.recurse().notes if isinstance(n, note.Note)]
        for a, b in zip(n1s, n2s):
            # find next notes
            idx1, idx2 = n1s.index(a), n2s.index(b)
            if idx1+1 < len(n1s) and idx2+1 < len(n2s):
                na, nb = n1s[idx1+1], n2s[idx2+1]
                iv1 = interval.Interval(a, b).semitones
                iv2 = interval.Interval(na, nb).semitones
                # parallel perfect 5th or octave
                if iv1 == iv2 and abs(iv1) in (7, 12):
                    counts['parallel'] += 1
                # contrary if voices move in opposite directions
                elif (na.pitch.midi - a.pitch.midi)*(nb.pitch.midi - b.pitch.midi) < 0:
                    counts['contrary'] += 1
                # oblique if one voice stationary
                elif a.pitch.midi == na.pitch.midi or b.pitch.midi == nb.pitch.midi:
                    counts['oblique'] += 1
                else:
                    counts['similar'] += 1
                total += 1
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k]/total for k in counts}

def density_advice(score, threshold=4.0):
    """
    Flag measures where average notes per voice exceeds threshold.
    """
    advice = []
    parts = score.parts
    max_m = max(m.measureNumber for m in parts[0].getElementsByClass('Measure'))
    for m in range(1, max_m+1):
        count = 0
        for p in parts:
            meas = p.measure(m)
            count += len([n for n in meas.notes if isinstance(n, note.Note)])
        avg = count/len(parts)
        if avg > threshold:
            advice.append(f"Measure {m}: dense texture ({avg:.1f} notes/voice)")
    return advice

def syncopation_advice(score):
    """
    Suggest adding syncopation when all voices align on downbeats.
    """
    advice = []
    for p in score.parts:
        for m in p.getElementsByClass('Measure'):
            notes = [n for n in m.notes if isinstance(n, note.Note)]
            if notes and all(n.offset % 1 == 0 for n in notes):
                advice.append(f"{p.partName}: consider syncopation in measure {m.measureNumber}")
    return advice

def reharmonization_advice(score, use_full_score_chords=False):
    """
    Propose secondary dominants for each diatonic chord.
    """
    advice = []
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    for c in chords:
        try:
            rn = m21key.roman.romanNumeralFromChord(c, key)
            if rn.degree not in (5,):
                sec = key.pitchFromDegree(5).transpose((rn.degree-1)*7)  # V of that degree
                advice.append(
                    f"At offset {c.offset:.1f}: try secondary dominant {pitch_to_abc(sec)} for {rn.figure}"
                )
        except:
            continue
    return advice

def style_advice(score):
    """
    Collate all style recommendations into a list of strings.
    """
    advice = []
    # 1) Voice-leading balance
    vocals, _ = classify_parts(score)
    vm = voice_motion_metrics(vocals)
    if vm['contrary'] < 0.3:
        advice.append("Low contrary motion (<30%)—consider more independent lines.")
    # 2) Texture density
    advice += density_advice(score)
    # 3) Rhythm variety
    advice += syncopation_advice(score)
    # 4) Reharmonization
    advice += reharmonization_advice(score)
    return advice

```

## `run_analysis.py`
```python
import sys
import os
import tempfile
import subprocess
from music21 import converter
from analyze_vocal_score import analyze_vocal
from analyze_instrumental_score import analyze_instrumental
from analyze_combined_score import analyze_combined
from analyze_style_score import analyze_style

def nwc_to_musicxml_jar(nwc_filepath, script_path='./jar/nwc2xml.sh', output_musicxml_filepath=None):
    """
    Wrapper for local shell script that converts NWCtxt file to MusicXML.
    """
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

def main():
    if len(sys.argv) < 3:
        print("Usage: run_analysis.py [vocal|instrumental|combined|style] path/to/score [--full-score-chords]")
        sys.exit(1)

    mode = sys.argv[1]
    path = sys.argv[2]
    use_full_score_chords = '--full-score-chords' in sys.argv

    musicxml_path = convert_if_nwc(path)

    # Pass a flag into each analyzer if needed — adjust analyzers to accept this if required
    if mode == "vocal":
        analyze_vocal(musicxml_path, use_full_score_chords=use_full_score_chords)
    elif mode == "instrumental":
        analyze_instrumental(musicxml_path, use_full_score_chords=use_full_score_chords)
    elif mode == "combined":
        analyze_combined(musicxml_path, use_full_score_chords=use_full_score_chords)
    elif mode == "style":
        analyze_style(musicxml_path)
    else:
        print("Unknown mode. Use 'vocal', 'instrumental', 'combined', or 'style'")
        sys.exit(1)

if __name__ == "__main__":
    main()

```


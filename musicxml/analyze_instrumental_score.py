from music21 import converter, note, pitch, interval, roman
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

def analyze_instrumental(filepath, use_full_score_chords=False):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    progression_issues = analyze_chord_progression(chords, key)

    issues_by_measure = {}
    chords_by_measure = {}
    part_names = []

    # Collect chords by measure for ABC rendering
    for c in chords:
        try:
            rn = roman.romanNumeralFromChord(c, key)
            m = int(c.measureNumber) if hasattr(c, 'measureNumber') else int(c.offset)
            chords_by_measure.setdefault(m, []).append(rn.figure)
        except:
            continue

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

    render_html_report(
        issues_by_measure,
        part_names,
        "report/instrumental_report.html",
        chords_by_measure=chords_by_measure,
        abc_key=key
    )
    print("Instrumental harmony analysis complete. Output: instrumental_report.html")
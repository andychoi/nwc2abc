# analyze_instrumental_score.py
from music21 import converter, note, pitch, interval, roman
from common.harmony_utils import detect_key, get_chords, analyze_chord_progression, extract_keys_by_measure, extract_meters_by_measure
from common.html_report import render_html_report
from common.part_utils import classify_parts

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

def analyze_instrumental(filepath, use_full_score_chords=False, exclude_piano=True):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    progression_issues = analyze_chord_progression(chords, key)

    issues_by_measure = {}
    chords_by_measure = get_chords(score, use_full_score=True, merge_same_chords=True, key=key)
    part_names = []
    vocal_parts, instrumental_parts = classify_parts(score, exclude_piano)

    if not instrumental_parts:
        print("No instrumental parts found.")
        return

    melody_part = instrumental_parts[0]

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

    measures = melody_part.getElementsByClass('Measure')
    max_m = max((m.measureNumber for m in measures), default=0)

    for part in instrumental_parts[1:]:
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

    for offset, issue in progression_issues:
        m = int(offset)
        name = get_instrument_name(melody_part)
        issues_by_measure.setdefault(m, {}).setdefault(name, []).append("prog")

    keys_by_measure = extract_keys_by_measure(score)
    meters_by_measure = extract_meters_by_measure(score)

    render_html_report(
        issues_by_measure,
        part_names,
        "report/instrumental_report.html",
        chords_by_measure=chords_by_measure,
        abc_key=key,
        keys_by_measure=keys_by_measure,
        meters_by_measure=meters_by_measure
    )
    print("Instrumental harmony analysis complete. Output: instrumental_report.html")
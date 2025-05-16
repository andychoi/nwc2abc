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

def analyze_vocal(filepath, use_full_score_chords=False):
    score = converter.parse(filepath)
    key = detect_key(score)
    chords = get_chords(score, use_full_score=use_full_score_chords)
    progression_issues = analyze_chord_progression(chords, key)

    chords_by_measure = {}
    for c in chords:
        try:
            rn = roman.RomanNumeral(c, key)
            m = int(c.measureNumber) if hasattr(c, 'measureNumber') else int(c.offset)
            chords_by_measure.setdefault(m, []).append(rn.figure)
        except:
            continue

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

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

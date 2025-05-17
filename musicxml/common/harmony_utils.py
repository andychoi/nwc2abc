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
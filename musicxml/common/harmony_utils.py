# common/harmony_utils.py
from music21 import analysis, chord, roman, key as key_module, meter

def detect_key(score):
    return score.analyze('key')

def find_piano_part(score):
    for part in score.parts:
        name = (part.partName or '').lower()
        instr = part.getInstrument(returnDefault=True)
        if "piano" in name or "piano" in str(instr).lower():
            return part
    return None

def get_chords(score, use_full_score=False, merge_same_chords=True, key=None):
    """
    Return chords_by_measure: Dict[int, List[Tuple[Chord|str, float]]]
    Ensures every measure is included and padded to match its time signature.

    If merge_same_chords is True, same Roman figures are merged and summed by duration.
    """
    from music21 import meter, chord

    if use_full_score:
        part = score.chordify()
    else:
        part = find_piano_part(score) or score.chordify()

    ts = part.recurse().getElementsByClass(meter.TimeSignature).first()
    measure_length = ts.barDuration.quarterLength if ts else 4.0

    chords_raw = list(part.recurse().getElementsByClass('Chord'))
    chords_by_measure = {}

    for ch in chords_raw:
        m = ch.measureNumber
        dur = float(ch.quarterLength)
        chords_by_measure.setdefault(m, []).append((ch, dur))

    # Normalize and optionally merge by Roman figure
    max_measure = max(chords_by_measure.keys(), default=0)
    normalized = {}

    for m in range(1, max_measure + 1):
        items = chords_by_measure.get(m, [])
        total = sum(d for _, d in items)
        normalized[m] = list(items)

        if total < measure_length:
            filler = chord.Chord()
            filler.offset = (m - 1) * measure_length + total
            normalized[m].append((filler, measure_length - total))
        elif total > measure_length:
            overflow = total - measure_length
            if normalized[m]:
                ch_last, d_last = normalized[m][-1]
                if d_last > overflow:
                    normalized[m][-1] = (ch_last, d_last - overflow)

    # Merge if requested
    if merge_same_chords and key:
        merged = {}
        for m, chords in normalized.items():
            out = {}
            for c, dur in chords:
                try:
                    rn = roman.romanNumeralFromChord(c, key)
                    fig = rn.figure
                    out[fig] = out.get(fig, 0) + dur
                except:
                    continue
            merged[m] = [(fig, dur) for fig, dur in out.items()]
        return merged

    return normalized


def analyze_chord_progression(chords_by_measure, key):
    """
    Analyze chord progression and detect:
    - V not resolving to I
    - Modal mixture chords (borrowed from parallel key)
    - Deceptive cadences (V → vi)
    
    Expects: chords_by_measure: Dict[int, List[Tuple[Chord, float]]]
    Returns: List[Tuple[int, str]] (measure number, issue description)
    """
    issues = []
    prev_rn = None
    parallel_key = key.parallel

    for m in sorted(chords_by_measure.keys()):
        for c, _ in chords_by_measure[m]:
            if not c.pitches:
                continue

            try:
                rn = roman.romanNumeralFromChord(c, key)
                p_rn = roman.romanNumeralFromChord(c, parallel_key)
            except:
                continue

            # --- V should resolve to I ---
            if prev_rn and prev_rn.figure.startswith("V"):
                if rn.figure != "I":
                    if rn.figure == "vi":
                        issues.append((m, f"deceptive cadence: V → vi"))
                    else:
                        issues.append((m, f"V does not resolve to I (found {rn.figure})"))

            # --- Modal mixture (if fits better in parallel key) ---
            if p_rn.figure not in ('I', 'V') and p_rn.figure != rn.figure:
                if p_rn.romanNumeral != rn.romanNumeral and p_rn.root().name != rn.root().name:
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
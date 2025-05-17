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

# common/abc_utils.py

from music21 import pitch, key

def pitch_to_abc(m21_pitch: pitch.Pitch, key_sig: key.Key = None) -> str:
    step = m21_pitch.step
    octave = m21_pitch.octave
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

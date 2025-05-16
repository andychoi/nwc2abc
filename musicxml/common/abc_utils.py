# common/abc_utils.py

from music21 import pitch

def pitch_to_abc(m21_pitch):
    """
    Convert a music21.pitch.Pitch into ABC pitch notation (with octave marks).
    """
    step = m21_pitch.step
    octave = m21_pitch.octave
    acc_token = ""
    if m21_pitch.accidental:
        acc_map = {"sharp": "^", "flat": "_", "natural": "="}
        acc_token = acc_map.get(m21_pitch.accidental.name, "")
    # ABC uses lowercase for octave ≥5
    if octave >= 5:
        note = step.lower() + "'" * (octave - 5)
    else:
        note = step.upper() + "," * max(0, 4 - octave)
    return acc_token + note

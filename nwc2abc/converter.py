
# Full v3 converter code goes here
import subprocess
import tempfile
import os
import requests
import music21
from fractions import Fraction

def pitch_to_abc(m21_pitch):
    name = m21_pitch.step
    octave = m21_pitch.octave
    accidental_char = ""
    if m21_pitch.accidental:
        if m21_pitch.accidental.name == 'sharp':
            accidental_char = "^"
        elif m21_pitch.accidental.name == 'flat':
            accidental_char = "_"
        elif m21_pitch.accidental.name == 'natural':
            accidental_char = "="
    if octave >= 5:
        abc_name = name.lower() + "'" * (octave - 5)
    else:
        abc_name = name.upper() + "," * (4 - octave if octave < 4 else 0)
    return accidental_char + abc_name

def duration_to_abc(m21_duration, L_unit_quarter_length):
    val = Fraction(m21_duration.quarterLength).limit_denominator(32) / Fraction(L_unit_quarter_length)
    if val == 1: return ""
    if val < 1: return f"/{int(1/val)}"
    if val.denominator == 1: return str(val.numerator)
    return f"{val.numerator}/{val.denominator}"

def get_abc_barline(m21_barline):
    if m21_barline:
        return {'regular': "|", 'double': "||", 'final': "|]", 'heavy-light': "|:", 'light-heavy': ":|"}.get(m21_barline.style, "|")
    return "|"

def musicxml_to_simplified_abc(musicxml_filepath, default_L_denom=8, simplicity_level='raw'):
    score = music21.converter.parse(musicxml_filepath)
    abc_output = [f"X: {score.metadata.title or 'Untitled'}"]
    if score.metadata and score.metadata.composer:
        abc_output.append(f"C: {score.metadata.composer}")
    ts = score.flat.getElementsByClass('TimeSignature')
    abc_output.append(f"M: {ts[0].ratioString}" if ts else "M: 4/4")
    abc_output.append(f"L: 1/{default_L_denom}")
    L_unit_ql = 4.0 / default_L_denom
    ks = score.flat.getElementsByClass('KeySignature')
    key_name = "C"
    if ks:
        key_obj = ks[0]
        scale = key_obj.getScale('minor') if key_obj.mode == 'minor' else key_obj.getScale('major')
        key_name = scale.tonic.name.replace('-', 'b')
        if key_obj.mode == 'minor': key_name += "m"
    abc_output.append(f"K: {key_name}")

    for part in score.parts:
        part_name = part.partName or f"Part"
        if simplicity_level == 'raw':
            clefs = part.getElementsByClass('Clef')
            clef_name = ""
            if clefs:
                c = clefs[0]
                if c.sign == 'G': clef_name = "treble"
                elif c.sign == 'F': clef_name = "bass"
                elif c.sign == 'C': clef_name = "alto"
                if c.octaveChange == -1: clef_name += "8"
            abc_output.append(f"V:{part_name} clef={clef_name or 'auto'}")

        part_str = ""
        for measure in part.getElementsByClass(music21.stream.Measure):
            elems = []
            if simplicity_level == 'raw':
                elems.append(f"[m{measure.measureNumber}]")
            for el in measure.notesAndRests:
                dur = duration_to_abc(el.duration, L_unit_ql) if simplicity_level != 'simple' else ""
                if isinstance(el, music21.note.Note):
                    token = pitch_to_abc(el.pitch) + dur
                    if el.tie and el.tie.type == 'start' and simplicity_level == 'raw':
                        token += "-"
                elif isinstance(el, music21.chord.Chord):
                    notes = [pitch_to_abc(p) for p in el.pitches]
                    token = max(notes) if simplicity_level == 'simple' else "[" + " ".join(notes) + "]" + dur
                    if not simplicity_level == 'simple' and el.tie and el.tie.type == 'start' and simplicity_level == 'raw':
                        token += "-"
                else:
                    token = "z" + dur
                if el.duration.tuplets and simplicity_level == 'raw':
                    tup = el.duration.tuplets[0]
                    if tup.type == 'start':
                        elems.append(f"({tup.tupletActual[0]}")
                elems.append(token)
            elems.append(get_abc_barline(measure.rightBarline))
            part_str += " ".join(elems) + " "
            if measure.measureNumber % 4 == 0:
                part_str += "\n"
        abc_output.append(part_str.strip())
    return "\n".join(abc_output)

def nwc_to_musicxml_jar(nwc_filepath, jar_path, output_musicxml_filepath=None):
    if output_musicxml_filepath is None:
        fd, output_musicxml_filepath = tempfile.mkstemp(suffix='.musicxml')
        os.close(fd)
    subprocess.run(['java', '-jar', jar_path, nwc_filepath], check=True, stdout=open(output_musicxml_filepath, 'wb'), stderr=subprocess.PIPE)
    return output_musicxml_filepath

def nwc_to_musicxml_service(nwc_filepath, service_url='https://nwc2musicxml.appspot.com/'):
    with open(nwc_filepath, 'rb') as f:
        resp = requests.post(service_url, files={'file': f})
    resp.raise_for_status()
    fd, tmp = tempfile.mkstemp(suffix='.musicxml')
    os.close(fd)
    with open(tmp, 'wb') as o:
        o.write(resp.content)
    return tmp

def nwc_to_simplified_abc(nwc_filepath, method='jar', jar_path='./nwc2musicxml.jar', service_url='https://nwc2musicxml.appspot.com/', default_L_denom=8, simplicity_level='raw'):
    if method == 'jar':
        xml_path = nwc_to_musicxml_jar(nwc_filepath, jar_path)
    elif method == 'service':
        xml_path = nwc_to_musicxml_service(nwc_filepath, service_url)
    else:
        raise ValueError("method must be 'jar' or 'service'")
    try:
        return musicxml_to_simplified_abc(xml_path, default_L_denom, simplicity_level)
    finally:
        if method == 'service' and os.path.exists(xml_path):
            os.remove(xml_path)

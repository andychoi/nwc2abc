# converter.py
import subprocess
import tempfile
import os
import sys
import requests
import music21
from fractions import Fraction
from pathlib import Path

def log(msg):
    print(msg, file=sys.stdout, flush=True)

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
    title = score.metadata.title.strip() if score.metadata and score.metadata.title else "Untitled"
    abc_output = [f"X: {title}"]
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

def nwc_to_musicxml_jar(nwc_filepath, script_path='./jar/nwc2xml.sh', output_musicxml_filepath=None):
    """
    Wrapper for shell script (nwc2xml.sh).
    """
    if output_musicxml_filepath is None:
        fd, output_musicxml_filepath = tempfile.mkstemp(suffix='.musicxml')
        os.close(fd)

    log(f"[INFO] Running script: {script_path} {nwc_filepath} {output_musicxml_filepath}")
    try:
        subprocess.run([script_path, nwc_filepath, output_musicxml_filepath], check=True, text=True)
        log(f"[INFO] Script completed, MusicXML file: {output_musicxml_filepath}")
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Script failed with return code {e.returncode}")
        raise

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

def nwc_to_simplified_abc(
        nwc_filepath,
        method='jar',
        script_path='./jar/nwc2xml.sh',   # <-- FIX
        service_url='https://nwc2musicxml.appspot.com/',
        default_L_denom=8,
        simplicity_level='raw'):
        
    log(f"[INFO] Starting NWC to ABC conversion: file={nwc_filepath}, method={method}, level={simplicity_level}")
    
    if method == 'jar':
        log(f"[INFO] Converting NWC → MusicXML using script: {script_path}")
        xml_path = nwc_to_musicxml_jar(nwc_filepath, script_path)
    elif method == 'service':
        log(f"[INFO] Converting NWC → MusicXML using online service: {service_url}")
        xml_path = nwc_to_musicxml_service(nwc_filepath, service_url)
    else:
        raise ValueError("method must be 'jar' or 'service'")

    log(f"[INFO] MusicXML file created at: {xml_path}")
    
    try:
        abc = musicxml_to_simplified_abc(xml_path, default_L_denom, simplicity_level)
        log(f"[INFO] ABC conversion completed successfully.")
        return abc
    except Exception as e:
        log(f"[ERROR] ABC conversion failed: {e}")
        raise
    finally:
        if method == 'service' and os.path.exists(xml_path):
            os.remove(xml_path)
            log(f"[INFO] Temporary MusicXML file removed: {xml_path}")

def simplified_abc_to_musicxml(abc_string, output_filepath=None):
    """
    Convert simplified ABC string to MusicXML using music21.
    """
    log(f"[INFO] Starting simplified ABC to MusicXML conversion")
    # Ensure minimal ABC header (music21 requires at least X, M, K)
    if not abc_string.startswith("X:"):
        abc_string = f"X:1\nM:4/4\nK:C\n{abc_string}"

    # Parse ABC to music21 stream
    score = music21.converter.parse(abc_string, format='abc')

    if output_filepath:
        score.write('musicxml', fp=output_filepath)
        return output_filepath
    else:
        return score.write('musicxml')


def abc_to_nwc(abc_string: str, output_filepath: str = None) -> str:
    """
    Convert multi-voice ABC notation into a valid NoteWorthy Composer
    ASCII (.nwctxt) file, with separate Staff blocks, voice names,
    clefs, key & meter.  Ready for import into NWC 2.75+.
    """
    log("[INFO] Starting ABC → NWCtxt conversion")

    # 1) Ensure minimal header:
    if not abc_string.startswith("X:"):
        abc_string = "X:1\nM:4/4\nK:C\n" + abc_string

    # 2) Parse into a Score with parts (voices):
    score = music21.converter.parse(abc_string, format='abc')

    # 3) Grab global metadata:
    title    = (score.metadata.title    or "Untitled").replace(":", "") 
    composer = (score.metadata.composer or "").replace(":", "")
    ts_elems = score.flat.getElementsByClass(music21.meter.TimeSignature)
    ks_elems = score.flat.getElementsByClass(music21.key.KeySignature)
    ts_str   = ts_elems[0].ratioString if ts_elems else "4/4"
    key_obj  = ks_elems[0] if ks_elems else None
    key_str  = key_obj.sharps if key_obj else 0

    # 4) Build the .nwctxt lines
    lines = []
    lines.append("!NoteWorthyComposer(2.75)")
    lines.append("!Info")
    lines.append(f"!SongInfo Title={title}")
    if composer:
        lines.append(f"!SongInfo Composer={composer}")
    lines.append(f"!Meter {ts_str}")
    lines.append(f"!KeySharps {key_str}")
    lines.append("")          # blank line before the Staff blocks

    # 5) One [Staff] block per part:
    for idx, part in enumerate(score.parts, start=1):
        part_name = part.partName or f"Voice{idx}"

        # Clef
        clefs = part.getElementsByClass(music21.clef.Clef)
        clef_name = (clefs[0].sign + str(clefs[0].octaveChange or "")).lower() if clefs else "g2"

        lines.append(f"[Staff {idx}]")
        lines.append(f"!VoiceName {part_name}")
        lines.append(f"!Clef {clef_name}")
        lines.append("")   # blank before the notes

        # Now emit each measure’s notes inline, using NWC barlines '|'
        for measure in part.getElementsByClass(music21.stream.Measure):
            tokens = []
            for el in measure.notesAndRests:
                dur = el.duration.quarterLength
                if isinstance(el, music21.note.Note):
                    tokens.append(f"{el.pitch.nameWithOctave}/{dur}")
                elif isinstance(el, music21.chord.Chord):
                    pcs = ".".join(p.nameWithOctave for p in el.pitches)
                    tokens.append(f"[{pcs}]/{dur}")
                else:  # Rest
                    tokens.append(f"R/{dur}")
            # join tokens, prepend barline
            lines.append("| " + " ".join(tokens))

        lines.append("[EndStaff]")
        lines.append("")   # blank line between staves

    nwc_text = "\n".join(lines)

    # 6) Write or return
    if output_filepath:
        Path(output_filepath).write_text(nwc_text, encoding="utf-8")
        log(f"[INFO] NWCtxt written to {output_filepath}")
        return output_filepath
    else:
        return nwc_text
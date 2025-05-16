# nwc2abc/converter.py

import subprocess
import tempfile
import os
import sys
import requests
import music21
from fractions import Fraction
from pathlib import Path

def log(msg: str):
    print(msg, file=sys.stdout, flush=True)

def pitch_to_abc(m21_pitch: music21.pitch.Pitch) -> str:
    """Convert a music21 Pitch into ABC pitch notation (with octave marks)."""
    step = m21_pitch.step
    octave = m21_pitch.octave
    acc = ""
    if m21_pitch.accidental:
        name = m21_pitch.accidental.name
        acc = {"sharp":"^", "flat":"_", "natural":"="}.get(name, "")
    if octave >= 5:
        note = step.lower() + "'" * (octave - 5)
    else:
        note = step.upper() + "," * max(0, 4 - octave)
    return acc + note

def duration_to_abc(m21_duration: music21.duration.Duration, L_unit_quarter_length: float) -> str:
    """Convert a music21 Duration to an ABC duration token."""
    frac = Fraction(m21_duration.quarterLength).limit_denominator(32)
    val = frac / Fraction(L_unit_quarter_length)
    if val == 1:
        return ""
    if val < 1:
        return f"/{int(1/val)}"
    if val.denominator == 1:
        return str(val.numerator)
    return f"{val.numerator}/{val.denominator}"

def musicxml_to_simplified_abc(
    musicxml_filepath: str,
    default_L_denom: int = 8,
    simplicity_level: str = "raw"
) -> str:
    """
    Convert a multi-part MusicXML file into fully-featured ABC:
     - Global X, T, C, M, L, K headers
     - One V: definition per voice (with name & clef)
     - Then each voice’s measures under its V: block
     - Groups 4 bars per line for readability
    """
    score = music21.converter.parse(musicxml_filepath)
    meta = score.metadata or music21.metadata.Metadata()
    title    = meta.title.strip()    if meta.title else "Untitled"
    composer = meta.composer.strip() if meta.composer else ""

    # Time signature & Key signature
    ts = score.flat.getElementsByClass(music21.meter.TimeSignature)
    ks = score.flat.getElementsByClass(music21.key.KeySignature)
    meter = ts[0].ratioString if ts else "4/4"
    if ks:
        keyobj = ks[0].asKey()
        kname = keyobj.tonic.name.replace('-', 'b')
        if keyobj.mode == 'minor':
            kname += "m"
    else:
        kname = "C"

    # Prepare L unit
    L_unit_ql = 4.0 / default_L_denom

    # Header lines
    abc_lines = [
        "X: 1",
        f"T: {title}",
        f"C: {composer}" if composer else None,
        f"M: {meter}",
        f"L: 1/{default_L_denom}",
        f"K: {kname}"
    ]
    abc_lines = [line for line in abc_lines if line]

    # Collect voices: (voice_id, partName, clef, part_stream)
    voices = []
    for part in score.parts:
        pname = part.partName or part.id or f"Voice{len(voices)+1}"
        vid = str(pname).strip()[0].upper()
        if vid not in ("S","A","T","B"):
            vid = f"V{len(voices)+1}"
        # detect clef
        clefs = part.getElementsByClass(music21.clef.Clef)
        if clefs:
            sign = clefs[0].sign.lower()
            octch = clefs[0].octaveChange or 0
            clef_name = sign + (str(octch) if octch else "")
        else:
            clef_name = "treble"
        voices.append((vid, pname, clef_name, part))

    # Voice definitions
    for vid, pname, clef_name, _ in voices:
        abc_lines.append(f'V:{vid} name="{pname}" clef={clef_name}')

    # Helper: render a single measure to an ABC bar string
    def render_measure(measure):
        tokens = []
        for el in measure.notesAndRests:
            if isinstance(el, music21.note.Note):
                tok = pitch_to_abc(el.pitch) + duration_to_abc(el.duration, L_unit_ql)
            elif isinstance(el, music21.chord.Chord):
                notes = [pitch_to_abc(p) for p in el.pitches]
                tok = "[" + " ".join(notes) + "]" + duration_to_abc(el.duration, L_unit_ql)
            else:  # rest
                tok = "z" + duration_to_abc(el.duration, L_unit_ql)
            tokens.append(tok)
        return " ".join(tokens) + " |"

    # Voice bodies: group 4 measures per line
    for vid, _, _, part in voices:
        abc_lines.append(f"V:{vid}")
        measures = part.getElementsByClass(music21.stream.Measure)
        bars = [render_measure(m) for m in measures]
        for i in range(0, len(bars), 4):
            chunk = bars[i : i + 4]
            abc_lines.append(" ".join(chunk))

    return "\n".join(abc_lines)

def nwc_to_musicxml_jar(nwc_filepath, script_path='./jar/nwc2xml.sh', output_musicxml_filepath=None):
    """Wrapper for local shell script conversion to MusicXML."""
    if output_musicxml_filepath is None:
        fd, output_musicxml_filepath = tempfile.mkstemp(suffix='.musicxml')
        os.close(fd)
    log(f"[INFO] Running script: {script_path} {nwc_filepath} {output_musicxml_filepath}")
    subprocess.run([script_path, nwc_filepath, output_musicxml_filepath], check=True, text=True)
    return output_musicxml_filepath

def nwc_to_musicxml_service(nwc_filepath, service_url='https://nwc2musicxml.appspot.com/'):
    """Wrapper for online service conversion to MusicXML."""
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
    script_path='./jar/nwc2xml.sh',
    service_url='https://nwc2musicxml.appspot.com/',
    default_L_denom=8,
    simplicity_level='raw'
):
    log(f"[INFO] Starting NWC → ABC: {nwc_filepath}")
    if method == 'jar':
        xml_path = nwc_to_musicxml_jar(nwc_filepath, script_path)
    else:
        xml_path = nwc_to_musicxml_service(nwc_filepath, service_url)
    abc = musicxml_to_simplified_abc(xml_path, default_L_denom, simplicity_level)
    log(f"[INFO] Converted to ABC successfully")
    return abc

def simplified_abc_to_musicxml(abc_string, output_filepath=None):
    """Convert simplified ABC string into MusicXML via music21."""
    log(f"[INFO] Starting simplified ABC → MusicXML")
    if not abc_string.startswith("X:"):
        abc_string = "X:1\nM:4/4\nK:C\n" + abc_string
    score = music21.converter.parse(abc_string, format='abc')
    if output_filepath:
        score.write('musicxml', fp=output_filepath)
        return output_filepath
    return score.write('musicxml')

def abc_to_nwc(abc_string: str, output_filepath: str = None) -> str:
    """Convert ABC notation into NoteWorthy Composer ASCII (.nwctxt)."""
    log("[INFO] Starting ABC → NWCtxt conversion")
    if not abc_string.startswith("X:"):
        abc_string = "X:1\nM:4/4\nK:C\n" + abc_string
    score = music21.converter.parse(abc_string, format='abc')

    # Metadata
    title    = (score.metadata.title    or "Untitled").replace(":", "")
    composer = (score.metadata.composer or "").replace(":", "")
    ts = score.flat.getElementsByClass(music21.meter.TimeSignature)
    ks = score.flat.getElementsByClass(music21.key.KeySignature)
    meter = ts[0].ratioString if ts else "4/4"
    sharps = ks[0].sharps if ks else 0

    # Header
    lines = [
        "!NoteWorthyComposer(2.75)",
        "!Info",
        f"!SongInfo Title={title}",
        f"!SongInfo Composer={composer}" if composer else None,
        f"!Meter {meter}",
        f"!KeySharps {sharps}",
        ""
    ]
    lines = [l for l in lines if l is not None]

    # One staff per part
    for idx, part in enumerate(score.parts, start=1):
        pname = part.partName or f"Voice{idx}"
        clefs = part.getElementsByClass(music21.clef.Clef)
        clef = (clefs[0].sign + str(clefs[0].octaveChange or "")).lower() if clefs else "g2"

        lines += [
            f"[Staff {idx}]",
            f"!VoiceName {pname}",
            f"!Clef {clef}",
            ""
        ]

        for measure in part.getElementsByClass(music21.stream.Measure):
            tokens = []
            for el in measure.notesAndRests:
                d = el.duration.quarterLength
                if isinstance(el, music21.note.Note):
                    tokens.append(f"{el.pitch.nameWithOctave}/{d}")
                elif isinstance(el, music21.chord.Chord):
                    pcs = ".".join(p.nameWithOctave for p in el.pitches)
                    tokens.append(f"[{pcs}]/{d}")
                else:
                    tokens.append(f"R/{d}")
            lines.append("| " + " ".join(tokens))

        lines += [
            "[EndStaff]",
            ""
        ]

    text = "\n".join(lines)
    if output_filepath:
        Path(output_filepath).write_text(text, encoding="utf-8")
        log(f"[INFO] Written NWCtxt to {output_filepath}")
        return output_filepath
    return text

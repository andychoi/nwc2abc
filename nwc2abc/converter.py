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

def pitch_to_abc(m21_pitch: music21.pitch.Pitch, key_sig: music21.key.Key = None) -> str:
    """Convert a music21 Pitch into ABC pitch notation (with octave marks), omitting accidentals implied by the key."""
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


def musicxml_to_simplified_abc(
    musicxml_filepath: str,
    default_L_denom: int = 8,
    simplicity_level: str = "raw"
) -> str:
    from music21 import note, chord, stream, clef, meter, key, tempo, metadata

    score = music21.converter.parse(musicxml_filepath)
    meta = score.metadata or metadata.Metadata()
    title = meta.title.strip() if meta.title else "Untitled"
    composer = meta.composer.strip() if meta.composer else ""

    # Time and key signatures
    ts = score.flat.getElementsByClass(meter.TimeSignature)
    ks = score.flat.getElementsByClass(key.KeySignature)
    meter_str = ts[0].ratioString if ts else "4/4"
    key_obj = ks[0].asKey() if ks else key.Key("C")
    kname = key_obj.tonic.name.replace("-", "b")
    if key_obj.mode == "minor":
        kname += "m"

    # L unit
    L_unit_ql = 4.0 / default_L_denom

    abc_lines = [
        "X: 1",
        f"T: {title}",
        f"C: {composer}" if composer else None,
        f"M: {meter_str}",
        f"L: 1/{default_L_denom}"
    ]

    # Tempo
    tempos = score.flat.getElementsByClass(tempo.MetronomeMark)
    if tempos:
        bpm = tempos[0].number or 120
        beat_unit = tempos[0].referent.quarterLength if tempos[0].referent else 1.0
        if beat_unit == 1.0:
            beat_note = "1/4"
        elif beat_unit == 0.5:
            beat_note = "1/8"
        elif beat_unit == 2.0:
            beat_note = "1/2"
        else:
            beat_note = str(beat_unit)
        abc_lines.append(f"Q: {beat_note}={int(bpm)}")

    abc_lines.append(f"K: {kname}")
    abc_lines = [line for line in abc_lines if line]

    # Collect voices
    voices = []
    for idx, part in enumerate(score.parts):
        pname = part.partName or part.id or f"Voice{idx+1}"
        vid = f"V{idx+1}"

        # Clef detection from first measure
        measures = list(part.recurse().getElementsByClass(stream.Measure))
        clef_obj = None
        if measures and measures[0].clef:
            clef_obj = measures[0].clef
        else:
            clefs_found = list(measures[0].recurse().getElementsByClass(clef.Clef)) if measures else []
            clef_obj = clefs_found[0] if clefs_found else None
        clef_name = "bass" if (clef_obj and clef_obj.sign == "F") else "treble"

        voices.append((vid, pname, clef_name, measures))

    for vid, pname, clef_name, _ in voices:
        abc_lines.append(f'V:{vid} name="{pname}" clef={clef_name}')

    # Note → ABC converter with key context
    def pitch_to_abc(m21_pitch: music21.pitch.Pitch, key_sig: music21.key.Key) -> str:
        step = m21_pitch.step
        octave = m21_pitch.octave
        acc_token = ""

        if m21_pitch.accidental:
            show_acc = True
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

    # Duration converter
    def duration_to_abc(m21_duration: music21.duration.Duration) -> str:
        from fractions import Fraction
        try:
            frac = Fraction(m21_duration.quarterLength).limit_denominator(32)
            if frac == 0:
                return ""  # skip zero-length objects
            val = frac / Fraction(L_unit_ql)
            if val == 1:
                return ""
            if val < 1:
                return f"/{int(1 / val)}"
            if val.denominator == 1:
                return str(val.numerator)
            return f"{val.numerator}/{val.denominator}"
        except Exception as e:
            log(f"[WARN] Failed to convert duration {m21_duration}: {e}")
            return ""


    # Measure renderer
    def render_measure(measure):
        tokens = []
        for el in measure.recurse().notesAndRests:
            if isinstance(el, note.Note):
                tokens.append(pitch_to_abc(el.pitch, key_obj) + duration_to_abc(el.duration))
            elif isinstance(el, chord.Chord):
                notes = [pitch_to_abc(p, key_obj) for p in el.pitches]
                tokens.append("[" + " ".join(notes) + "]" + duration_to_abc(el.duration))
            elif isinstance(el, note.Rest):
                tokens.append("z" + duration_to_abc(el.duration))
        return " ".join(tokens) + " |"

    # Render each voice
    for vid, _, _, measures in voices:
        abc_lines.append(f"V:{vid}")
        bars = [render_measure(m) for m in measures]
        for i in range(0, len(bars), 4):
            abc_lines.append(" ".join(bars[i:i + 4]))

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

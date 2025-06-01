# musicxml2abc.py
"""
musicxml2abc.py

Usage:
    python musicxml2abc.py "converted/musicxml" [--force]

Description:
    Recursively converts .musicxml files to simplified ABC notation (.abc)
    using the music21 library.

    - Uses part names, meter, key, tempo, clefs, notes/rests
    - Skips existing .abc files unless --force is specified
    - Outputs to same directory as input file

Dependencies:
    pip install music21
"""

import argparse
from pathlib import Path
from fractions import Fraction
from music21 import converter, note, chord, stream, clef, meter, key, tempo, metadata


def pitch_to_abc(m21_pitch, key_sig=None):
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

    note_token = step.lower() + "'" * (octave - 5) if octave >= 5 else step.upper() + "," * max(0, 4 - octave)
    return acc_token + note_token


def duration_to_abc(m21_duration, L_unit_ql):
    try:
        frac = Fraction(m21_duration.quarterLength).limit_denominator(32)
        if frac == 0:
            return ""
        val = frac / Fraction(L_unit_ql)
        if val == 1:
            return ""
        elif val < 1:
            return f"/{int(1 / val)}"
        elif val.denominator == 1:
            return str(val.numerator)
        else:
            return f"{val.numerator}/{val.denominator}"
    except Exception:
        return ""


def render_measure(measure, key_obj, L_unit_ql):
    tokens = []
    for el in measure.recurse().notesAndRests:
        if isinstance(el, note.Note):
            tokens.append(pitch_to_abc(el.pitch, key_obj) + duration_to_abc(el.duration, L_unit_ql))
        elif isinstance(el, chord.Chord):
            notes = [pitch_to_abc(p, key_obj) for p in el.pitches]
            tokens.append("[" + " ".join(notes) + "]" + duration_to_abc(el.duration, L_unit_ql))
        elif isinstance(el, note.Rest):
            tokens.append("z" + duration_to_abc(el.duration, L_unit_ql))
    return " ".join(tokens) + " |"


def musicxml_to_abc(musicxml_path: Path, default_L_denom=8) -> str:
    score = converter.parse(str(musicxml_path))
    meta = score.metadata or metadata.Metadata()
    title = meta.title.strip() if meta.title else "Untitled"
    composer = meta.composer.strip() if meta.composer else ""

    ts = score.flat.getElementsByClass(meter.TimeSignature)
    ks = score.flat.getElementsByClass(key.KeySignature)
    meter_str = ts[0].ratioString if ts else "4/4"
    key_obj = ks[0].asKey() if ks else key.Key("C")
    kname = key_obj.tonic.name.replace("-", "b")
    if key_obj.mode == "minor":
        kname += "m"

    L_unit_ql = 4.0 / default_L_denom

    abc_lines = [
        "X: 1",
        f"T: {title}",
        f"C: {composer}" if composer else None,
        f"M: {meter_str}",
        f"L: 1/{default_L_denom}"
    ]

    tempos = score.flat.getElementsByClass(tempo.MetronomeMark)
    if tempos:
        bpm = tempos[0].number or 120
        beat_unit = tempos[0].referent.quarterLength if tempos[0].referent else 1.0
        beat_note = {1.0: "1/4", 0.5: "1/8", 2.0: "1/2"}.get(beat_unit, str(beat_unit))
        abc_lines.append(f"Q: {beat_note}={int(bpm)}")

    abc_lines.append(f"K: {kname}")
    abc_lines = [line for line in abc_lines if line]

    voices = []
    for idx, part in enumerate(score.parts):
        pname = part.partName or part.id or f"Voice{idx+1}"
        vid = f"V{idx+1}"
        measures = list(part.recurse().getElementsByClass(stream.Measure))
        clef_obj = measures[0].clef if measures and measures[0].clef else None
        clefs_found = measures[0].recurse().getElementsByClass(clef.Clef) if measures else []
        clef_obj = clef_obj or (clefs_found[0] if clefs_found else None)
        clef_name = "bass" if (clef_obj and clef_obj.sign == "F") else "treble"
        voices.append((vid, pname, clef_name, measures))

    for vid, pname, clef_name, measures in voices:
        abc_lines.append(f'V:{vid} name="{pname}" clef={clef_name}')
        abc_lines.append(f"V:{vid}")
        bars = [render_measure(m, key_obj, L_unit_ql) for m in measures]
        for i in range(0, len(bars), 4):
            abc_lines.append(" ".join(bars[i:i+4]))

    return "\n".join(abc_lines)


def convert_folder(root_folder: Path, force: bool):
    musicxml_files = sorted(root_folder.rglob("*.musicxml"))
    if not musicxml_files:
        print(f"⚠️  No .musicxml files found in: {root_folder}")
        return

    output_root = root_folder.parent / "abc"
    output_root.mkdir(parents=True, exist_ok=True)

    for file in musicxml_files:
        try:
            relative_path = file.relative_to(root_folder)
            abc_file = output_root / relative_path.with_suffix(".abc")
            abc_file.parent.mkdir(parents=True, exist_ok=True)

            if abc_file.exists() and not force:
                print(f"⏭️  Skipping (exists): {abc_file.name}")
                continue

            print(f"🎼 Converting: {file.name}")
            abc_text = musicxml_to_abc(file)
            abc_file.write_text(abc_text, encoding="utf-8")
            print(f"✅ Saved: {abc_file.name}")

        except Exception as e:
            print(f"❌ Error converting {file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .musicxml files to ABC notation.")
    parser.add_argument("folder", type=Path, help="Root folder containing .musicxml files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .abc files")
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"❌ Folder not found: {args.folder}")
        exit(1)

    convert_folder(args.folder, args.force)

# Python Project Summary

## `convert-all.py`
```python
# convert-all.py
import argparse
import subprocess
import sys
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("./converted")

STEP_TITLES = {
    "1": "🎼 Step 1: Converting .nwc → .nwctxt",
    "2": "🛠️ Step 2: Fixing Korean mojibake in .nwctxt → .nwctxt-fixed",
    "3": "🧪 Step 3: Applying general fixes to .nwctxt",
    "4": "🎶 Step 4: Converting .nwctxt → .musicxml",
    "5": "🧹 Step 5: Organizing .musicxml by layout",
    "6": "🪄 Step 6: Converting .musicxml → .abc",
}

ALL_STEPS = "12346"  # step 5 is optional

def show_steps():
    print("\n📋 Available Steps:")
    for key in sorted(STEP_TITLES):
        print(f"{key}. {STEP_TITLES[key]}")
    print("\n💡 To run all steps:   python convert_all.py nwcoriginal --steps all")
    print("💡 To run some steps:  python convert_all.py nwcoriginal --steps 135")
    print("💡 Use --outdir to specify the output folder (default: ./converted)")

def run_command(cmd, shell=False):
    print(f"🔧 Running: {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, shell=shell)
    if result.returncode != 0:
        print(f"❌ Error: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run music conversion pipeline steps.")
    parser.add_argument("InputFolder", help="Initial input folder (e.g. .nwc or preprocessed)")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT, help="Output root folder (intermediate + final results)")
    parser.add_argument("--steps", default="", help="Steps to run (e.g., 135 or 'all')")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    args = parser.parse_args()

    input_dir = Path(args.InputFolder).expanduser().resolve()
    output_root = Path(args.outdir).expanduser().resolve()

    nwctxt_dir = output_root / "nwctxt"
    nwctxt_fixed_dir = output_root / "nwctxt-fixed"
    musicxml_dir = output_root / "musicxml"

    steps = args.steps.lower()
    if steps == "all":
        steps = ALL_STEPS

    if not steps:
        show_steps()
        return

    if "1" in steps:
        print(f"\n{STEP_TITLES['1']}")
        cmd = [
            "python", str(TOOL_PATH / "nwc2nwctxt.py"),
            str(input_dir),
            str(nwctxt_dir)
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)

    if "2" in steps:
        print(f"\n{STEP_TITLES['2']}")
        cmd = [
            "python", str(TOOL_PATH / "fix-korean.py"),
            str(nwctxt_dir),
            str(nwctxt_fixed_dir)
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)

    if "3" in steps:
        print(f"\n{STEP_TITLES['3']}")
        cmd = [
            "python", str(TOOL_PATH / "nwctxt_fix.py"),
            str(nwctxt_fixed_dir)
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)

    if "4" in steps:
        print(f"\n{STEP_TITLES['4']}")
        cmd = [
            "python", str(TOOL_PATH / "nwctxt2musicxml.py"),
            str(nwctxt_fixed_dir),
            str(musicxml_dir)
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)

    if "5" in steps:
        print(f"\n{STEP_TITLES['5']}")
        cmd = [
            "python", str(TOOL_PATH / "musicxml_organize.py"),
            str(musicxml_dir)
        ]
        run_command(cmd)

    if "6" in steps:
        print(f"\n{STEP_TITLES['6']}")
        cmd = [
            "python", str(TOOL_PATH / "musicxml2abc.py"),
            str(musicxml_dir)
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)

    print(f"\n✅ Done: Steps [{args.steps}] completed.")

if __name__ == "__main__":
    main()
```

## `fix-korean.py`
```python
import sys
import os
import re
from pathlib import Path
from shutil import copy2
import unicodedata

def is_mojibake(text: str) -> bool:
    # Heuristic: if >50% characters are � (replacement) or look Latin-like but invalid Korean
    count = len(text)
    bad = sum(1 for ch in text if ch in '�?' or unicodedata.category(ch).startswith('C'))
    return (count > 0 and bad / count > 0.5)

def recover_lyrics_line(mojibake_text, wrong_encoding='windows-1252', correct_encoding='euc-kr'):
    if not is_mojibake(mojibake_text):
        return mojibake_text  # Looks okay, skip fixing

    try:
        raw_bytes = mojibake_text.encode(wrong_encoding, errors='replace')
        return raw_bytes.decode(correct_encoding, errors='replace')
    except Exception:
        return mojibake_text


def recover_nwctxt_file(input_path, output_path):
    with open(input_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    recovered = []
    text_attr_pattern = re.compile(r'^(.*Text:)\"(.*)\"$')
    info_field_pattern = re.compile(r'^(!SongInfo\s+\w+)=([^\n]+)$')
    typeface_pattern = re.compile(r'^(.*Typeface:)\"(.*)\"(.*)$')

    changes = 0
    for line in lines:
        stripped = line.strip()

        # Pattern 1: Text fields like |Text|Text:"..."
        match = text_attr_pattern.match(stripped)
        if match:
            prefix, content = match.groups()
            fixed = recover_lyrics_line(content)
            recovered.append(f'{prefix}"{fixed}"\n')
            changes += 1
            continue

        # Pattern 2: !SongInfo fields like !SongInfo Title=...
        match = info_field_pattern.match(stripped)
        if match:
            field, content = match.groups()
            fixed = recover_lyrics_line(content)
            recovered.append(f'{field}={fixed}\n')
            changes += 1
            continue

        # Pattern 3: Font face lines like Typeface:"±¼¸²ü"
        match = typeface_pattern.match(stripped)
        if match:
            prefix, content, suffix = match.groups()
            fixed = recover_lyrics_line(content)

            # Replace mojibake with a known-good Hangul font if still broken
            if "?" in fixed or "�" in fixed or not re.search(r'[가-힣]', fixed):
                fixed = "Arial" # Malgun Gothic"   # 굴림체 fallback to safe Hangul font

            recovered.append(f'{prefix}"{fixed}"{suffix}\n')
            changes += 1
            continue

        recovered.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(recovered)

    print(f"✅ {output_path} ({changes} field(s) fixed)")

def should_process(source: Path, target: Path, force=False) -> bool:
    if force:
        return True
    if not target.exists():
        return True
    return source.stat().st_mtime > target.stat().st_mtime

def process_folder(source_dir, dest_dir, force=False):
    source_dir = Path(source_dir).resolve()
    dest_dir = Path(dest_dir).resolve()

    files = list(source_dir.rglob("*.nwctxt"))
    if not files:
        print("No .nwctxt files found.")
        return

    for source_file in files:
        relative = source_file.relative_to(source_dir)
        target_file = dest_dir / relative

        if should_process(source_file, target_file, force=force):
            recover_nwctxt_file(source_file, target_file)
        else:
            print(f"⏩ Skipped (up-to-date): {relative}")

# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recover Korean text from mojibake .nwctxt files.")
    parser.add_argument("source", help="Source folder with .nwctxt files")
    parser.add_argument("destination", nargs="?", help="Destination folder (default: <source>-fixed)")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if target is newer")

    args = parser.parse_args()

    dest_folder = args.destination
    if not dest_folder:
        dest_folder = f"{args.source}-fixed"

    process_folder(args.source, dest_folder, force=args.force)

```

## `musicxml2abc.py`
```python
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

```

## `musicxml_organize.py`
```python
# convert/musicxml_organize.py
import argparse
import shutil
import sys
import re
from pathlib import Path
from music21 import converter

# Add path to use infer_staff_roles
from nwctxt_fix import infer_staff_roles

def detect_layout_postfix(xml_path: Path) -> str:
    try:
        # Convert to nwctxt-like format string from MusicXML
        score = converter.parse(str(xml_path))
        lines = []
        for part in score.parts:
            part_id = part.id if hasattr(part, 'id') else ""
            part_name = part.partName or ""
            clef = part.getElementsByClass('Clef')[0].sign if part.getElementsByClass('Clef') else ""
            brace = 'Piano' in (part_name or '').lower()
            lines.append(f'|AddStaff|Name:"{part_name}"')
            lines.append(f'|Label:"{part_name}"')
            lines.append(f'|Clef|Type:{clef}')
            if brace:
                lines.append('|StaffProperties|Brace')
            # Fake instrument patch
            lines.append(f'|StaffInstrument|Patch:0')

        content = "\n".join(lines)
        _, postfix, *_ = infer_staff_roles(content)
        return postfix or "Unknown"
    except Exception as e:
        print(f"⚠️  Failed to infer layout from {xml_path.name}: {e}")
        return "Unknown"

def organize_musicxml(root_path: Path, output_path: Path):
    print(f"\n📂 Organizing .musicxml files from:\n  {root_path}")
    if root_path == output_path:
        print(f"📁 reorganizing in-place within:\n  {output_path}\n")
    else:
        print(f"📁 into:\n  {output_path}\n")
    files = list(root_path.rglob("*.musicxml"))
    if not files:
        print(f"⚠️  No .musicxml files found under {root_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        try:
            layout = detect_layout_postfix(file)
            target_dir = output_path / layout
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / file.name
            if file.resolve() != dest_path.resolve():
                print(f"📦 Moving '{file.name}' → '{layout}/'")
                shutil.move(str(file), str(dest_path))
        except Exception as e:
            print(f"⚠️  Error processing '{file}': {e}")

    print("\n✅ Organization complete.")

def main():
    parser = argparse.ArgumentParser(description="Organize .musicxml files by vocal/instrument layout.")
    parser.add_argument("RootFolder", help="Root folder containing .musicxml files")
    parser.add_argument("OutputFolder", nargs="?", help="Optional output folder (default: RootFolder)")
    args = parser.parse_args()

    root_path = Path(args.RootFolder).expanduser().resolve()
    if not root_path.exists():
        print(f"❌ ERROR: Root folder not found: {root_path}")
        sys.exit(1)

    output_path = (
        Path(args.OutputFolder).expanduser().resolve()
        if args.OutputFolder else root_path
    )

    organize_musicxml(root_path, output_path)

if __name__ == "__main__":
    main()
```

## `nwc2nwctxt.py`
```python
#!/usr/bin/env python3
import ctypes
import subprocess
from pathlib import Path
import argparse

def get_short_path_name(long_path: str) -> str:
    """Convert a long Windows path to its short (8.3) form."""
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetShortPathNameW(str(long_path), buf, 260)
    return buf.value

def convert_nwc_to_nwctxt(
    source_dir: Path,
    dest_dir: Path,
    nwc2_exe_path: Path,
    force: bool = False
):
    if not source_dir.exists():
        print(f"❌ Source folder not found: {source_dir}")
        return

    if not nwc2_exe_path.exists():
        print(f"❌ Noteworthy CLI tool not found: {nwc2_exe_path}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    nwc_files = list(source_dir.rglob("*.nwc"))
    if not nwc_files:
        print("⚠️ No .nwc files found.")
        return

    for file in nwc_files:
        rel_path = file.relative_to(source_dir)
        output_file = dest_dir / rel_path.with_suffix(".nwctxt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not force:
            if file.stat().st_mtime <= output_file.stat().st_mtime:
                print(f"⏭️  Skipping (up to date): {rel_path}")
                continue

        print(f"🎼 Converting: {file} → {output_file}")
        try:
            short_input = get_short_path_name(str(file))
            output_dir_short = get_short_path_name(str(output_file.parent))
            short_output = str(Path(output_dir_short) / output_file.name)

            cmd = f'"{nwc2_exe_path}" -convert "{short_input}" "{short_output}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0 and output_file.exists():
                print(f"✅ Success: {output_file}")
            else:
                print(f"⚠️  Failed: {file.name}")
                if result.stderr:
                    print(f"    STDERR: {result.stderr.strip()}")
                else:
                    print("    No stderr output. Try running manually to debug.")

        except Exception as e:
            print(f"❌ ERROR converting {file.name}: {e}")

    print("\n✅ Batch NWC → NWCTXT conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .nwc files to .nwctxt using nwc2.exe")
    parser.add_argument("source", type=Path, help="Folder with .nwc files")
    parser.add_argument("dest", type=Path, nargs="?", default=None,
                        help="Destination folder for .nwctxt files (default: ./converted/nwctxt)")
    parser.add_argument("--force", action="store_true", help="Force re-convert even if target is newer")
    parser.add_argument("--nwc2exe", type=Path, default=Path(r"C:\Program Files (x86)\Noteworthy Software\NoteWorthy Composer 2\nwc2.exe"),
                        help="Path to nwc2.exe")

    args = parser.parse_args()

    # Use default destination if not provided
    dest_path = args.dest or Path("converted") / "nwctxt"

    convert_nwc_to_nwctxt(args.source, dest_path, args.nwc2exe, args.force)

```

## `nwctxt2musicxml.py`
```python
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

def resolve_path(path_str):
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p

def ensure_directory(path: Path):
    if not path.exists():
        print(f"Creating output folder: {path}")
        path.mkdir(parents=True, exist_ok=True)

def is_up_to_date(src: Path, dst: Path):
    return dst.exists() and src.stat().st_mtime <= dst.stat().st_mtime

def run_java_converter(jar_path: Path, input_file: Path, output_file: Path):
    cmd = [
        "java",
        "-Dfile.encoding=UTF-8",
        "-cp", str(jar_path),
        "fr.lasconic.nwc2musicxml.convert.Nwc2MusicXML",
        str(input_file),
        str(output_file)
    ]
    result = subprocess.run(cmd, shell=False)
    return result.returncode == 0 and output_file.exists()

def convert_files(source_dir: Path, dest_dir: Path, jar_path: Path, force: bool):
    files = list(source_dir.rglob("*.nwctxt"))

    for file in files:
        try:
            rel_path = file.relative_to(source_dir)
        except ValueError:
            print(f"⚠️ Skipping unrelative file: {file}")
            continue

        output_path = dest_dir / rel_path.with_suffix(".musicxml")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not force and is_up_to_date(file, output_path):
            print(f"⏩ Skipping (up to date): {rel_path}")
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".nwctxt", delete=False) as tmp_in, \
                 tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp_out:

                shutil.copy2(file, tmp_in.name)
                tmp_in_path = Path(tmp_in.name)
                tmp_out_path = Path(tmp_out.name)

            success = run_java_converter(jar_path, tmp_in_path, tmp_out_path)

            if success:
                shutil.move(tmp_out_path, output_path)
                print(f"✅ Success: {output_path}")
            else:
                print(f"❌ Conversion failed: {file}", file=sys.stderr)

        finally:
            if tmp_in_path.exists():
                tmp_in_path.unlink(missing_ok=True)
            if tmp_out_path.exists():
                tmp_out_path.unlink(missing_ok=True)

    print("\n✅ Batch MusicXML conversion complete.")


def main():
    parser = argparse.ArgumentParser(description="Convert .nwctxt files to .musicxml using nwc2musicxml.jar")
    parser.add_argument("SourceDir", help="Directory with .nwctxt files")
    parser.add_argument("DestDir", nargs="?", default="musicxml", help="Destination folder for .musicxml output")
    parser.add_argument("--force", action="store_true", help="Force conversion even if output is up to date")
    args = parser.parse_args()

    source_dir = resolve_path(args.SourceDir)
    dest_dir = Path(args.DestDir).expanduser().resolve()
    ensure_directory(dest_dir)

    script_dir = Path(__file__).resolve().parent
    jar_path = script_dir / "nwc2musicxml.jar"

    if not jar_path.exists():
        print(f"❌ ERROR: Converter not found at {jar_path}", file=sys.stderr)
        sys.exit(1)

    convert_files(source_dir, dest_dir, jar_path, args.force)


if __name__ == "__main__":
    main()
```

## `nwctxt_fix.py`
```python

# convert/nwctxt_fix.py
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# General MIDI Patch Names
GM_PATCH_NAMES = {
    0: "Acoustic Grand Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    13: "Xylophone", 24: "Acoustic Guitar (nylon)", 32: "Acoustic Bass",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    46: "Orchestral Harp", 48: "String Ensemble 1", 49: "String Ensemble 2",
    50: "SynthStrings 1", 51: "SynthStrings 2", 52: "Choir Aahs", 53: "Voice Oohs",
    54: "Synth Voice", 56: "Trumpet", 64: "Soprano Sax", 65: "Alto Sax",
    66: "Tenor Sax", 67: "Baritone Sax", 68: "Oboe", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute"
}

ALT_PATCHES = {
    "alt1": {
        "Soprano 1": 52, "Soprano 2": 53, "Soprano": 52,
        "Alto": 41,
        "Tenor": 66,
        "Bass": 32,
    },
    "alt2": {
        "Soprano 1": 72, "Soprano 2": 73, "Soprano": 72,
        "Alto": 65,
        "Tenor": 42,
        "Bass": 67,
    }
}

ROLE_KEYWORDS = {
    "s": "Soprano", "sop": "Soprano", "soprano": "Soprano", "소프라노": "Soprano",
    "a": "Alto", "alt": "Alto", "alto": "Alto", "알토": "Alto",
    "t": "Tenor", "ten": "Tenor", "tenor": "Tenor", "테너": "Tenor",
    "b": "Bass", "bas": "Bass", "bass": "Bass", "베이스": "Bass",
    "sa": "SA", "tb": "TB",
    "rh": "Piano-RH", "right": "Piano-RH", "right hand": "Piano-RH",
    "lh": "Piano-LH", "left": "Piano-LH", "left hand": "Piano-LH",
    "피아노r": "Piano-RH", "피아노l": "Piano-LH", "피아노": "Piano",
    "violin": "Violin", "cello": "Cello", "flute": "Flute"
}
VCF = {"Violin", "Cello", "Flute"}

def normalize(text: str) -> str:
    return text.strip().lower()

def abbreviate_label(role: str) -> str:
    if role.startswith("Soprano"):
        parts = role.split()
        return "S" if len(parts) == 1 else f"S{parts[1]}"
    if role == "Alto": return "A"
    if role == "Alto 2": return "A2"
    if role == "Tenor": return "T"
    if role == "Bass": return "B"
    if role in {"SA", "TB"}: return role
    if role == "Piano-RH": return "PRH"
    if role == "Piano-LH": return "PLH"
    return role[:3]

def parse_staff_blocks(lines: List[str]) -> List[List[str]]:
    blocks, current = [], []
    for ln in lines:
        if ln.startswith("|AddStaff"):
            if current:
                blocks.append(current)
            current = [ln]
        elif current and (ln.startswith("|StaffProperties") or ln.startswith("|Label:") or "|Clef|Type:" in ln or ln.startswith("|StaffInstrument")):
            current.append(ln)
    if current:
        blocks.append(current)
    return blocks

def classify_by_keyword(name: str, clef: str = "", brace: bool = False) -> str:
    key = normalize(name)
    for k, role in ROLE_KEYWORDS.items():
        if key == k or re.search(rf"\b{k}\b", key):
            return role
    if brace:
        return "Piano-RH"
    return ""

# ... [keep original import and global constants unchanged] ...

def infer_staff_roles(content: str, return_details=False) -> Tuple[Dict[int, str], str, List[str], List[str], List[str], List[str]]:
    lines = content.splitlines()
    blocks = parse_staff_blocks(lines)
    n = len(blocks)

    names, labels, clefs, instruments, braces = [], [], [], [], []
    for block in blocks:
        name = label = clef = patch = ""
        brace = False
        for ln in block:
            if ln.startswith("|AddStaff"):
                m = re.search(r'Name:"([^"]+)"', ln)
                if m: name = m.group(1)
                m2 = re.search(r'Label:"([^"]+)"', ln)
                if m2: label = m2.group(1)
            elif ln.startswith("|Label:"):
                m = re.search(r'"([^"]+)"', ln)
                if m: label = m.group(1)
            elif "|Clef|Type:" in ln:
                m = re.search(r'Type:([^\s|]+)', ln)
                if m: clef = m.group(1)
            elif ln.startswith("|StaffProperties") and "Brace" in ln:
                brace = True
            elif ln.startswith("|StaffInstrument"):
                m = re.search(r'Patch:(\d+)', ln)
                if m: patch = f"Patch:{m.group(1)}"
        names.append(name)
        labels.append(label)
        clefs.append(clef)
        instruments.append(patch or "-")
        braces.append(brace)

    piano_rh_idxs = [i for i, br in enumerate(braces) if br]
    piano_idxs = set()
    for i in piano_rh_idxs:
        piano_idxs.add(i)
        if i + 1 < n:
            piano_idxs.add(i + 1)

    voice_idxs = [i for i in range(n) if i not in piano_idxs]
    role_map: Dict[int, str] = {}

    treble_idxs = [i for i in voice_idxs if clefs[i].lower() == "treble"]
    bass_idxs = [i for i in voice_idxs if clefs[i].lower() == "bass"]

    treble_roles = []
    if len(treble_idxs) == 1:
        treble_roles = ["Soprano"]
    elif len(treble_idxs) == 2:
        treble_roles = ["Soprano", "Alto"]
    elif len(treble_idxs) == 3:
        treble_roles = ["Soprano 1", "Soprano 2", "Alto"]
    elif len(treble_idxs) >= 4:
        treble_roles = ["Soprano 1", "Soprano 2", "Alto", "Alto 2"]

    for i, idx in enumerate(treble_idxs):
        role_map[idx] = treble_roles[i] if i < len(treble_roles) else "Soprano"

    for idx in bass_idxs:
        role_map[idx] = "Tenor" if "Tenor" not in role_map.values() else "Bass"

    for i in sorted(piano_idxs):
        role_map[i] = "Piano-RH" if i in piano_rh_idxs else "Piano-LH"

    for i in range(n):
        if i not in role_map:
            inferred = classify_by_keyword(names[i], clefs[i], braces[i])
            role_map[i] = inferred or "Extra"

    all_roles = set(role_map.values())
    base_roles = {r.split()[0] for r in all_roles}
    voices_set = base_roles & {"Soprano", "Alto", "Tenor", "Bass", "SA", "TB"}
    piano_set = all_roles & {"Piano-RH", "Piano-LH"}
    extras = base_roles & VCF

    postfix = ""
    
    if {"Soprano", "Alto", "Tenor", "Bass"} <= voices_set:
        postfix = "SATB_P" if piano_set else "SATB"
    elif {"SA", "TB"} <= voices_set:
        postfix = "SA_TB_P" if piano_set else "SA_TB"
    elif voices_set:
        if len(voices_set) >= 2:
            postfix = "Choral_P" if piano_set else "Choral"
        else:
            postfix = list(voices_set)[0]
            if piano_set:
                postfix += "_P"
    elif piano_set:
        postfix = "P"

    if extras:
        postfix += "_" + "".join(sorted(e[0] for e in extras))

    return role_map, postfix, names, labels, clefs, instruments

def apply_updates(content: str, role_map: Dict[int, str], alt: str = None) -> str:
    lines = content.splitlines()
    new_lines: List[str] = []
    staff_idx = -1
    for ln in lines:
        if ln.startswith("|AddStaff"):
            staff_idx += 1
            role = role_map.get(staff_idx)
            if role:
                abbr = abbreviate_label(role)
                ln = re.sub(r'(\|AddStaff\|Name:)"[^"]+"', rf'\1"{role}"', ln)
                if "|Label:" in ln:
                    ln = re.sub(r'(\|Label:)"[^"]+"', rf'\1"{abbr}"', ln)
                else:
                    ln = ln.replace("|AddStaff", f'|AddStaff|Label:"{abbr}"')
        elif ln.startswith("|Label:"):
            abbr = abbreviate_label(role_map.get(staff_idx, ""))
            ln = f'|Label:"{abbr}"'
        elif ln.startswith("|StaffInstrument") and alt:
            role = role_map.get(staff_idx)
            patch_map = ALT_PATCHES.get(alt, {})
            if role in patch_map:
                new_patch = patch_map[role]
                ln = re.sub(r'Patch:\d+', f'Patch:{new_patch}', ln)
        new_lines.append(ln)
    return "\n".join(new_lines)

def rename_file_with_postfix(file: Path, postfix: str) -> Path:
    base = re.sub(r'__[^.]+$', '', file.stem)
    new_name = f"{base}__{postfix}{file.suffix}"
    new_path = file.with_name(new_name)
    if new_path != file:
        file.rename(new_path)
        print(f"✅ Renamed: {file.name} → {new_path.name}")
    else:
        print(f"✅ Name unchanged: {file.name}")
    return new_path

def process_folder(folder: Path, rename: bool = True, test_mode: bool = False, alt_patch: str = None, organize: bool = True):
    if not folder.exists():
        print(f"📁 Creating missing folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    files = list(folder.glob("*.nwctxt"))
    if not files:
        print(f"⚠️  No .nwctxt files found in {folder}")
        return

    for file in files:
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
            role_map, postfix, names, labels, clefs, instruments = infer_staff_roles(content, return_details=True)

            if test_mode:
                print(f"\n🔍 Simulating: {file.name}")
                for i, role in role_map.items():
                    name, label, clef, instr = names[i], labels[i], clefs[i], instruments[i]
                    patch_num = int(instr.split(":")[1]) if instr.startswith("Patch:") else None
                    patch_name = GM_PATCH_NAMES.get(patch_num, "Unknown") if patch_num is not None else "-"
                    target_patch = ALT_PATCHES.get(alt_patch, {}).get(role)
                    target_patch_name = GM_PATCH_NAMES.get(target_patch, "Unknown") if target_patch else "-"
                    if target_patch and patch_num != target_patch:
                        patch_display = f"{instr} ({patch_name}) → Patch:{target_patch} ({target_patch_name})"
                    else:
                        patch_display = f"{instr} ({patch_name})"
                    print(f"  STAFF {i}: {name} / {label} / {clef} / {patch_display} → {role} ({abbreviate_label(role)})")
                continue

            updated = apply_updates(content, role_map, alt_patch)
            file.write_text(updated, encoding="utf-8")

            # Optionally rename
            if rename and postfix:
                file = rename_file_with_postfix(file, postfix)
            else:
                print(f"✅ Updated (no renaming): {file.name}")

            # Optionally organize into subfolders
            if organize and postfix:
                target_dir = folder / postfix
                target_dir.mkdir(exist_ok=True)
                target_path = target_dir / file.name
                if file.resolve() != target_path.resolve():
                    print(f"📦 Moving '{file.name}' → '{postfix}/'")
                    file.replace(target_path)

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nwctxt_fix.py <folder_path> [--no-rename] [--no-organize] [--test] [--alt=alt1|alt2]")
        sys.exit(1)

    target = Path(sys.argv[1])
    do_rename = "--no-rename" not in sys.argv
    is_test = "--test" in sys.argv
    do_organize = "--no-organize" not in sys.argv
    alt_patch = "alt1"
    for arg in sys.argv[2:]:
        if arg.startswith("--alt="):
            alt_patch = arg.split("=", 1)[1]

    process_folder(
        target,
        rename=do_rename,
        test_mode=is_test,
        alt_patch=alt_patch,
        organize=do_organize
    )

```


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
# musicxml_organize.py
import argparse
import shutil
import sys
from pathlib import Path
from music21 import converter
import re

def sanitize_composer(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[._]', ' ', name)
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if not name:
        name = 'unknown'
    # Title-case each word
    name = ' '.join(word.capitalize() for word in name.split())
    return name

def extract_composer(path: Path) -> str:
    try:
        score = converter.parse(str(path))
        composer = (score.metadata.composer or '').strip()
        return sanitize_composer(composer)
    except Exception:
        return 'Unknown'

def organize_musicxml(root_path: Path, output_path: Path):
    print(f"\n📂 Organizing .musicxml files from:\n  {root_path}")
    print(f"📁 into:\n  {output_path}\n")
    files = list(root_path.rglob("*.musicxml"))
    if not files:
        print(f"⚠️  No .musicxml files found under {root_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        try:
            composer = extract_composer(file)
            target_dir = output_path / composer
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / file.name
            if file.resolve() != dest_path.resolve():
                print(f"📦 Moving '{file.name}' → '{composer}/'")
                shutil.move(str(file), str(dest_path))
        except Exception as e:
            print(f"⚠️  Error processing '{file}': {e}")

    print("\n✅ Organization complete.")


def main():
    parser = argparse.ArgumentParser(description="Organize .musicxml files by composer metadata.")
    parser.add_argument("RootFolder", help="Root folder containing .musicxml files")
    parser.add_argument("OutputFolder", nargs="?", help="Optional output folder (default: RootFolder-organized)")
    args = parser.parse_args()

    root_path = Path(args.RootFolder).expanduser().resolve()
    if not root_path.exists():
        print(f"❌ ERROR: Root folder not found: {root_path}")
        sys.exit(1)

    output_path = (
        Path(args.OutputFolder).expanduser().resolve()
        if args.OutputFolder else Path(f"{root_path}-organized")
    )

    organize_musicxml(root_path, output_path)


if __name__ == "__main__":
    main()
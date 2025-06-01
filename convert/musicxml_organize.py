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
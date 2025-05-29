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

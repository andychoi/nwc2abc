import ctypes
import subprocess
import hashlib
import argparse
import csv
from pathlib import Path
from collections import defaultdict


def get_short_path_name(long_path: str) -> str:
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetShortPathNameW(str(long_path), buf, 260)
    return buf.value


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert_nwc_to_nwctxt(source_dir: Path, dest_dir: Path, nwc2_exe_path: Path, force: bool = False):
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

    nwctxt_to_nwc = {}

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
                nwctxt_to_nwc[output_file.resolve()] = file.resolve()
            else:
                print(f"⚠️  Failed: {file.name}")
                if result.stderr:
                    print(f"    STDERR: {result.stderr.strip()}")
                else:
                    print("    No stderr output. Try running manually to debug.")
        except Exception as e:
            print(f"❌ ERROR converting {file.name}: {e}")

    print("\n🔍 Checking for duplicate .nwctxt files...")
    deduplicate_nwctxt_files(dest_dir, nwctxt_to_nwc)
    print("\n✅ Conversion and deduplication complete.")


def deduplicate_nwctxt_files(dest_dir: Path, nwctxt_to_nwc: dict):
    hash_to_files = defaultdict(list)
    for file in dest_dir.rglob("*.nwctxt"):
        if "_duplicates" in str(file):
            continue  # skip already moved files
        try:
            file_hash = hash_file(file)
            hash_to_files[file_hash].append(file.resolve())
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    duplicates_dir = dest_dir / "_duplicates"
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    log_path = duplicates_dir / "duplicate_log.csv"

    with open(log_path, mode='w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Duplicate", "Kept", "Original_NWC_Timestamp", "Duplicate_NWC_Timestamp"])

        for files in hash_to_files.values():
            if len(files) > 1:
                latest = max(
                    files,
                    key=lambda f: nwctxt_to_nwc.get(f, f).stat().st_mtime
                )
                print(f"\n🗃️ Duplicate group ({len(files)}): keeping → {latest.relative_to(dest_dir)}")

                for f in files:
                    if f == latest:
                        continue

                    source_nwc = nwctxt_to_nwc.get(f)
                    latest_nwc = nwctxt_to_nwc.get(latest)

                    rel_path = f.relative_to(dest_dir)
                    target_path = duplicates_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    print(f"  📦 Moving duplicate → {target_path.relative_to(dest_dir)}")
                    f.rename(target_path)

                    writer.writerow([
                        str(target_path.relative_to(dest_dir)),
                        str(latest.relative_to(dest_dir)),
                        source_nwc.stat().st_mtime if source_nwc else "",
                        latest_nwc.stat().st_mtime if latest_nwc else ""
                    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .nwc files to .nwctxt using nwc2.exe")
    parser.add_argument("source", type=Path, help="Folder with .nwc files")
    parser.add_argument("dest", type=Path, nargs="?", default=None,
                        help="Destination folder for .nwctxt files (default: ./converted/nwctxt)")
    parser.add_argument("--force", action="store_true", help="Force re-convert even if target is newer")
    parser.add_argument("--nwc2exe", type=Path, default=Path(r"C:\Program Files (x86)\Noteworthy Software\NoteWorthy Composer 2\nwc2.exe"),
                        help="Path to nwc2.exe")

    args = parser.parse_args()
    dest_path = args.dest or Path("converted") / "nwctxt"
    convert_nwc_to_nwctxt(args.source, dest_path, args.nwc2exe, args.force)

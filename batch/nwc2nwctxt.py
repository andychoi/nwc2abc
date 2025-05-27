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

import argparse
import subprocess
import sys
from pathlib import Path

# Configuration
TOOL_PATH = Path(__file__).resolve().parent
NWC_SOURCE_DIR = "nwcoriginal"
OUTPUT_ROOT_DIR = Path("./converted")

STEP_TITLES = {
    "1": "🎼 Step 1: Converting .nwc → .nwctxt",
    "2": "🛠️  Step 2: Fixing Korean mojibake in .nwctxt → .nwctxt-fixed",
    "3": "🧪 Step 3: Applying general fixes to .nwctxt",
    "4": "🎶 Step 4: Converting .nwctxt → .musicxml",
    "5": "🧹 Step 5: Organizing .musicxml by composer",
    "6": "🪄 Step 6: Converting .musicxml → .abc",
}

ALL_STEPS = "123456"


def show_steps():
    print("\n📋 Available Steps:")
    for key in sorted(STEP_TITLES):
        print(f"{key}. {STEP_TITLES[key]}")
    print("\n💡 To run all steps:   python convert_all.py --steps all")
    print("💡 To run some steps:  python convert_all.py --steps 135")


def run_command(cmd, shell=False):
    print(f"🔧 Running: {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, shell=shell)
    if result.returncode != 0:
        print(f"❌ Error: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run music conversion pipeline steps.")
    parser.add_argument("--nwcdir", default=NWC_SOURCE_DIR)
    parser.add_argument("--outdir", default=OUTPUT_ROOT_DIR)
    parser.add_argument("--steps", default="", help="Steps to run (e.g., 135 or 'all')")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    args = parser.parse_args()

    nwc_dir = Path(args.nwcdir).expanduser().resolve()
    output_root = Path(args.outdir).expanduser().resolve()

    nwctxt_dir = nwc_dir
    nwctxt_fixed_dir = Path(f"{nwctxt_dir}-fixed").resolve()
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
            str(nwc_dir),
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
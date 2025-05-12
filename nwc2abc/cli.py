# nwc2abc/cli.py
import argparse
from pathlib import Path
from .converter import nwc_to_simplified_abc, log

def main():
    parser = argparse.ArgumentParser(description="Convert NWCtxt to ABC notation")
    parser.add_argument("input_file", help="Path to NWCtxt file")
    parser.add_argument("--output", "-o", type=str, help="Output ABC file (if not given, print to stdout)")
    parser.add_argument("--method", choices=["jar", "service"], default="jar", help="Conversion method")
    parser.add_argument("--script", default="./jar/nwc2xml.sh", help="Path to nwc2xml.sh script (only for jar method)")
    parser.add_argument("--level", choices=["raw", "medium", "simple"], default="raw", help="ABC simplicity level")

    args = parser.parse_args()

    # run conversion
    abc = nwc_to_simplified_abc(
        args.input_file,
        method=args.method,
        script_path=args.script,
        simplicity_level=args.level
    )

    # output handling
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(abc, encoding="utf-8")
        log(f"[INFO] ABC written to file: {output_path}")
    else:
        print(abc)

if __name__ == "__main__":
    main()
